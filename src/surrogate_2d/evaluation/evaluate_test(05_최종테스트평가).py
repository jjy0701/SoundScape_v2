"""(05) 학습에 사용하지 않은 test 15개 최종 평가.

04번 결과 폴더의 최적 MLP와 Ridge 계수를 다시 불러와 test split만 평가합니다.
test 정답은 지표 계산/시각화에만 사용하며 재학습, 정규화 통계 재계산,
모델 선택에는 사용하지 않습니다.

기본 실행(04번 결과 폴더 자동 선택)::

    python "evaluate_test(05_최종테스트평가).py"

특정 결과 폴더를 지정하려면::

    python "evaluate_test(05_최종테스트평가).py" \
        --baseline-result "results/baseline(04_기초모델학습)_YYYYMMDD_HHMMSS_ffffff"

저장 결과에는 모델별 test 오차, 샘플별 오차, 예측 NPY 묶음, 비교 heatmap,
그리고 시뮬레이션을 다시 돌리지 않은 상태에서의 추론 시간 측정이 포함됩니다.
"""

import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    raise SystemExit(
        "PyTorch가 필요합니다. 현재 가상환경에서 "
        "python -m pip install torch 를 실행하세요."
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRID_SIZE = 20
N_RECEIVERS = GRID_SIZE * GRID_SIZE
GEOM = [
    "speaker_x",
    "speaker_y",
    "obstacle_x",
    "obstacle_y",
    "obstacle_w",
    "obstacle_l",
]
FEATURES = GEOM + [
    "receiver_x",
    "receiver_y",
    "log10_distance",
    "log10_min_scene_distance",
]


def sha(path: Path) -> str:
    """파일 SHA-256. 03번 분할 manifest와 파일이 같은지 확인합니다."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def canonical_id(value: str) -> str:
    """metadata의 sample_id가 숫자/ sample_숫자 어느 쪽이어도 정규화합니다."""

    text = str(value).strip()
    if text.startswith("sample_"):
        text = text.removeprefix("sample_")
    return f"sample_{int(text):04d}"


def read_metadata(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {canonical_id(row["sample_id"]): row for row in csv.DictReader(handle)}


def validate_splits(dataset: Path):
    """03번 manifest와 현재 metadata/ID 목록을 검증하고 test 목록을 반환합니다."""

    manifest_path = dataset / "splits" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = dataset / "metadata.csv"
    if sha(metadata_path) != manifest["metadata_sha256"]:
        raise ValueError("03번 분할 이후 metadata.csv가 변경됐습니다")

    split_names = ["train", "validation", "test"]
    splits = manifest["splits"]
    all_ids = sum([splits[name] for name in split_names], [])
    if [len(splits[name]) for name in split_names] != [70, 15, 15]:
        raise ValueError("train/validation/test 수량이 70/15/15가 아닙니다")
    if len(all_ids) != 100 or len(set(all_ids)) != 100:
        raise ValueError("분할 ID가 100개가 아니거나 중복됐습니다")
    for name in split_names:
        ids_path = dataset / "splits" / f"{name}_ids.txt"
        listed = ids_path.read_text(encoding="utf-8").splitlines()
        if listed != splits[name]:
            raise ValueError(f"{name}_ids.txt와 split_manifest.json의 순서/ID가 다릅니다")

    rows = read_metadata(metadata_path)
    if set(rows) != set(all_ids):
        raise ValueError("metadata와 split manifest의 sample_id 집합이 다릅니다")
    near = {
        item["sample_id"]: bool(item["near_source"])
        for item in manifest["samples"]
    }
    if set(near) != set(all_ids):
        raise ValueError("manifest의 near_source 목록이 전체 ID를 포함하지 않습니다")
    return manifest_path, manifest, rows, near


def build_scene(dataset: Path, sid: str, row: dict, near: bool, expected_hash: str):
    """04번과 동일한 20x20 입력/마스크를 만들고 test NPY hash를 확인합니다."""

    npy_path = dataset / "npy" / f"{sid}.npy"
    if not npy_path.is_file():
        raise FileNotFoundError(f"{sid}: NPY 파일이 없습니다: {npy_path}")
    if sha(npy_path) != expected_hash:
        raise ValueError(f"{sid}: 03번 분할 이후 NPY가 변경됐습니다")

    x_axis = np.linspace(0.5, float(row["room_x"]) - 0.5, GRID_SIZE)
    y_axis = np.linspace(0.5, float(row["room_y"]) - 0.5, GRID_SIZE)
    X, Y = np.meshgrid(x_axis, y_axis)
    sx, sy, ox, oy, w, length = [float(row[key]) for key in GEOM]
    invalid = (
        (X >= ox)
        & (X <= ox + w)
        & (Y >= oy)
        & (Y <= oy + length)
    )
    valid = (~invalid).reshape(-1)

    target_grid = np.load(npy_path, allow_pickle=False)
    if target_grid.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"{sid}: target shape가 {target_grid.shape}입니다")
    if not np.array_equal(np.isnan(target_grid), invalid):
        raise ValueError(f"{sid}: NPY NaN 장애물 mask가 metadata와 다릅니다")
    if not np.isfinite(target_grid[~invalid]).all():
        raise ValueError(f"{sid}: 유효 receiver에 NaN/무한대가 있습니다")
    if not valid.any() or abs(float(target_grid[~invalid].max())) > 1e-5:
        raise ValueError(f"{sid}: 유효 영역의 최대값 0 dB 정규화가 아닙니다")

    distance = np.hypot(X - sx, Y - sy).reshape(-1)
    dmin = float(distance[valid].min())
    if dmin <= 0.0:
        raise ValueError(f"{sid}: 음원과 receiver가 같은 위치입니다")
    if bool(dmin < 0.1) != near:
        raise ValueError(f"{sid}: 최소거리와 near_source 표시가 불일치합니다")

    feature_matrix = np.column_stack(
        [np.full(N_RECEIVERS, float(row[key])) for key in GEOM]
        + [
            X.ravel(),
            Y.ravel(),
            np.log10(np.maximum(distance, 1e-8)),
            np.full(N_RECEIVERS, np.log10(dmin)),
        ]
    ).astype(np.float32)
    target = np.where(invalid, 0.0, target_grid).reshape(-1).astype(np.float32)
    distance_prediction = (
        -20.0 * np.log10(np.maximum(distance, 1e-8) / dmin)
    ).astype(np.float32)
    return {
        "id": sid,
        "x": feature_matrix,
        "y": target,
        "mask": valid,
        "near": near,
        "X": X,
        "Y": Y,
        "geom": [sx, sy, ox, oy, w, length],
        "distance_prediction": distance_prediction,
    }


def make_mlp(input_dim: int, hidden_sizes):
    layers = []
    previous = input_dim
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(previous, int(hidden)), nn.ReLU()])
        previous = int(hidden)
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch has no weights_only keyword
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"MLP checkpoint 형식이 올바르지 않습니다: {path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("mlp_best.pt 안에 config가 없습니다")
    return checkpoint, config


def find_baseline_result(results_root: Path) -> Path:
    candidates = [
        path
        for path in results_root.glob("baseline(04_기초모델학습)_*")
        if (path / "mlp_best.pt").is_file()
        and (path / "ridge.npz").is_file()
        and (path / "config.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "04번 baseline 결과 폴더를 찾지 못했습니다. "
            "--baseline-result로 폴더를 직접 지정하세요."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def evaluate_scenes(scenes, predictions):
    if not np.isfinite(predictions).all():
        raise ValueError("예측에 NaN/무한대가 발생했습니다")
    rows = []
    for scene, prediction in zip(scenes, predictions):
        error = prediction[scene["mask"]] - scene["y"][scene["mask"]]
        rows.append(
            {
                "sample_id": scene["id"],
                "near_source": scene["near"],
                "mae_db": float(np.abs(error).mean()),
                "rmse_db": float(np.sqrt(np.mean(error**2))),
                "max_abs_error_db": float(np.abs(error).max()),
            }
        )
    groups = {}
    for name, selected in [
        ("all", rows),
        ("near", [row for row in rows if row["near_source"]]),
        ("non_near", [row for row in rows if not row["near_source"]]),
    ]:
        groups[name] = {
            "n": len(selected),
            "mae_db": float(np.mean([row["mae_db"] for row in selected]))
            if selected
            else None,
            "mean_scene_rmse_db": float(
                np.mean([row["rmse_db"] for row in selected])
            )
            if selected
            else None,
            "max_abs_error_db": float(
                max(row["max_abs_error_db"] for row in selected)
            )
            if selected
            else None,
        }
    return groups, rows


def stack_prediction_grid(scenes, predictions):
    return np.stack(
        [
            np.where(scene["mask"], prediction, np.nan).reshape(GRID_SIZE, GRID_SIZE)
            for scene, prediction in zip(scenes, predictions)
        ]
    ).astype(np.float32)


def time_numpy(fn, n_scenes: int, repeats: int = 20):
    fn()  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = (time.perf_counter() - start) / repeats
    return {
        "repeats": repeats,
        "mean_total_ms": elapsed * 1000.0,
        "mean_per_scene_ms": elapsed * 1000.0 / n_scenes,
        "mean_per_receiver_us": elapsed * 1e6 / (n_scenes * N_RECEIVERS),
    }


def time_mlp(model, x, device, repeats: int = 20):
    model.eval()
    with torch.inference_mode():
        model(x)  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) / repeats
    n_scenes = int(x.shape[0])
    return {
        "repeats": repeats,
        "mean_total_ms": elapsed * 1000.0,
        "mean_per_scene_ms": elapsed * 1000.0 / n_scenes,
        "mean_per_receiver_us": elapsed * 1e6 / (n_scenes * N_RECEIVERS),
    }


def plot_comparisons(out: Path, scenes, predictions, max_plots: int):
    """test 첫 샘플부터 고정 개수만 그려 과도한 파일 생성을 막습니다."""

    selected = scenes[: max(0, min(max_plots, len(scenes)))]
    level_vmin = -50.0
    level_vmax = 0.0
    error_limit = 5.0
    level_cmap = plt.get_cmap("viridis").copy()
    level_cmap.set_bad("white")
    error_cmap = plt.get_cmap("coolwarm").copy()
    error_cmap.set_bad("white")

    for index, scene in enumerate(selected):
        fig, axes = plt.subplots(1, 5, figsize=(19, 5), layout="constrained")
        arrays = [
            scene["y"],
            predictions["distance"][index],
            predictions["ridge"][index],
            predictions["mlp"][index],
            predictions["mlp"][index] - scene["y"],
        ]
        titles = [
            "Simulation target",
            "Distance baseline",
            "Ridge",
            "MLP",
            "MLP - simulation",
        ]
        for column, (axis, array, title) in enumerate(zip(axes, arrays, titles)):
            data = np.where(scene["mask"], array, np.nan).reshape(GRID_SIZE, GRID_SIZE)
            image = axis.pcolormesh(
                scene["X"],
                scene["Y"],
                data,
                shading="auto",
                cmap=error_cmap if column == 4 else level_cmap,
                vmin=-error_limit if column == 4 else level_vmin,
                vmax=error_limit if column == 4 else level_vmax,
            )
            sx, sy, ox, oy, w, length = scene["geom"]
            axis.add_patch(
                plt.Rectangle(
                    (ox, oy), w, length, facecolor="white", edgecolor="black", linewidth=1.5
                )
            )
            axis.scatter(sx, sy, marker="*", color="red", edgecolors="black", s=80)
            axis.set(title=title, xlabel="x (m)", ylabel="y (m)", aspect="equal")
            axis.set_xlim(0, float(scene["X"].max() + 0.5))
            axis.set_ylim(0, float(scene["Y"].max() + 0.5))
            fig.colorbar(image, ax=axis, label="dB", shrink=0.72, extend="both")
        fig.suptitle(
            f"{scene['id']} | test | levels fixed -50..0 dB; error fixed ±5 dB"
        )
        fig.savefig(out / f"{scene['id']}(최종예측비교).png", dpi=130)
        plt.close(fig)
    return {
        "selected_test_ids": [scene["id"] for scene in selected],
        "level_vmin": level_vmin,
        "level_vmax": level_vmax,
        "error_vmin": -error_limit,
        "error_vmax": error_limit,
        "note": "색상 범위는 시각화용 고정값이며 예측/정답을 clipping하지 않았습니다.",
    }


def main():
    project_root = next(parent for parent in Path(__file__).resolve().parents
                        if (parent / "configs" / "classroom").is_dir()
                        and (parent / "data").is_dir()
                        and (parent / "results").is_dir())
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=project_root / "data" / "surrogate_2d" / "dataset")
    parser.add_argument("--output-root", type=Path, default=project_root / "results" / "surrogate_2d")
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=None,
        help="04번 결과 폴더. 생략하면 가장 최근 폴더를 선택합니다.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--plot-count", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=20)
    args = parser.parse_args()
    if args.plot_count < 0 or args.timing_repeats < 1:
        parser.error("plot-count는 0 이상, timing-repeats는 양수여야 합니다")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA를 사용할 수 없습니다. --device cpu 로 실행하세요")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    dataset = args.dataset.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path, manifest, metadata, near = validate_splits(dataset)
    baseline = (
        args.baseline_result.resolve()
        if args.baseline_result is not None
        else find_baseline_result(output_root)
    )
    if not baseline.is_dir():
        raise FileNotFoundError(f"baseline 결과 폴더가 없습니다: {baseline}")
    checkpoint, model_config = load_checkpoint(baseline / "mlp_best.pt")
    checkpoint_features = model_config.get("features")
    if checkpoint_features != FEATURES:
        raise ValueError(
            "04번과 05번의 feature 정의가 다릅니다: "
            f"{checkpoint_features!r} != {FEATURES!r}"
        )
    if model_config.get("split_manifest_sha256") not in (None, sha(manifest_path)):
        raise ValueError("04번 모델이 현재 03번 split manifest로 학습되지 않았습니다")

    x_mean = np.asarray(model_config["x_mean"], dtype=np.float32)
    x_std = np.asarray(model_config["x_std"], dtype=np.float32)
    y_mean = float(model_config["y_mean"])
    y_std = float(model_config["y_std"])
    if x_mean.shape != (len(FEATURES),) or x_std.shape != (len(FEATURES),):
        raise ValueError("checkpoint의 x_mean/x_std 길이가 feature 수와 다릅니다")
    if not np.isfinite(x_mean).all() or not np.isfinite(x_std).all() or np.any(x_std <= 0):
        raise ValueError("checkpoint의 x_mean/x_std가 올바르지 않습니다")
    if not np.isfinite([y_mean, y_std]).all() or y_std <= 0:
        raise ValueError("checkpoint의 y_mean/y_std가 올바르지 않습니다")

    hidden_sizes = model_config.get("hidden_sizes", [64, 64])
    model = make_mlp(len(FEATURES), hidden_sizes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with np.load(baseline / "ridge.npz", allow_pickle=False) as ridge_file:
        ridge_coefficients = np.asarray(ridge_file["coefficients"], dtype=np.float64)
        ridge_x_mean = np.asarray(ridge_file["x_mean"], dtype=np.float32)
        ridge_x_std = np.asarray(ridge_file["x_std"], dtype=np.float32)
        ridge_y_mean = float(ridge_file["y_mean"])
        ridge_y_std = float(ridge_file["y_std"])
    if ridge_coefficients.shape != (len(FEATURES) + 1,):
        raise ValueError("ridge.npz coefficients shape가 올바르지 않습니다")
    if not (
        np.allclose(ridge_x_mean, x_mean)
        and np.allclose(ridge_x_std, x_std)
        and np.isclose(ridge_y_mean, y_mean)
        and np.isclose(ridge_y_std, y_std)
    ):
        raise ValueError("MLP checkpoint와 Ridge의 train 정규화 통계가 다릅니다")

    test_ids = manifest["splits"]["test"]
    test_scenes = [
        build_scene(dataset, sid, metadata[sid], near[sid], manifest["npy_sha256"][sid])
        for sid in test_ids
    ]
    raw_x = np.stack([scene["x"] for scene in test_scenes]).astype(np.float32)
    standardized_x = (raw_x - x_mean[None, None, :]) / x_std[None, None, :]
    target = np.stack([scene["y"] for scene in test_scenes]).astype(np.float32)

    predictions = {
        "distance": np.stack([scene["distance_prediction"] for scene in test_scenes]),
    }
    ridge_input = np.concatenate(
        [standardized_x.astype(np.float64), np.ones((len(test_scenes), N_RECEIVERS, 1))],
        axis=2,
    )
    predictions["ridge"] = (
        ridge_input @ ridge_coefficients * ridge_y_std + ridge_y_mean
    ).astype(np.float32)

    mlp_input = torch.from_numpy(standardized_x).to(device)
    with torch.inference_mode():
        mlp_output = model(mlp_input).squeeze(-1).cpu().numpy()
    predictions["mlp"] = (mlp_output * y_std + y_mean).astype(np.float32)

    groups = {}
    details = []
    for name in ["distance", "ridge", "mlp"]:
        groups[name], rows = evaluate_scenes(test_scenes, predictions[name])
        details.extend({"model": name, **row} for row in rows)

    timing = {
        "device": str(device),
        "simulation_time_measured": False,
        "simulation_time_note": "이 스크립트는 Pyroomacoustics를 다시 실행하지 않습니다. 아래 값은 15개 test 장면의 모델 추론 시간입니다.",
        "scope": "메모리에 올라온 표준화 feature의 예측 계산만 측정; NPY 읽기·heatmap 저장·Pyroomacoustics 재계산 시간은 제외",
        "distance": time_numpy(
            lambda: np.stack([scene["distance_prediction"] for scene in test_scenes]),
            len(test_scenes),
            args.timing_repeats,
        ),
        "ridge": time_numpy(
            lambda: ridge_input @ ridge_coefficients * ridge_y_std + ridge_y_mean,
            len(test_scenes),
            args.timing_repeats,
        ),
        "mlp": time_mlp(model, mlp_input, device, args.timing_repeats),
    }

    out = output_root / (
        "test(05_최종테스트평가)_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    out.mkdir(parents=True, exist_ok=False)
    np.savez(
        out / "test_target_and_mask.npz",
        sample_ids=np.array(test_ids),
        target=np.stack([np.where(scene["mask"], scene["y"], np.nan).reshape(GRID_SIZE, GRID_SIZE) for scene in test_scenes]),
        valid_mask=np.stack([scene["mask"].reshape(GRID_SIZE, GRID_SIZE) for scene in test_scenes]),
    )
    for name, prediction in predictions.items():
        np.savez(
            out / f"test_prediction_{name}.npz",
            sample_ids=np.array(test_ids),
            prediction=stack_prediction_grid(test_scenes, prediction),
        )
    with (out / "test(03_샘플별오차).csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    write_json(
        out / "metrics(02_최종테스트평가).json",
        {
            "test_evaluated": True,
            "test_targets_loaded_for_evaluation_only": True,
            "training_or_tuning_on_test": False,
            "baseline_result": str(baseline),
            "split_manifest_sha256": sha(manifest_path),
            "sample_ids": test_ids,
            "model_results": groups,
            "timing": timing,
            "plot_scale": plot_comparisons(out, test_scenes, predictions, args.plot_count),
            "metric_definition": "각 장면의 유효 receiver만 사용한 MAE/RMSE를 먼저 계산한 뒤 장면 평균",
            "target_definition": "RIR energy -> 10 log10 -> 장면별 유효 receiver 최대값을 0 dB로 정규화; obstacle 내부는 NaN",
        },
    )
    write_json(
        out / "config(01_평가설정).json",
        {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha(Path(__file__)),
            "baseline_result": str(baseline),
            "checkpoint_best_epoch": checkpoint.get("best_epoch"),
            "checkpoint_validation_rmse_db": checkpoint.get("validation_rmse_db"),
            "features": FEATURES,
            "device": str(device),
            "test_count": len(test_scenes),
            "test_targets_loaded_for_evaluation_only": True,
            "model_selection_split": "validation (04번에서 완료)",
            "test_split_used_for": ["최종 오차 계산", "예측 저장", "heatmap 시각화"],
            "test_split_not_used_for": ["학습", "정규화 통계 계산", "hyperparameter/model 선택"],
        },
    )
    write_json(out / "timing(04_추론시간).json", timing)

    print("(05) 최종 test 평가 v1", flush=True)
    print(f"baseline: {baseline}", flush=True)
    print("test=15 / 학습·튜닝에 test 미사용 / device=" + str(device), flush=True)
    for name in ["distance", "ridge", "mlp"]:
        result = groups[name]["all"]
        print(
            f"{name}: MAE={result['mae_db']:.4f} dB / "
            f"mean-scene RMSE={result['mean_scene_rmse_db']:.4f} dB",
            flush=True,
        )
    print(
        "MLP 추론: "
        f"{timing['mlp']['mean_total_ms']:.3f} ms/15 scenes "
        f"({timing['mlp']['mean_per_scene_ms']:.3f} ms/scene)",
        flush=True,
    )
    print(f"결과 폴더: {out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)

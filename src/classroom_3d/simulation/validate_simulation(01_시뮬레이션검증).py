"""(01) 내부 장애물의 직접음 차폐 검증.

sound_scape 폴더에 저장한 뒤 기존 pyroomacoustics 환경에서 실행:
    python "validate_simulation(01_시뮬레이션검증).py"

기존 dataset은 읽거나 수정하지 않습니다. results 아래 실행별 폴더에 저장합니다.
PASS는 이 고정 장면의 직접음 검증만 통과했다는 뜻입니다.
반사 경로, 반사 차수 수렴, 실제 음향 정확도까지 보증하지 않습니다.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 16000
SOURCE = np.array([2.0, 5.0], dtype=np.float32)
RECEIVERS = np.array([[6.0, 5.0], [2.0, 7.0]], dtype=np.float32)
NAMES = ["blocked_probe", "clear_probe"]
# 장애물 좌하단 (3.5, 4), 크기 1 x 2 m. CW 순서.
OBSTACLE = [[3.5, 4.0], [3.5, 6.0], [4.5, 6.0], [4.5, 4.0]]
# 직접 경로가 다시 열리는 이동 장면.
MOVED = [[3.5, 1.0], [3.5, 3.0], [4.5, 3.0], [4.5, 1.0]]
OUTER = [[0.0, 0.0], [8.0, 0.0], [8.0, 10.0], [0.0, 10.0]]
CASES = {"empty": None, "blocked": OBSTACLE, "moved": MOVED}


def find_project_root(start):
    start = Path(start).resolve()
    start = start.parent if start.is_file() else start
    for candidate in (start, *start.parents):
        if ((candidate / "configs" / "classroom").is_dir()
                and (candidate / "configs" / "simulation").is_dir()
                and (candidate / "results").is_dir()):
            return candidate
    raise RuntimeError("sound_scape 프로젝트 루트를 찾을 수 없습니다.")


def make_walls(pra, corners, alpha, prefix):
    """사용자가 제공한 생성 코드와 같은 wall_factory 호출 방식."""
    walls = []
    for i in range(4):
        points = np.array([corners[i], corners[(i + 1) % 4]],
                          dtype=np.float32).T
        walls.append(pra.wall_factory(
            points, np.array([alpha], dtype=np.float32),
            np.array([0.0], dtype=np.float32), f"{prefix}_{i}"))
    return walls


def simulate(pra, obstacle):
    """수음점별 가시성을 먼저 검사. 경로가 없는 0차 ISM은 명시적 0 응답.

    장애물 위치로 결과를 강제하지 않습니다. PRA의 visibility를 사용합니다.
    예상과 다르게 차단/통과되면 최종 검증은 FAIL입니다.
    """
    rirs, diagnostics = [], []
    for i, receiver in enumerate(RECEIVERS):
        walls = make_walls(pra, OUTER, 0.2, "outer")
        if obstacle is not None:
            walls.extend(make_walls(pra, obstacle, 0.1, "obstacle"))
        room = pra.Room(walls, fs=FS, max_order=0, ray_tracing=False)
        room.add_source(SOURCE)
        room.add_microphone_array(pra.MicrophoneArray(receiver.reshape(2, 1), FS))
        room.image_source_model()
        if room.visibility is None or len(room.visibility) != 1:
            raise RuntimeError("PRA의 source별 visibility 구조를 확인할 수 없습니다.")
        visibility = np.asarray(room.visibility[0])
        orders = np.asarray(room.sources[0].orders)
        if orders.ndim != 1 or visibility.shape != (1, orders.size):
            raise RuntimeError(f"예상 밖 visibility/차수 shape: {visibility.shape}, {orders.shape}")
        if not np.isin(visibility, [0, 1]).all() or np.any(orders != 0):
            raise RuntimeError("0차 직접음 검증에 맞지 않는 가시성/반사 차수입니다.")
        visible_count = int(np.count_nonzero(visibility[0]))
        if visible_count == 0:
            # PRA의 경로 탐색 결과가 0개인 경우에만 정의하는 빈 경로합입니다.
            # compute_rir가 계산한 파형으로 오인하지 않도록 provenance를 저장합니다.
            rir = np.zeros(1, dtype=np.float64)
            origin = "zero_response_from_no_visible_paths"
            print(f"  {NAMES[i]}: visible_paths=0 → 0차 경로 없음; RIR 생성 생략", flush=True)
        else:
            room.compute_rir()  # 다른 계산 오류는 0으로 바꾸지 않고 그대로 보고합니다.
            rir = np.asarray(room.rir[0][0], dtype=np.float64)
            if rir.size == 0:
                raise RuntimeError("가시 경로가 있는데 RIR이 비어 있습니다.")
            origin = "pra_compute_rir"
            print(f"  {NAMES[i]}: visible_paths={visible_count}", flush=True)
        rirs.append(rir)
        diagnostics.append(dict(visible_paths=visible_count,
                                visibility=visibility.tolist(),
                                image_orders=orders.tolist(), rir_origin=origin))
    return rirs, diagnostics


def energy_of(rir):
    # 빈 RIR도 기록합니다. 차폐된 0차 경로에서 빈 배열은 허용합니다.
    if rir.ndim != 1 or not np.isfinite(rir).all():
        raise ValueError("RIR 차원 오류 또는 NaN/무한대가 있습니다.")
    energy = float(np.dot(rir, rir))
    if not np.isfinite(energy):
        raise ValueError("RIR 에너지 계산이 유한하지 않습니다.")
    return energy


def checks_for(energies):
    base = energies["empty"]
    checks = {"empty_both_nonzero": bool(np.all(base > 0))}
    if checks["empty_both_nonzero"]:
        # 1e-8은 물리적 차폐 성능 기준이 아닌 수치적 0 판정용 비율입니다.
        checks["blocked_direct_removed"] = bool(
            energies["blocked"][0] / base[0] <= 1e-8)
        checks["clear_probe_unchanged"] = bool(np.isclose(
            energies["blocked"][1] / base[1], 1.0, rtol=1e-4, atol=1e-8))
        checks["moved_direct_restored"] = bool(np.allclose(
            energies["moved"] / base, 1.0, rtol=1e-4, atol=1e-8))
    else:
        checks.update(blocked_direct_removed=False,
                      clear_probe_unchanged=False, moved_direct_restored=False)
    return checks


def save_figures(out, rirs, diagnostics):
    # 글꼴 설치 차이를 피하기 위해 그림 내부 라벨은 영어로 표시합니다.
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
    for ax, (case, obstacle) in zip(axes, CASES.items()):
        ax.add_patch(plt.Rectangle((0, 0), 8, 10, fill=False, color="black"))
        if obstacle is not None:
            ax.add_patch(plt.Polygon(obstacle, facecolor="0.8", edgecolor="black"))
        ax.scatter(*SOURCE, marker="*", s=150, color="red", label="Source")
        for i, point in enumerate(RECEIVERS):
            ax.scatter(*point, label=NAMES[i])
            ax.plot([SOURCE[0], point[0]], [SOURCE[1], point[1]], "--", alpha=0.4)
        ax.set(title=case, xlabel="x (m)", ylabel="y (m)", xlim=(-.3, 8.3),
               ylim=(-.3, 10.3), aspect="equal")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "geometry(01_배치확인).png", dpi=150)
    plt.close(fig)

    # 같은 receiver의 세 장면을 같은 진폭 축으로 비교합니다. 정규화하지 않습니다.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey="row")
    for col, case in enumerate(CASES):
        for i in range(2):
            rir = rirs[case][i]
            ax = axes[i, col]
            if diagnostics[case][i]["visible_paths"] == 0:
                ax.axhline(0, linewidth=1)
                ax.text(.5, .8, "No visible direct path\nZero response by empty path sum",
                        transform=ax.transAxes, ha="center", fontsize=8)
            else:
                ax.plot(np.arange(rir.size) / FS * 1000, rir, linewidth=1)
            ax.set(title=f"{case}: {NAMES[i]}", xlabel="Time (ms)",
                   ylabel="RIR amplitude (uncalibrated)")
            ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "rir(02_직접음파형비교).png", dpi=150)
    plt.close(fig)


def main():
    try:
        import pyroomacoustics as pra
    except ImportError:
        raise SystemExit("pyroomacoustics가 있는 기존 프로젝트 Python 환경에서 실행하세요.")
    version = getattr(pra, "__version__", "unknown")
    print("검증 코드 v2: 경로 가시성 확인 후 RIR 계산", flush=True)
    print(f"pyroomacoustics 버전: {version}", flush=True)
    if version != "0.10.1":
        print("참고: 기존 데이터 생성 버전(0.10.1)과 다릅니다.", flush=True)
    project_root = find_project_root(__file__)
    out = (project_root / "results" / "classroom_3d" /
           ("validation(01_시뮬레이션검증)_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    out.mkdir(parents=True, exist_ok=False)
    config = dict(pyroomacoustics_version=version, fs=FS, max_order=0,
                  ray_tracing=False, source=SOURCE.tolist(),
                  receivers=RECEIVERS.tolist(), receiver_names=NAMES,
                  outer=OUTER, cases=CASES, wall_alpha=0.2, obstacle_alpha=0.1,
                  energy_floor_applied=False, validator_version=2,
                  no_path_policy="zero response from empty path sum; explicitly recorded",
                  scope="direct-path smoke test, not full acoustic validation")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rirs, energies, diagnostics, rows = {}, {}, {}, []
    for case, obstacle in CASES.items():
        print(f"장면 계산 중: {case}", flush=True)
        try:
            rirs[case], diagnostics[case] = simulate(pra, obstacle)
            energies[case] = np.array([energy_of(r) for r in rirs[case]])
        except Exception as exc:
            (out / "error.txt").write_text(f"{case}: {type(exc).__name__}: {exc}", encoding="utf-8")
            print(f"계산 실패. 오류 위치: {out}", flush=True)
            raise
        np.savez(out / f"rir_{case}.npz",
                 **{NAMES[i]: rirs[case][i] for i in range(2)}, fs=FS,
                 visible_paths=[d["visible_paths"] for d in diagnostics[case]],
                 rir_origin=[d["rir_origin"] for d in diagnostics[case]])
        for i, energy in enumerate(energies[case]):
            base = energies["empty"][i]
            ratio = float(energy / base) if base > 0 else None
            rows.append(dict(case=case, receiver=NAMES[i],
                             rir_length=len(rirs[case][i]), energy=float(energy),
                             energy_ratio_to_empty=ratio,
                             visible_paths=diagnostics[case][i]["visible_paths"],
                             rir_origin=diagnostics[case][i]["rir_origin"]))
            print(f"  {NAMES[i]}: energy={energy:.8e}, ratio={ratio}", flush=True)
    with (out / "energy(03_에너지비교).csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checks = checks_for(energies)
    expected = {"empty": [1, 1], "blocked": [0, 1], "moved": [1, 1]}
    for case, counts in expected.items():
        checks[f"visibility_{case}"] = [d["visible_paths"] for d in diagnostics[case]] == counts
    passed = all(checks.values())
    (out / "summary(04_검증결과).json").write_text(
        json.dumps({"direct_path_pass": passed, "checks": checks,
                    "diagnostics": diagnostics}, indent=2), encoding="utf-8")
    save_figures(out, rirs, diagnostics)
    for name, result in checks.items():
        print(f"{'PASS' if result else 'FAIL'}: {name}")
    print("\n직접음 검증:", "PASS" if passed else "FAIL — 학습 전에 원인 확인 필요")
    print("이 결과는 반사 차수 수렴이나 실제 공간의 정확도를 검증하지 않습니다.")
    print(f"결과 폴더: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""(12) 11번 heatmap을 다시 시각화하고 좌우 추가 측정 지점을 선정합니다.

RIR은 다시 계산하지 않습니다. 최신 11번 결과의 NPZ를 읽어 공통 색상 범위를
데이터에 맞게 조정하고, P02와 같은 y=5.0 m 선에서 좌우 상쇄 후보를 찾습니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path


VERSION = "1.0"
CONFIG_NAME = "classroom_config(07_강의실측정조건).json"
NPZ_NAME = "heatmap(01_상대레벨원시배열).npz"


def find_project_root(start):
    for first in (start.resolve(), Path.cwd().resolve()):
        for root in (first, *first.parents):
            if (root / "configs/classroom" / CONFIG_NAME).is_file() and (root / "results/classroom_3d").is_dir():
                return root
    raise FileNotFoundError("sound_scape 프로젝트 루트를 찾지 못했습니다.")


def find_latest_stage11(root):
    parent = root / "results/classroom_3d"
    candidates = [path for path in parent.glob("classroom_heatmap(11_강의실예측분포)_*") if (path / NPZ_NAME).is_file()]
    if not candidates:
        raise FileNotFoundError("11번 결과 폴더와 원시 NPZ를 찾지 못했습니다.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def interpolate_horizontal(array, x, y, target_y, np):
    return np.array([np.interp(target_y, y, array[:, column]) for column in range(len(x))])


def interpolate_point(array, x, y, px, py, np):
    row = interpolate_horizontal(array, x, y, py, np)
    return float(np.interp(px, x, row))


def select_side_minimum(x, values, anchor_x, side, clearance, wall_clearance, room_width, np):
    if side == "left":
        allowed = (x <= anchor_x - clearance) & (x >= wall_clearance)
    else:
        allowed = (x >= anchor_x + clearance) & (x <= room_width - wall_clearance)
    indices = np.flatnonzero(allowed)
    if len(indices) == 0:
        raise ValueError(f"{side} 측정 후보를 선택할 수 없습니다. 격자나 여유 거리 설정을 확인하세요.")
    return int(indices[np.argmin(values[indices])])


def draw_levels(path, x, y, levels, config, selected, plt, np):
    names = ["Left speaker only", "Right speaker only", "Both speakers (coherent)"]
    data_min = float(np.min(levels))
    data_max = float(np.max(levels))
    vmin = math.floor(data_min) - 0.5
    vmax = math.ceil(data_max) + 0.5
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
    mesh = None
    for index, (ax, title) in enumerate(zip(axes, names)):
        mesh = ax.pcolormesh(x, y, levels[index], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        for source in config["sources"]:
            sx = source["side_wall_distance_m"] if source["side_wall"] == "left" else config["room"]["width_m"] - source["side_wall_distance_m"]
            ax.scatter(sx, source["front_wall_distance_m"], marker="*", s=155, color="#ED7D31",
                       edgecolor="black", linewidth=0.8, zorder=4)
        for point in selected:
            ax.scatter(point["x_m"], point["y_m"], s=43, facecolor="white", edgecolor="#27364A", zorder=5)
            ax.annotate(point["id"], (point["x_m"], point["y_m"]), xytext=(5, 5),
                        textcoords="offset points", fontsize=9)
        ax.set(title=title, xlabel="x: distance from left wall (m)",
               xlim=(0, config["room"]["width_m"]), ylim=(0, config["room"]["length_m"]), aspect="equal")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("y: distance from front wall (m)")
    fig.subplots_adjust(left=0.06, right=0.88, top=0.86, bottom=0.14, wspace=0.18)
    cax = fig.add_axes([0.905, 0.20, 0.018, 0.58])
    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.set_label("Relative band level (dB), P02-both reference")
    fig.suptitle("Classroom model prediction | data-range common color scale", fontsize=15)
    fig.text(0.5, 0.035,
             f"Common display range {vmin:.1f} to {vmax:.1f} dB. Raw values are unchanged. "
             "QL/QR are model-selected measurement candidates.",
             ha="center", fontsize=9, color="#576579")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return vmin, vmax


def draw_interaction(path, x, y, interaction, anchor, selected, plt, np):
    line = interpolate_horizontal(interaction, x, y, anchor[1], np)
    bound = max(1.0, float(np.max(np.abs(interaction))))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    mesh = ax.pcolormesh(x, y, interaction, shading="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    for point in selected:
        ax.scatter(point["x_m"], point["y_m"], s=48, facecolor="white", edgecolor="#27364A", zorder=5)
        ax.annotate(point["id"], (point["x_m"], point["y_m"]), xytext=(5, 5), textcoords="offset points")
    ax.axhline(anchor[1], color="#303846", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set(title="Two-speaker interaction map", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.grid(alpha=0.14)
    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.03)
    colorbar.set_label("Coherent / incoherent energy sum (dB)")

    ax = axes[1]
    ax.plot(x, line, marker="o", markersize=3.5, color="#355F8A", linewidth=1.5)
    ax.axhline(0, color="#444444", linewidth=1)
    for point, color in zip(selected, ("#333333", "#C44E52", "#C44E52")):
        value = float(np.interp(point["x_m"], x, line))
        ax.scatter(point["x_m"], value, s=55, color=color, zorder=5)
        ax.annotate(f"{point['id']}\n{value:+.2f} dB", (point["x_m"], value),
                    xytext=(5, 7), textcoords="offset points", fontsize=9)
    ax.set(title=f"Horizontal section at y={anchor[1]:.2f} m", xlabel="x (m)",
           ylabel="Two-speaker interaction (dB)", xlim=(x.min(), x.max()))
    ax.grid(alpha=0.22)
    fig.suptitle("Model-guided lateral measurement positions", fontsize=15)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="(12) 11번 heatmap 재시각화와 측정지점 선정")
    parser.add_argument("--input-dir", help="11번 결과 폴더; 생략하면 최신 결과 자동 선택")
    parser.add_argument("--output-dir", help="12번 결과 폴더")
    parser.add_argument("--center-clearance-m", type=float, default=0.6,
                        help="P02 중심선에서 후보까지 최소 좌우 거리")
    parser.add_argument("--wall-clearance-m", type=float, default=0.75,
                        help="측면 벽에서 후보까지 최소 거리")
    return parser.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = find_project_root(Path(__file__).parent)
    input_dir = Path(args.input_dir) if args.input_dir else find_latest_stage11(root)
    npz_path = input_dir / NPZ_NAME
    if not npz_path.is_file():
        raise FileNotFoundError(f"원시 배열이 없습니다: {npz_path}")
    config_path = root / "configs/classroom" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    anchor_receiver = next((item for item in config["receivers"] if item["id"] == "P02"), None)
    if anchor_receiver is None:
        raise ValueError("07번 설정에 P02가 없습니다.")
    anchor = [float(value) for value in anchor_receiver["position_m"]]

    with np.load(npz_path, allow_pickle=False) as data:
        x = np.asarray(data["x_m"], dtype=float)
        y = np.asarray(data["y_m"], dtype=float)
        levels = np.asarray(data["relative_db"], dtype=float)
        interaction = np.asarray(data["interference_db"], dtype=float)
        anchor_levels = np.asarray(data["anchor_condition_relative_db"], dtype=float)
    if levels.shape != (3, len(y), len(x)) or interaction.shape != (len(y), len(x)):
        raise ValueError("11번 NPZ 배열 크기가 예상과 다릅니다.")
    if not np.all(np.isfinite(levels)) or not np.all(np.isfinite(interaction)):
        raise ValueError("11번 결과에 NaN 또는 무한대가 있습니다.")

    interaction_line = interpolate_horizontal(interaction, x, y, anchor[1], np)
    width = float(config["room"]["width_m"])
    left_index = select_side_minimum(x, interaction_line, anchor[0], "left",
                                     args.center_clearance_m, args.wall_clearance_m, width, np)
    right_index = select_side_minimum(x, interaction_line, anchor[0], "right",
                                      args.center_clearance_m, args.wall_clearance_m, width, np)

    selected = [{
        "id": "P02",
        "purpose_ko": "기존 중앙 기준점; 왼쪽만·오른쪽만·둘 다 측정",
        "x_m": anchor[0], "y_m": anchor[1], "z_m": anchor[2],
        "predicted_interaction_db": interpolate_point(interaction, x, y, anchor[0], anchor[1], np),
        "predicted_relative_db": [float(value) for value in anchor_levels],
    }]
    for point_id, index, purpose in (
        ("QL", left_index, "중앙 왼쪽의 모델상 상쇄 후보"),
        ("QR", right_index, "중앙 오른쪽의 모델상 상쇄 후보"),
    ):
        selected.append({
            "id": point_id,
            "purpose_ko": purpose,
            "x_m": float(x[index]), "y_m": anchor[1], "z_m": anchor[2],
            "predicted_interaction_db": float(interaction_line[index]),
            "predicted_relative_db": [
                float(interpolate_horizontal(levels[condition], x, y, anchor[1], np)[index])
                for condition in range(3)
            ],
        })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = Path(args.output_dir) if args.output_dir else (
        root / "results/classroom_3d" / f"heatmap_analysis(12_시각화와측정지점선정)_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=False)
    level_plot = out / "analysis(01_공통색상범위Heatmap).png"
    interaction_plot = out / "analysis(02_좌우측정후보).png"
    vmin, vmax = draw_levels(level_plot, x, y, levels, config, selected, plt, np)
    draw_interaction(interaction_plot, x, y, interaction, anchor, selected, plt, np)

    csv_path = out / "measurement_points(03_추가측정후보).csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["point_id", "x_m", "y_m", "z_m", "predicted_interaction_db",
                         "left_only_relative_db", "right_only_relative_db", "both_relative_db", "purpose_ko"])
        for point in selected:
            writer.writerow([point["id"], point["x_m"], point["y_m"], point["z_m"],
                             point["predicted_interaction_db"], *point["predicted_relative_db"], point["purpose_ko"]])

    summary = {
        "stage": 12,
        "version": VERSION,
        "source_stage11_directory": str(input_dir.resolve()),
        "source_npz": str(npz_path.resolve()),
        "selection_rule": {
            "cross_section_y_m": anchor[1],
            "left_and_right_candidates": "minimum modelled interaction on each side",
            "center_clearance_m": args.center_clearance_m,
            "wall_clearance_m": args.wall_clearance_m,
        },
        "display_range_db": [vmin, vmax],
        "points": selected,
        "interpretation_limit_ko": (
            "QL과 QR은 실측 결과가 아니라 현재 동일 신호·동일 출력·추가 지연 0 모델에서 "
            "좌우 패턴을 구분하기 위한 후보입니다. 현장 접근 가능성을 확인한 뒤 사용합니다."
        ),
    }
    (out / "summary(04_선정조건과해석).json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"(12) 시각화와 측정지점선정 v{VERSION}")
    print(f"입력: {input_dir}")
    print(f"색상 범위: {vmin:+.1f} ~ {vmax:+.1f} dB")
    for point in selected:
        print(f"{point['id']}: ({point['x_m']:.3f}, {point['y_m']:.3f}, {point['z_m']:.3f}) m / "
              f"모델상 상호작용 {point['predicted_interaction_db']:+.3f} dB")
    print("QL/QR은 나중에 현장에서 확인할 후보이며 실측 지점으로 확정된 값이 아닙니다.")
    print(f"결과 폴더: {out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("분석 실패", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)

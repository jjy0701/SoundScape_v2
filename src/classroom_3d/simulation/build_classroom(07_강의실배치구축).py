"""(07) 강의실 좌표 검사와 2D/3D 배치도 생성.

configs/classroom/classroom_config(07_강의실측정조건).json을 읽습니다.
출력은 results/classroom_3d/classroom(07_강의실배치구축)_날짜시간에 저장합니다.
--show를 붙이면 로컬 Matplotlib 창도 엽니다(지원 환경에서 3D 회전 가능).
이 단계는 기하 배치 확인용이며 RIR, SPL, AI 예측값을 계산하지 않습니다.
"""

import argparse
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path

VERSION = "1.0"
CONFIG_NAME = "classroom_config(07_강의실측정조건).json"
COLORS = {"speaker": "#D06A20", "recorded": "#087F8C", "planned": "#788493"}


def find_project_root(start):
    start = Path(start).resolve()
    start = start.parent if start.is_file() else start
    for candidate in (start, *start.parents):
        if ((candidate / "configs" / "classroom").is_dir()
                and (candidate / "configs" / "simulation").is_dir()
                and (candidate / "results").is_dir()):
            return candidate
    raise RuntimeError("sound_scape 프로젝트 루트를 찾을 수 없습니다.")


def number(value, field):
    """단위는 m. 숫자/유한값 여부를 검사합니다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: m 단위 숫자를 입력하세요.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field}: 유한한 숫자여야 합니다.")
    return value


def inside_room(position, dimensions, label):
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError(f"{label}: 좌표는 [x, y, z] 3개가 필요합니다.")
    result = [number(v, f"{label}.{axis}") for axis, v in zip("xyz", position)]
    for axis, value, limit in zip("xyz", result, dimensions):
        if not 0 < value < limit:
            raise ValueError(
                f"{label}: {axis}={value:g} m가 방 내부에 없습니다(0 < {axis} < {limit:g}). "
                "cm를 m로 바꾸었는지도 확인하세요."
            )
    return result


def resolve_scene(config):
    """벽에서 잰 거리를 공통 x/y/z로 변환합니다. 원본 config는 수정하지 않습니다."""
    if config.get("schema_version") != 1 or config.get("units") != "m":
        raise ValueError("schema_version=1, units='m' 설정이 필요합니다.")
    room = config["room"]
    dimensions = [number(room[key], key) for key in ("width_m", "length_m", "height_m")]
    if any(v <= 0 for v in dimensions):
        raise ValueError("방의 가로·세로·높이는 0보다 커야 합니다.")
    sources, receivers, used_ids = [], [], set()

    def check_id(item):
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in used_ids:
            raise ValueError(f"비어 있거나 중복된 ID입니다: {identifier!r}")
        used_ids.add(identifier)
        return identifier

    for source in config["sources"]:
        identifier = check_id(source)
        side = source["side_wall"]
        if side not in ("left", "right"):
            raise ValueError(f"{identifier}: side_wall은 left 또는 right여야 합니다.")
        offset = number(source["side_wall_distance_m"], f"{identifier}.side_wall_distance_m")
        if not 0 < offset < dimensions[0]:
            raise ValueError(f"{identifier}: 옆벽과의 거리는 방 가로보다 작고 0보다 커야 합니다.")
        x = offset if side == "left" else dimensions[0] - offset
        position = inside_room([x, source["front_wall_distance_m"], source["height_m"]],
                               dimensions, identifier)
        if not isinstance(source.get("active"), bool):
            raise ValueError(f"{identifier}: active는 true 또는 false여야 합니다.")
        sources.append({**source, "position_m": position})
    if not sources:
        raise ValueError("스피커 설정이 없습니다.")

    for receiver in config["receivers"]:
        identifier = check_id(receiver)
        if receiver.get("measurement_status") not in ("recorded", "planned"):
            raise ValueError(f"{identifier}: measurement_status는 recorded 또는 planned여야 합니다.")
        position = inside_room(receiver["position_m"], dimensions, identifier)
        receivers.append({**receiver, "position_m": position})
    by_id = {r["id"]: r for r in receivers}
    plane_id = config["display_plane_receiver_id"]
    if plane_id not in by_id:
        raise ValueError("display_plane_receiver_id와 일치하는 수음점이 없습니다.")
    distances = [
        {"source_id": s["id"], "receiver_id": r["id"],
         "straight_line_distance_m": math.dist(s["position_m"], r["position_m"])}
        for r in receivers for s in sources
    ]
    return {
        "version": VERSION,
        "stage": "geometry_only",
        "coordinate_system_ko": config["coordinate_system_ko"],
        "room": {**room, "dimensions_m": dimensions},
        "sources": sources,
        "receivers": receivers,
        "display_plane_receiver_id": plane_id,
        "display_plane_height_m": by_id[plane_id]["position_m"][2],
        "straight_line_distances": distances,
        "measurement_conditions": config.get("measurement_conditions", {}),
        "acoustic_parameters": config.get("acoustic_parameters", {}),
        "furniture": config.get("furniture", {}),
        "interpretation_ko": "방 내부 좌표 검사와 배치 확인 결과입니다. 음향 계산·실측 검증 결과는 아닙니다. 미측정 가구는 생략되어 있습니다.",
    }


def plot_plan(scene, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    width, length, height = scene["room"]["dimensions_m"]
    fig, ax = plt.subplots(figsize=(8.3, 9.4))
    fig.subplots_adjust(left=.12, right=.94, top=.90, bottom=.14)
    ax.add_patch(Rectangle((0, 0), width, length, facecolor="#F5F8FA", edgecolor="#344054", lw=2))
    for source in scene["sources"]:
        x, y, z = source["position_m"]
        ax.scatter(x, y, marker="*", s=240, c=COLORS["speaker"] if source["active"] else "#AAB2BB",
                   edgecolors="#343A40", linewidths=.8, zorder=5)
        left = source["side_wall"] == "left"
        ax.annotate(f"{source['id']}   z={z:.2f} m", (x, y),
                    xytext=(10 if left else -10, 16), textcoords="offset points",
                    ha="left" if left else "right", color="#9B4810", fontsize=10)
    for receiver in scene["receivers"]:
        x, y, z = receiver["position_m"]
        status = receiver["measurement_status"]
        ax.scatter(x, y, s=100, marker="o",
                   facecolors=COLORS[status] if status == "recorded" else "white",
                   edgecolors=COLORS[status], linewidths=2, zorder=6)
        ax.annotate(f"{receiver['id']}  {status}\nz={z:.3f} m", (x, y),
                    xytext=(13, 2), textcoords="offset points", va="center",
                    color=COLORS[status], fontsize=10)
    p = next(r for r in scene["receivers"] if r["id"] == scene["display_plane_receiver_id"])
    x, y, _ = p["position_m"]
    arrow_end = max(.2, y - .8)
    ax.annotate("", (x, arrow_end), (x, y - .12),
                arrowprops={"arrowstyle": "->", "color": COLORS["recorded"], "lw": 1.8})
    ax.text(x - .18, (arrow_end + y) / 2, "Phone bottom\ntoward front", ha="right", va="center",
            fontsize=8.5, color=COLORS["recorded"])
    ax.text(width / 2, -.47, "FRONT WALL / BOARD  (y = 0)", ha="center", va="center", fontsize=10)
    ax.text(width / 2, length + .35, "BACK WALL", ha="center", fontsize=10, color="#596579")
    handles = [
        Line2D([], [], marker="*", color="none", markerfacecolor=COLORS["speaker"],
               markeredgecolor="#343A40", markersize=12, label="Speaker (plan projection)"),
        Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["recorded"],
               markeredgecolor=COLORS["recorded"], markersize=8, label="Recorded position"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COLORS["planned"], markersize=8, label="Planned position"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(.02, .935), framealpha=.95, fontsize=9)
    ax.set(xlim=(-.35, width + .35), ylim=(-.8, length + .7),
           xlabel="x: distance from left wall (m)", ylabel="y: distance from front wall (m)")
    ax.set_aspect("equal")
    ax.set_xticks(list(range(math.floor(width) + 1)))
    ax.set_yticks(list(range(math.floor(length) + 1)))
    ax.grid(alpha=.16)
    fig.suptitle(f"Classroom plan | {width:.2f} x {length:.2f} x {height:.2f} m", fontsize=15, y=.97)
    fig.text(.5, .045, "Provisional geometry; furniture not measured. P02 x/y are approximate.\n"
             "Symbols show positions only; this is not a sound-level heatmap.",
             ha="center", fontsize=9, color="#596579")
    return fig


def plot_3d(scene, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    width, length, height = scene["room"]["dimensions_m"]
    plane = scene["display_plane_height_m"]
    fig = plt.figure(figsize=(10.8, 7.6))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=.02, right=.95, bottom=.12, top=.89)
    corners = list(itertools.product((0, width), (0, length), (0, height)))
    for a, b in itertools.combinations(corners, 2):
        if sum(v1 != v2 for v1, v2 in zip(a, b)) == 1:
            ax.plot(*zip(a, b), color="#65758B", lw=1.1, alpha=.65)
    floor = [[(0, 0, 0), (width, 0, 0), (width, length, 0), (0, length, 0)]]
    ax.add_collection3d(Poly3DCollection(floor, facecolors="#AAB6C4", alpha=.07, edgecolors="none"))
    plane_poly = [[(0, 0, plane), (width, 0, plane), (width, length, plane), (0, length, plane)]]
    ax.add_collection3d(Poly3DCollection(plane_poly, facecolors="#5CB9CE", alpha=.14,
                                        edgecolors="#42A2BA", linewidths=.8))
    for source in scene["sources"]:
        x, y, z = source["position_m"]
        ax.plot([x, x], [y, y], [0, z], linestyle="--", color=COLORS["speaker"],
                linewidth=1.1, alpha=.65)
        ax.scatter(x, y, z, marker="*", s=180, c=COLORS["speaker"] if source["active"] else "#AAB2BB",
                   edgecolors="#343A40", depthshade=False, zorder=8)
        ax.text(x + .10, y + .22, z + .06, f"{source['id']}  z={z:.2f} m", color="#9B4810", fontsize=9)
    for receiver in scene["receivers"]:
        x, y, z = receiver["position_m"]
        status = receiver["measurement_status"]
        ax.scatter(x, y, z, s=68, facecolors=COLORS[status] if status == "recorded" else "white",
                   edgecolors=COLORS[status], linewidths=1.8, depthshade=False, zorder=9)
        ax.text(x + .20, y, z + .08, receiver["id"], color=COLORS[status], fontsize=10)
    ax.set(xlim=(0, width), ylim=(0, length), zlim=(0, height), xlabel="x (m)", ylabel="y (m)",
           zlabel="Height z (m)")
    ax.set_zticks([0, plane, height])
    ax.set_zticklabels(["0", f"{plane:.3f}", f"{height:.2f}"])
    ax.set_box_aspect((width, length, height))
    ax.view_init(elev=25, azim=-58)
    ax.tick_params(labelsize=9)
    ax.text(width * .28, 0, .03, "FRONT / BOARD", color="#596579", fontsize=9)
    handles = [
        Line2D([], [], marker="*", color="none", markerfacecolor=COLORS["speaker"],
               markersize=12, label="Speakers"),
        Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["recorded"],
               markersize=8, label="P02 recorded"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COLORS["planned"], markersize=8, label="P01 / P03 planned"),
        Patch(facecolor="#5CB9CE", alpha=.35, label=f"Reference plane z={plane:.3f} m"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .905), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Classroom 3D geometry | {width:.2f} x {length:.2f} x {height:.2f} m", fontsize=15, y=.98)
    fig.text(.5, .035, "Room dimensions are provisional; furniture is not included.\n"
             "Dashed lines are vertical height guides. Blue plane: receiver height, not pressure.",
             ha="center", fontsize=9, color="#596579")
    return fig


def main():
    project_root = find_project_root(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "classroom" / CONFIG_NAME)
    parser.add_argument("--output-root", type=Path, default=project_root / "results" / "classroom_3d")
    parser.add_argument("--show", action="store_true", help="저장 후 2D/3D 그림 창 열기")
    args = parser.parse_args()
    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes.decode("utf-8-sig"))
        scene = resolve_scene(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"설정 확인 필요: {exc}", file=sys.stderr)
        return 1
    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib이 없습니다. sound_scape의 .venv Python으로 실행하세요.\n'
              '필요한 경우 해당 환경에서: python -m pip install matplotlib', file=sys.stderr)
        return 1
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    out = args.output_root.resolve() / ("classroom(07_강의실배치구축)_" +
                                       datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    scene.update({"created_at": datetime.now().astimezone().isoformat(),
                  "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "config_snapshot": config})
    (out / "classroom(01_좌표와조건).json").write_text(
        json.dumps(scene, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    figures = [plot_plan(scene, plt), plot_3d(scene, plt)]
    names = ["classroom(02_평면배치).png", "classroom(03_입체배치).png"]
    for fig, name in zip(figures, names):
        fig.savefig(out / name, dpi=150)
    print("(07) 강의실배치구축 v" + VERSION)
    print("좌표 기준: " + scene["coordinate_system_ko"])
    print("방 가로/세로/높이(m): " + " / ".join(f"{v:g}" for v in scene["room"]["dimensions_m"]))
    for source in scene["sources"]:
        print(f"{source['id']} {source['label_ko']}: {source['position_m']} m / 사용={source['active']}")
    for receiver in scene["receivers"]:
        status = "녹음 있음" if receiver["measurement_status"] == "recorded" else "예정 위치, 녹음 없음"
        print(f"{receiver['id']}: {receiver['position_m']} m / {status}")
    print("좌표 검사: PASS — 모든 설정 지점이 방 내부에 있습니다.")
    print("방 치수는 이전 메모의 잠정값, P02 x/y는 타일 기준 약값입니다.")
    print("배치도에는 미측정 책상·의자·교탁을 생략했습니다.")
    print("이번 단계는 배치 확인입니다. RIR/SPL 계산 및 실측 오차 평가는 수행하지 않았습니다.")
    print(f"결과 폴더: {out}")
    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

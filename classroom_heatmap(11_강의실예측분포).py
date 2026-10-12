"""(11) 3D 강의실의 주파수 대역별 상대 음향 분포를 계산합니다.

07번 강의실 배치, 08번 음향 가정, 06번 시험 신호를 사용합니다.
휴대폰 높이의 격자에서 왼쪽 스피커, 오른쪽 스피커, 두 스피커의
250~4,000 Hz 상대 에너지 분포를 계산합니다.

결과는 실측 보정 전 모델 예측이며 절대 SPL이 아닙니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
import wave
from datetime import datetime
from pathlib import Path


VERSION = "1.0"
CONFIG_NAME = "classroom_config(07_강의실측정조건).json"
SETTINGS_NAME = "simulation_settings(08_음향가정).json"
REFERENCE_NAME = "test_signal(06_예비녹음_50초).wav"


def finite_number(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: 숫자가 필요합니다.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name}: 유한한 값이 필요합니다.")
    if lower is not None and value < lower:
        raise ValueError(f"{name}: {lower} 이상이어야 합니다.")
    if upper is not None and value > upper:
        raise ValueError(f"{name}: {upper} 이하여야 합니다.")
    return value


def integer(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name}: {lower}~{upper} 범위의 정수가 필요합니다.")
    return value


def load_json(path):
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def find_project_root(start):
    expected = (
        Path("configs/classroom") / CONFIG_NAME,
        Path("configs/simulation") / SETTINGS_NAME,
        Path("recordings/reference") / REFERENCE_NAME,
    )
    starts = [start.resolve(), Path.cwd().resolve()]
    checked = set()
    for first in starts:
        for root in (first, *first.parents):
            if root in checked:
                continue
            checked.add(root)
            if all((root / item).is_file() for item in expected):
                return root
    raise FileNotFoundError(
        "프로젝트 루트를 찾지 못했습니다. 스크립트를 sound_scape 안에 두거나 "
        "--config, --settings, --reference를 지정하세요."
    )


def resolve_paths(args):
    if args.config and args.settings and args.reference:
        return Path(args.config), Path(args.settings), Path(args.reference), Path.cwd()
    root = find_project_root(Path(__file__).parent)
    return (
        Path(args.config) if args.config else root / "configs/classroom" / CONFIG_NAME,
        Path(args.settings) if args.settings else root / "configs/simulation" / SETTINGS_NAME,
        Path(args.reference) if args.reference else root / "recordings/reference" / REFERENCE_NAME,
        root,
    )


def resolve_scene(config, settings):
    if config.get("schema_version") != 1 or config.get("units") != "m":
        raise ValueError("07번 설정은 schema_version=1, units='m'이어야 합니다.")
    if settings.get("schema_version") != 1:
        raise ValueError("08번 설정의 schema_version은 1이어야 합니다.")

    room = config["room"]
    dims = [finite_number(room[key], key, 0.01) for key in ("width_m", "length_m", "height_m")]
    fs = integer(settings["sample_rate_hz"], "sample_rate_hz", 8000, 192000)
    speed = finite_number(settings["speed_of_sound_m_s"], "speed_of_sound_m_s", 250, 450)
    order = integer(settings["max_order"], "max_order", 0, 128)
    alpha = {
        key: finite_number(settings["energy_absorption"][key], key, 0, 1)
        for key in ("walls", "floor", "ceiling")
    }

    receivers = {item["id"]: item for item in config["receivers"]}
    if "P02" not in receivers:
        raise ValueError("07번 설정에 공통 기준점 P02가 없습니다.")
    anchor = [finite_number(v, "P02.position_m") for v in receivers["P02"]["position_m"]]
    if any(not 0 < v < d for v, d in zip(anchor, dims)):
        raise ValueError("P02가 방 내부 좌표가 아닙니다.")

    sources = []
    for item in config["sources"]:
        if not item.get("active", False):
            continue
        sid = item["id"]
        if sid not in settings["sources"]:
            raise ValueError(f"08번 설정에 {sid}의 음원 가정이 없습니다.")
        if item["side_wall"] == "left":
            x = finite_number(item["side_wall_distance_m"], f"{sid}.side_wall_distance_m", 0)
        elif item["side_wall"] == "right":
            x = dims[0] - finite_number(item["side_wall_distance_m"], f"{sid}.side_wall_distance_m", 0)
        else:
            raise ValueError(f"{sid}.side_wall은 left 또는 right여야 합니다.")
        position = [
            x,
            finite_number(item["front_wall_distance_m"], f"{sid}.front_wall_distance_m"),
            finite_number(item["height_m"], f"{sid}.height_m"),
        ]
        if any(not 0 < v < d for v, d in zip(position, dims)):
            raise ValueError(f"{sid}가 방 내부 좌표가 아닙니다.")
        assumption = settings["sources"][sid]
        sources.append({
            "id": sid,
            "position_m": position,
            "gain": finite_number(assumption["relative_gain"], f"{sid}.relative_gain", 0),
            "delay": integer(assumption["extra_delay_samples"], f"{sid}.extra_delay_samples", 0, fs),
        })

    source_ids = {source["id"] for source in sources}
    if not {"S_L", "S_R"}.issubset(source_ids):
        raise ValueError("11번 비교에는 활성 음원 S_L과 S_R이 모두 필요합니다.")
    if config.get("furniture", {}).get("objects"):
        raise ValueError("가구 객체가 설정되어 있지만 이 ShoeBox 계산에는 구현되지 않았습니다.")

    return {
        "dimensions_m": dims,
        "fs": fs,
        "speed": speed,
        "order": order,
        "alpha": alpha,
        "anchor": anchor,
        "sources": sources,
        "receivers": receivers,
    }


def read_reference(path, fs, np):
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
            raise ValueError("시험 신호는 16-bit PCM WAV여야 합니다.")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        frames = wav.getnframes()
        if rate != fs or channels not in (1, 2):
            raise ValueError(f"시험 신호는 {fs} Hz, 1~2채널이어야 합니다.")
        pcm = np.frombuffer(wav.readframes(frames), dtype="<i2").reshape(-1, channels)
    if channels == 2 and not np.array_equal(pcm[:, 0], pcm[:, 1]):
        raise ValueError("좌우가 다른 스테레오 시험 신호는 동일 신호 가정과 맞지 않습니다.")
    signal = pcm[:, 0].astype(np.float64) / 32768.0
    if not np.any(signal):
        raise ValueError("시험 신호가 전부 무음입니다.")
    return signal, {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate_hz": rate,
        "channels": channels,
        "duration_s": frames / rate,
    }


def choose_active_core(signal, fs, np):
    """첫 번째 긴 비무음 구간에서 양 끝 1초를 제외합니다."""
    hop = max(1, round(0.01 * fs))
    padded = np.pad(signal, (0, (-len(signal)) % hop))
    active = np.max(np.abs(padded.reshape(-1, hop)), axis=1) > 0
    edges = np.diff(np.r_[False, active, False].astype(int))
    blocks = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    blocks = [(int(a * hop), min(int(b * hop), len(signal))) for a, b in blocks if (b - a) * hop >= 3 * fs]
    if not blocks:
        raise ValueError("시험 신호에서 3초 이상의 비무음 구간을 찾지 못했습니다.")
    start, stop = blocks[0]
    if stop - start > 3 * fs:
        start += fs
        stop -= fs
    core = signal[start:stop]
    if len(core) < fs:
        raise ValueError("분석할 시험 신호 구간이 너무 짧습니다.")
    return core, {"start_s": start / fs, "stop_s": stop / fs, "duration_s": len(core) / fs}


def next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def build_room(scene, microphones, pra, np):
    a = scene["alpha"]
    materials = pra.make_materials(
        west=a["walls"], east=a["walls"], south=a["walls"], north=a["walls"],
        floor=a["floor"], ceiling=a["ceiling"],
    )
    room = pra.ShoeBox(
        scene["dimensions_m"], fs=scene["fs"], materials=materials,
        max_order=scene["order"], ray_tracing=False, air_absorption=False,
        use_rand_ism=False, sigma2_awgn=None,
    )
    room.set_sound_speed(scene["speed"])
    for source in scene["sources"]:
        room.add_source(source["position_m"])
    room.add_microphone_array(np.asarray(microphones, dtype=float).T)
    room.compute_rir()
    return room


def frequency_weights(core, fs, nfft, low_hz, high_hz, butter, sosfreqz, np):
    spectrum = np.fft.rfft(core, n=nfft)
    frequencies = np.fft.rfftfreq(nfft, 1.0 / fs)
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    _, response = sosfreqz(sos, worN=frequencies, fs=fs)
    # 09번의 sosfiltfilt와 같은 전후방 필터의 정상상태 크기 응답을 근사합니다.
    filter_power = np.abs(response) ** 4
    parseval = np.ones_like(frequencies)
    if nfft % 2 == 0:
        parseval[1:-1] = 2.0
    else:
        parseval[1:] = 2.0
    return np.abs(spectrum) ** 2 * filter_power * parseval / nfft


def rir_spectrum(rir, source, nfft, np):
    padded = np.pad(np.asarray(rir, dtype=np.float64) * source["gain"], (source["delay"], 0))
    return np.fft.rfft(padded, n=nfft)


def weighted_energy(transfer, weights, np):
    value = float(np.sum(weights * np.abs(transfer) ** 2))
    return max(value, np.finfo(np.float64).tiny)


def calculate_maps(room, scene, grid_shape, core, band, butter, sosfreqz, np):
    source_index = {source["id"]: index for index, source in enumerate(scene["sources"])}
    left_i, right_i = source_index["S_L"], source_index["S_R"]
    left_source, right_source = scene["sources"][left_i], scene["sources"][right_i]
    max_rir = max(len(rir) + source["delay"] for mic in room.rir for rir, source in zip(mic, scene["sources"]))
    nfft = next_power_of_two(len(core) + max_rir - 1)
    weights = frequency_weights(core, scene["fs"], nfft, band[0], band[1], butter, sosfreqz, np)

    point_count = grid_shape[0] * grid_shape[1]
    energies = np.empty((3, point_count + 1), dtype=np.float64)
    interference = np.empty(point_count + 1, dtype=np.float64)
    for mic_i in range(point_count + 1):
        h_left = rir_spectrum(room.rir[mic_i][left_i], left_source, nfft, np)
        h_right = rir_spectrum(room.rir[mic_i][right_i], right_source, nfft, np)
        e_left = weighted_energy(h_left, weights, np)
        e_right = weighted_energy(h_right, weights, np)
        e_both = weighted_energy(h_left + h_right, weights, np)
        energies[:, mic_i] = (e_left, e_right, e_both)
        interference[mic_i] = 10.0 * math.log10(e_both / (e_left + e_right))

    anchor_energy = energies[2, -1]
    relative = 10.0 * np.log10(energies[:, :-1] / anchor_energy).reshape(3, *grid_shape)
    anchor_levels = 10.0 * np.log10(energies[:, -1] / anchor_energy)
    return relative, interference[:-1].reshape(grid_shape), anchor_levels, nfft


def draw_level_maps(out_path, scene, x, y, levels, plot_min, plt, np):
    names = ["Left speaker only", "Right speaker only", "Both speakers (coherent)"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)
    vmax = max(6.0, float(math.ceil(np.nanmax(levels))))
    mesh = None
    for index, (ax, title) in enumerate(zip(axes, names)):
        mesh = ax.pcolormesh(x, y, levels[index], shading="auto", cmap="viridis", vmin=plot_min, vmax=vmax)
        for source in scene["sources"]:
            ax.scatter(source["position_m"][0], source["position_m"][1], marker="*", s=150,
                       color="#ED7D31", edgecolor="black", linewidth=0.8, zorder=4)
        for rid, receiver in scene["receivers"].items():
            px, py, _ = receiver["position_m"]
            ax.scatter(px, py, s=30, facecolor="white", edgecolor="#27364A", zorder=4)
            ax.annotate(rid, (px, py), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set(title=title, xlabel="x: distance from left wall (m)", xlim=(0, scene["dimensions_m"][0]),
               ylim=(0, scene["dimensions_m"][1]), aspect="equal")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("y: distance from front wall (m)")
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.82, pad=0.025)
    colorbar.set_label("Relative band level (dB), common P02-both reference")
    fig.suptitle("Classroom model prediction at z=0.706 m | 250–4,000 Hz", fontsize=15)
    fig.text(0.5, 0.02,
             "Uncalibrated 3D ShoeBox simulation; P02 with both speakers is 0 dB. "
             "Raw arrays are not clipped. Furniture, measured directivity and phone processing are omitted.",
             ha="center", fontsize=8.5, color="#576579")
    fig.subplots_adjust(left=0.06, right=0.91, top=0.86, bottom=0.15, wspace=0.18)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_interference(out_path, scene, x, y, interference, plt, np):
    bound = max(1.0, float(np.nanpercentile(np.abs(interference), 98)))
    fig, ax = plt.subplots(figsize=(7.2, 8.1))
    mesh = ax.pcolormesh(x, y, interference, shading="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    for source in scene["sources"]:
        ax.scatter(source["position_m"][0], source["position_m"][1], marker="*", s=170,
                   color="#ED7D31", edgecolor="black", linewidth=0.8, zorder=4)
    for rid, receiver in scene["receivers"].items():
        px, py, _ = receiver["position_m"]
        ax.scatter(px, py, s=35, facecolor="white", edgecolor="#27364A", zorder=4)
        ax.annotate(rid, (px, py), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set(
        title="Modelled two-speaker interaction | 250–4,000 Hz",
        xlabel="x: distance from left wall (m)", ylabel="y: distance from front wall (m)",
        xlim=(0, scene["dimensions_m"][0]), ylim=(0, scene["dimensions_m"][1]), aspect="equal",
    )
    ax.grid(alpha=0.15)
    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.025)
    colorbar.set_label("Coherent both / incoherent energy sum (dB)")
    fig.text(0.5, 0.025,
             "Positive and negative values show interaction under the assumed identical-signal, zero-extra-delay model. "
             "This is not a measured interference map.", ha="center", fontsize=8.5, color="#576579")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="(11) 강의실 예측 heatmap 생성")
    parser.add_argument("--config", help="07번 강의실 JSON")
    parser.add_argument("--settings", help="08번 음향 가정 JSON")
    parser.add_argument("--reference", help="06번 시험 신호 WAV")
    parser.add_argument("--output-dir", help="결과 폴더")
    parser.add_argument("--grid-size", type=int, default=20, help="가로·세로 격자 수 (기본 20)")
    parser.add_argument("--margin-m", type=float, default=0.25, help="벽과 격자의 여백 m (기본 0.25)")
    parser.add_argument("--low-hz", type=float, default=250.0)
    parser.add_argument("--high-hz", type=float, default=4000.0)
    parser.add_argument("--plot-min-db", type=float, default=-30.0, help="그림 색상 최소값; 원시 배열은 자르지 않음")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    if not 5 <= args.grid_size <= 100:
        raise ValueError("grid-size는 5~100이어야 합니다.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pyroomacoustics as pra
    import scipy
    from scipy.signal import butter, sosfreqz

    config_path, settings_path, reference_path, root = resolve_paths(args)
    config, config_sha = load_json(config_path)
    settings, settings_sha = load_json(settings_path)
    scene = resolve_scene(config, settings)
    if not 0 < args.margin_m < min(scene["dimensions_m"][:2]) / 2:
        raise ValueError("margin-m이 방 크기에 맞지 않습니다.")
    if not 0 < args.low_hz < args.high_hz < scene["fs"] / 2:
        raise ValueError("분석 대역은 0 < low < high < Nyquist 조건이어야 합니다.")

    signal, reference_info = read_reference(reference_path, scene["fs"], np)
    core, core_info = choose_active_core(signal, scene["fs"], np)
    z = scene["anchor"][2]
    x = np.linspace(args.margin_m, scene["dimensions_m"][0] - args.margin_m, args.grid_size)
    y = np.linspace(args.margin_m, scene["dimensions_m"][1] - args.margin_m, args.grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_mics = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
    microphones = np.vstack((grid_mics, np.asarray(scene["anchor"], dtype=float)))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = Path(args.output_dir) if args.output_dir else (
        root / "results/classroom_3d" / f"classroom_heatmap(11_강의실예측분포)_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=False)

    print(f"(11) 강의실 예측분포 v{VERSION}")
    print(f"격자={args.grid_size}×{args.grid_size} / 높이={z:.3f} m / 대역={args.low_hz:g}~{args.high_hz:g} Hz")
    print(f"공통 기준: 두 스피커를 사용한 P02 = 0 dB")
    print(f"RIR 계산 중: 음원 2개 × 수음점 {len(microphones)}개 / 반사 차수 {scene['order']}")
    room = build_room(scene, microphones, pra, np)
    print("주파수 가중 상대 에너지 계산 중...")
    levels, interference, anchor_levels, nfft = calculate_maps(
        room, scene, xx.shape, core, (args.low_hz, args.high_hz), butter, sosfreqz, np
    )

    arrays_path = out / "heatmap(01_상대레벨원시배열).npz"
    np.savez_compressed(
        arrays_path,
        x_m=x,
        y_m=y,
        z_m=np.array(z),
        condition_ids=np.array(["left_only", "right_only", "both_coherent"]),
        relative_db=levels,
        interference_db=interference,
        anchor_condition_relative_db=anchor_levels,
    )
    map_path = out / "heatmap(02_좌우및동시재생).png"
    interference_path = out / "heatmap(03_두스피커간섭).png"
    draw_level_maps(map_path, scene, x, y, levels, args.plot_min_db, plt, np)
    draw_interference(interference_path, scene, x, y, interference, plt, np)

    metrics = {
        "stage": 11,
        "version": VERSION,
        "status": "uncalibrated_simulation_prediction",
        "description": "3D ShoeBox model band-weighted relative energy heatmap; not absolute SPL",
        "grid": {
            "size": [args.grid_size, args.grid_size],
            "receiver_count": int(xx.size),
            "height_m": z,
            "wall_margin_m": args.margin_m,
        },
        "analysis": {
            "band_hz": [args.low_hz, args.high_hz],
            "reference_core": core_info,
            "fft_size": nfft,
            "filter": "4th-order Butterworth band-pass; squared-magnitude filtfilt steady-state approximation",
            "energy_rule": "source-spectrum-weighted RIR transfer energy",
            "combination_rule": "complex transfer functions summed before energy for both_coherent",
        },
        "normalization": {
            "anchor": "P02",
            "anchor_condition": "both_coherent",
            "anchor_relative_db_by_condition": {
                "left_only": float(anchor_levels[0]),
                "right_only": float(anchor_levels[1]),
                "both_coherent": float(anchor_levels[2]),
            },
            "raw_predictions_clipped_or_renormalized": False,
        },
        "range_db": {
            name: {"min": float(np.min(levels[i])), "max": float(np.max(levels[i]))}
            for i, name in enumerate(("left_only", "right_only", "both_coherent"))
        },
        "interference_db": {
            "definition": "10log10(E_both_coherent / (E_left + E_right))",
            "min": float(np.min(interference)),
            "max": float(np.max(interference)),
            "mean": float(np.mean(interference)),
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": config_sha},
            "settings": {"path": str(settings_path), "sha256": settings_sha},
            "reference": {"path": str(reference_path), **reference_info},
        },
        "model_assumptions": {
            "dimensions_m": scene["dimensions_m"],
            "max_order": scene["order"],
            "absorption": scene["alpha"],
            "source_ids": [source["id"] for source in scene["sources"]],
            "same_signal_and_coherent_sum": True,
            "omitted": ["furniture", "measured directivity", "air absorption", "phone processing", "measurement calibration"],
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyroomacoustics": pra.__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    metrics_path = out / "metrics(04_계산조건과범위).json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("완료")
    for name, values in metrics["range_db"].items():
        print(f"  {name}: {values['min']:+.3f} ~ {values['max']:+.3f} dB")
    print(f"  모델상 두 스피커 상호작용: {metrics['interference_db']['min']:+.3f} ~ "
          f"{metrics['interference_db']['max']:+.3f} dB")
    print("현재 결과는 실측 보정 전 상대 예측이며 절대 SPL이 아닙니다.")
    print(f"결과 폴더: {out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("계산 실패", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)

"""(13) 강의실 모델의 흡음률·좌우 출력·상대 지연 민감도를 분석합니다.

실측값에 파라미터를 맞추는 최적화가 아닙니다. P01/P02/P03과 P02를 지나는
y=5 m 가로 단면에서, 가정 변화가 상대 레벨 예측을 얼마나 바꾸는지 계산합니다.
"""

from __future__ import annotations

import argparse
import csv
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
ALPHAS = (0.1, 0.2, 0.3)
RIGHT_GAINS = (0.8, 1.0, 1.2)
RELATIVE_DELAYS = (-5, -2, 0, 2, 5)


def load_json(path):
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def finite(value, name, lower=None, upper=None):
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


def find_root(start):
    expected = (
        Path("configs/classroom") / CONFIG_NAME,
        Path("configs/simulation") / SETTINGS_NAME,
        Path("recordings/reference") / REFERENCE_NAME,
    )
    checked = set()
    for first in (start.resolve(), Path.cwd().resolve()):
        for root in (first, *first.parents):
            if root in checked:
                continue
            checked.add(root)
            if all((root / item).is_file() for item in expected):
                return root
    raise FileNotFoundError("sound_scape 프로젝트 루트를 찾지 못했습니다.")


def resolve_scene(config, settings):
    room = config["room"]
    dims = [finite(room[key], key, 0.01) for key in ("width_m", "length_m", "height_m")]
    fs = int(settings["sample_rate_hz"])
    if not 8000 <= fs <= 192000:
        raise ValueError("sample_rate_hz 범위를 확인하세요.")
    order = int(settings["max_order"])
    if not 0 <= order <= 128:
        raise ValueError("max_order 범위를 확인하세요.")
    speed = finite(settings["speed_of_sound_m_s"], "speed_of_sound_m_s", 250, 450)
    receivers = {item["id"]: item for item in config["receivers"]}
    for rid in ("P01", "P02", "P03"):
        if rid not in receivers:
            raise ValueError(f"07번 설정에 {rid}이 없습니다.")
    points = {rid: [finite(v, f"{rid}.position_m") for v in receivers[rid]["position_m"]]
              for rid in ("P01", "P02", "P03")}

    sources = {}
    for item in config["sources"]:
        if not item.get("active", False):
            continue
        sid = item["id"]
        offset = finite(item["side_wall_distance_m"], f"{sid}.side_wall_distance_m", 0)
        if item["side_wall"] == "left":
            x = offset
        elif item["side_wall"] == "right":
            x = dims[0] - offset
        else:
            raise ValueError(f"{sid}.side_wall은 left/right여야 합니다.")
        sources[sid] = [x, finite(item["front_wall_distance_m"], f"{sid}.front_wall_distance_m"),
                        finite(item["height_m"], f"{sid}.height_m")]
    if "S_L" not in sources or "S_R" not in sources:
        raise ValueError("활성 음원 S_L과 S_R이 모두 필요합니다.")
    if config.get("furniture", {}).get("objects"):
        raise ValueError("가구 객체는 이번 ShoeBox 민감도 분석에 구현되지 않았습니다.")
    return {"dims": dims, "fs": fs, "order": order, "speed": speed,
            "points": points, "sources": sources}


def read_reference(path, fs, np):
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
            raise ValueError("시험 신호는 16-bit PCM WAV여야 합니다.")
        channels, rate, frames = wav.getnchannels(), wav.getframerate(), wav.getnframes()
        if channels not in (1, 2) or rate != fs:
            raise ValueError(f"시험 신호는 {fs} Hz, 1~2채널이어야 합니다.")
        pcm = np.frombuffer(wav.readframes(frames), dtype="<i2").reshape(-1, channels)
    if channels == 2 and not np.array_equal(pcm[:, 0], pcm[:, 1]):
        raise ValueError("좌우가 다른 스테레오 시험 신호입니다.")
    signal = pcm[:, 0].astype(np.float64) / 32768.0
    hop = max(1, round(0.01 * fs))
    padded = np.pad(signal, (0, (-len(signal)) % hop))
    active = np.max(np.abs(padded.reshape(-1, hop)), axis=1) > 0
    edges = np.diff(np.r_[False, active, False].astype(int))
    blocks = [(int(a * hop), min(int(b * hop), len(signal)))
              for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
              if (b - a) * hop >= 3 * fs]
    if not blocks:
        raise ValueError("시험 신호에서 긴 비무음 구간을 찾지 못했습니다.")
    start, stop = blocks[0]
    start += fs
    stop -= fs
    core = signal[start:stop]
    if len(core) < fs:
        raise ValueError("분석 신호 구간이 너무 짧습니다.")
    return core, {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "core_start_s": start / fs, "core_stop_s": stop / fs}


def next_pow2(value):
    return 1 << (int(value) - 1).bit_length()


def build_room(scene, microphones, alpha, pra, np):
    materials = pra.make_materials(west=alpha, east=alpha, south=alpha, north=alpha,
                                   floor=alpha, ceiling=alpha)
    room = pra.ShoeBox(scene["dims"], fs=scene["fs"], materials=materials,
                       max_order=scene["order"], ray_tracing=False, air_absorption=False,
                       use_rand_ism=False, sigma2_awgn=None)
    room.set_sound_speed(scene["speed"])
    room.add_source(scene["sources"]["S_L"])
    room.add_source(scene["sources"]["S_R"])
    room.add_microphone_array(np.asarray(microphones, dtype=float).T)
    room.compute_rir()
    return room


def make_weights(core, fs, nfft, low_hz, high_hz, butter, sosfreqz, np):
    frequencies = np.fft.rfftfreq(nfft, 1 / fs)
    x = np.fft.rfft(core, n=nfft)
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    _, response = sosfreqz(sos, worN=frequencies, fs=fs)
    parseval = np.ones_like(frequencies)
    parseval[1:-1 if nfft % 2 == 0 else None] = 2.0
    weights = np.abs(x) ** 2 * np.abs(response) ** 4 * parseval / nfft
    omega = 2 * np.pi * np.arange(len(frequencies)) / nfft
    return weights, omega


def compute_components(room, core, fs, delays, butter, sosfreqz, np):
    max_rir = max(len(rir) for mic in room.rir for rir in mic)
    nfft = next_pow2(len(core) + max_rir + max(abs(value) for value in delays) - 1)
    weights, omega = make_weights(core, fs, nfft, 250.0, 4000.0, butter, sosfreqz, np)
    count = len(room.rir)
    left_energy = np.empty(count)
    right_energy = np.empty(count)
    cross = {delay: np.empty(count) for delay in delays}
    phases = {delay: np.exp(1j * omega * delay) for delay in delays}
    for index, microphone_rirs in enumerate(room.rir):
        left = np.fft.rfft(np.asarray(microphone_rirs[0], dtype=float), n=nfft)
        right = np.fft.rfft(np.asarray(microphone_rirs[1], dtype=float), n=nfft)
        left_energy[index] = max(float(np.sum(weights * np.abs(left) ** 2)), np.finfo(float).tiny)
        right_energy[index] = max(float(np.sum(weights * np.abs(right) ** 2)), np.finfo(float).tiny)
        product = left * np.conj(right)
        for delay in delays:
            cross[delay][index] = float(2.0 * np.real(np.sum(weights * product * phases[delay])))
    return left_energy, right_energy, cross, nfft


def scenario_energy(left, right, cross, right_gain, delay, np):
    energy = left + right_gain ** 2 * right + right_gain * cross[delay]
    floor = np.finfo(np.float64).tiny
    if np.any(energy <= floor):
        raise ValueError("시나리오 에너지가 0 이하입니다. 수치 또는 위상 계산을 확인하세요.")
    return energy


def draw_sensitivity(path, x, scenarios, cross_indices, point_indices, plt, np):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    groups = [
        ("absorption", ALPHAS, lambda value: scenarios[(value, 1.0, 0)], "Absorption α"),
        ("gain", RIGHT_GAINS, lambda value: scenarios[(0.2, value, 0)], "Right / left gain"),
        ("delay", RELATIVE_DELAYS, lambda value: scenarios[(0.2, 1.0, value)], "Right relative delay (samples)"),
    ]
    colors = ("#286AA4", "#C56924", "#087F8C", "#8D5A97", "#6A994E")
    p01_i, p02_i, p03_i = point_indices
    for column, (_, values, getter, xlabel) in enumerate(groups):
        ax = axes[0, column]
        for color, value in zip(colors, values):
            result = getter(value)
            ax.plot(x, result["relative_db"][cross_indices], marker="o", markersize=2.7,
                    linewidth=1.25, color=color, label=str(value))
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set(title=f"Cross-section sensitivity: {xlabel}", xlabel="x at y=5.0 m (m)", ylabel="Relative level to P02 (dB)")
        ax.grid(alpha=0.2)
        ax.legend(title=xlabel, fontsize=8, title_fontsize=8)

        ax = axes[1, column]
        p01_values = [getter(value)["relative_db"][p01_i] for value in values]
        p03_values = [getter(value)["relative_db"][p03_i] for value in values]
        positions = np.arange(len(values))
        ax.plot(positions, p01_values, marker="o", label="P01 relative to P02", color="#C44E52")
        ax.plot(positions, p03_values, marker="o", label="P03 relative to P02", color="#4C72B0")
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set(title="Three-point prediction", xlabel=xlabel, ylabel="Relative level (dB)",
               xticks=positions, xticklabels=[str(value) for value in values])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Classroom acoustic-assumption sensitivity | 250–4,000 Hz", fontsize=16)
    fig.text(0.5, 0.015,
             "One parameter varies in each column; other parameters remain at α=0.2, gain=1.0, delay=0. "
             "Model sensitivity, not parameter fitting or measured accuracy.",
             ha="center", fontsize=9, color="#576579")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_ranking(path, rows, plt, np):
    selected = [row for row in rows if row["is_one_at_a_time"] and not row["is_baseline"]]
    labels = [row["short_label"] for row in selected]
    values = [row["cross_section_rmse_from_baseline_db"] for row in selected]
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.array(labels)[order], np.array(values)[order], color="#4C78A8")
    for index in order:
        ax.text(values[index] + 0.01, list(order).index(index), f"{values[index]:.3f}", va="center", fontsize=9)
    ax.set(title="Sensitivity ranking relative to baseline", xlabel="Cross-section RMSE change (dB)")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="(13) 강의실 음향 가정 민감도 분석")
    parser.add_argument("--output-dir", help="결과 폴더")
    parser.add_argument("--cross-points", type=int, default=31, help="y=5 m 단면의 x 지점 수 (기본 31)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 15 <= args.cross_points <= 101:
        raise ValueError("cross-points는 15~101이어야 합니다.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pyroomacoustics as pra
    import scipy
    from scipy.signal import butter, sosfreqz

    started = time.perf_counter()
    root = find_root(Path(__file__).parent)
    config_path = root / "configs/classroom" / CONFIG_NAME
    settings_path = root / "configs/simulation" / SETTINGS_NAME
    reference_path = root / "recordings/reference" / REFERENCE_NAME
    config, config_sha = load_json(config_path)
    settings, settings_sha = load_json(settings_path)
    scene = resolve_scene(config, settings)
    core, reference_info = read_reference(reference_path, scene["fs"], np)

    anchor = scene["points"]["P02"]
    cross_x = np.linspace(0.5, scene["dims"][0] - 0.5, args.cross_points)
    cross_mics = np.column_stack((cross_x, np.full(args.cross_points, anchor[1]),
                                  np.full(args.cross_points, anchor[2])))
    point_mics = np.array([scene["points"][rid] for rid in ("P01", "P02", "P03")], dtype=float)
    microphones = np.vstack((cross_mics, point_mics))
    cross_indices = np.arange(args.cross_points)
    point_indices = (args.cross_points, args.cross_points + 1, args.cross_points + 2)
    anchor_index = point_indices[1]

    all_results = {}
    fft_sizes = {}
    print(f"(13) 음향가정민감도분석 v{VERSION}")
    print(f"수음점: 가로 단면 {args.cross_points}개 + P01/P02/P03 / 반사 차수 {scene['order']}")
    for alpha in ALPHAS:
        print(f"흡음률 α={alpha:.1f}: RIR 계산 중...")
        room = build_room(scene, microphones, alpha, pra, np)
        left, right, cross, nfft = compute_components(
            room, core, scene["fs"], RELATIVE_DELAYS, butter, sosfreqz, np
        )
        fft_sizes[str(alpha)] = nfft
        for gain in RIGHT_GAINS:
            for delay in RELATIVE_DELAYS:
                energy = scenario_energy(left, right, cross, gain, delay, np)
                relative = 10.0 * np.log10(energy / energy[anchor_index])
                incoherent = left + gain ** 2 * right
                interaction = 10.0 * np.log10(energy / incoherent)
                all_results[(alpha, gain, delay)] = {
                    "relative_db": relative,
                    "interaction_db": interaction,
                }

    baseline = all_results[(0.2, 1.0, 0)]
    rows = []
    for (alpha, gain, delay), result in all_results.items():
        delta = result["relative_db"][cross_indices] - baseline["relative_db"][cross_indices]
        one_at_a_time = ((gain == 1.0 and delay == 0) or
                         (alpha == 0.2 and delay == 0) or
                         (alpha == 0.2 and gain == 1.0))
        is_baseline = alpha == 0.2 and gain == 1.0 and delay == 0
        label_parts = []
        if alpha != 0.2:
            label_parts.append(f"α={alpha:.1f}")
        if gain != 1.0:
            label_parts.append(f"gain={gain:.1f}")
        if delay != 0:
            label_parts.append(f"delay={delay:+d}")
        rows.append({
            "alpha": alpha,
            "right_gain": gain,
            "relative_delay_samples": delay,
            "relative_delay_ms": delay / scene["fs"] * 1000,
            "p01_relative_db": float(result["relative_db"][point_indices[0]]),
            "p03_relative_db": float(result["relative_db"][point_indices[2]]),
            "p02_interaction_db": float(result["interaction_db"][point_indices[1]]),
            "cross_section_rmse_from_baseline_db": float(np.sqrt(np.mean(delta ** 2))),
            "cross_section_max_abs_change_db": float(np.max(np.abs(delta))),
            "is_one_at_a_time": one_at_a_time,
            "is_baseline": is_baseline,
            "short_label": ", ".join(label_parts) if label_parts else "baseline",
        })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = Path(args.output_dir) if args.output_dir else (
        root / "results/classroom_3d" / f"sensitivity(13_음향가정민감도분석)_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=False)
    draw_sensitivity(out / "sensitivity(01_가정별예측변화).png", cross_x, all_results,
                     cross_indices, point_indices, plt, np)
    draw_ranking(out / "sensitivity(02_민감도순위).png", rows, plt, np)

    csv_path = out / "sensitivity(03_전체시나리오).csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    one_at_time = sorted(
        [row for row in rows if row["is_one_at_a_time"] and not row["is_baseline"]],
        key=lambda row: row["cross_section_rmse_from_baseline_db"], reverse=True,
    )
    summary = {
        "stage": 13,
        "version": VERSION,
        "scope": "model-only sensitivity analysis; no fitting and no new measurement",
        "baseline": {"absorption": 0.2, "right_gain": 1.0, "relative_delay_samples": 0},
        "tested": {"absorption": list(ALPHAS), "right_gain": list(RIGHT_GAINS),
                   "relative_delay_samples": list(RELATIVE_DELAYS)},
        "delay_distance_equivalent_cm": {
            str(delay): delay / scene["fs"] * scene["speed"] * 100
            for delay in RELATIVE_DELAYS
        },
        "evaluation": {"band_hz": [250, 4000], "cross_section_y_m": anchor[1],
                       "cross_section_z_m": anchor[2], "cross_points": args.cross_points,
                       "additional_points": ["P01", "P02", "P03"]},
        "one_at_a_time_ranking": one_at_time,
        "inputs": {
            "config": {"path": str(config_path), "sha256": config_sha},
            "settings": {"path": str(settings_path), "sha256": settings_sha},
            "reference": {"path": str(reference_path), **reference_info},
        },
        "fft_sizes": fft_sizes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pyroomacoustics": pra.__version__},
        "runtime_seconds": time.perf_counter() - started,
        "interpretation_limit_ko": (
            "민감도가 크다는 것은 해당 가정을 실제로 확인할 필요가 크다는 뜻입니다. "
            "그 값이 실제 오차의 원인임을 증명하거나 최적 파라미터를 결정한 결과는 아닙니다."
        ),
    }
    (out / "summary(04_민감도해석).json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("민감도 순위 (기준 대비 가로 단면 RMSE 변화)")
    for row in one_at_time:
        print(f"  {row['short_label']}: RMSE={row['cross_section_rmse_from_baseline_db']:.4f} dB / "
              f"최대 변화={row['cross_section_max_abs_change_db']:.4f} dB")
    print("이 결과는 가정 변화에 대한 모델 민감도이며 실측 보정이나 정확도 평가가 아닙니다.")
    print(f"결과 폴더: {out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("민감도 분석 실패", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)

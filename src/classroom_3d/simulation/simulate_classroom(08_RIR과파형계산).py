"""(08) 3D 강의실에서 P02의 좌우 RIR과 동일 시험 신호의 합성 파형을 계산합니다.

configs/classroom의 07번 JSON, configs/simulation의 08번 음향 가정 JSON,
recordings/reference의 재생 원본 WAV를 읽습니다.
실측 녹음과 AI 모델은 입력으로 사용하지 않습니다. 결과는 미보정 시뮬레이션입니다.
Python 3.12 / pyroomacoustics 0.10.1에서 실행 확인.
"""

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
REFERENCE_SHA256 = "a131e473eebba2a1803d7605e535b2335f1d5d8ac5b37f1931b652fba37ca329"
DOC_URL = "https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html"
COLORS = ["#C56924", "#286AA4", "#087F8C"]


def find_project_root(start):
    start = Path(start).resolve()
    start = start.parent if start.is_file() else start
    for candidate in (start, *start.parents):
        if ((candidate / "configs" / "classroom").is_dir()
                and (candidate / "configs" / "simulation").is_dir()
                and (candidate / "results").is_dir()):
            return candidate
    raise RuntimeError("sound_scape 프로젝트 루트를 찾을 수 없습니다.")


def finite_number(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: 숫자가 필요합니다.")
    value = float(value)
    if not math.isfinite(value) or (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f"{name}: 유한한 값이 필요합니다. 허용 범위: {lower} ~ {upper}")
    return value


def integer(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name}: {lower} ~ {upper} 범위의 정수가 필요합니다.")
    return value


def load_json(path):
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def resolve_inputs(config, settings):
    if config.get("schema_version") != 1 or config.get("units") != "m":
        raise ValueError("07번 설정은 schema_version=1, units='m'이어야 합니다.")
    if settings.get("schema_version") != 1 or settings.get("status") != "assumptions_not_measured":
        raise ValueError("08번 설정은 schema_version=1, status='assumptions_not_measured'를 사용합니다.")
    room = config["room"]
    dims = [finite_number(room[k], k, .01) for k in ("width_m", "length_m", "height_m")]

    def position(values, name):
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"{name}: [x, y, z] 좌표가 필요합니다.")
        result = [finite_number(v, name) for v in values]
        if any(not 0 < v < d for v, d in zip(result, dims)):
            raise ValueError(f"{name}: 방 내부 좌표가 아닙니다. cm를 m로 변환했는지 확인하세요.")
        return result

    ids = [item["id"] for item in config["sources"] + config["receivers"]]
    if len(ids) != len(set(ids)):
        raise ValueError("07번 설정에 중복된 ID가 있습니다.")
    receiver_id = settings["receiver_id"]
    receiver = next((r for r in config["receivers"] if r["id"] == receiver_id), None)
    if receiver is None:
        raise ValueError(f"수음점 {receiver_id}가 07번 설정에 없습니다.")
    receiver = {**receiver, "position_m": position(receiver["position_m"], receiver_id)}
    fs = integer(settings["sample_rate_hz"], "sample_rate_hz", 8000, 192000)
    order = integer(settings["max_order"], "max_order", 1, 128)
    comparison = integer(settings["comparison_order"], "comparison_order", 0, order - 1)
    speed = finite_number(settings["speed_of_sound_m_s"], "speed_of_sound_m_s", 250, 450)
    alpha = {k: finite_number(settings["energy_absorption"][k], k, 0, 1)
             for k in ("walls", "floor", "ceiling")}
    sources = []
    for source in config["sources"]:
        if not isinstance(source.get("active"), bool):
            raise ValueError(f"{source['id']}: active는 true/false여야 합니다.")
        if not source["active"]:
            continue
        sid = source["id"]
        offset = finite_number(source["side_wall_distance_m"], f"{sid}.side_wall_distance_m")
        side = source["side_wall"]
        if side not in ("left", "right"):
            raise ValueError(f"{sid}: side_wall은 left/right여야 합니다.")
        x = offset if side == "left" else dims[0] - offset
        loc = position([x, source["front_wall_distance_m"], source["height_m"]], sid)
        assumption = settings["sources"][sid]
        gain = finite_number(assumption["relative_gain"], f"{sid}.relative_gain", 0)
        delay = integer(assumption["extra_delay_samples"], f"{sid}.extra_delay_samples", 0, fs)
        if math.dist(loc, receiver["position_m"]) <= 1e-6:
            raise ValueError("음원과 수음점이 같은 위치입니다.")
        sources.append({"id": sid, "position_m": loc, "relative_gain": gain,
                        "extra_delay_samples": delay})
    if len(sources) not in (1, 2) or not any(s["relative_gain"] > 0 for s in sources):
        raise ValueError("이 단계에서는 활성 음원 1~2개와 0보다 큰 상대 출력이 필요합니다.")
    if config.get("furniture", {}).get("objects"):
        raise ValueError("08번 ShoeBox에는 가구 객체가 구현되지 않았습니다. 객체가 있는 설정을 조용히 생략하지 않습니다.")
    return {"dimensions_m": dims, "receiver": receiver, "sources": sources,
            "fs": fs, "speed": speed, "order": order, "comparison_order": comparison,
            "alpha": alpha}


def read_reference(path, fs, np):
    """이번 실험의 16-bit PCM mono 또는 동일한 L/R stereo를 허용합니다."""
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
            raise ValueError("시험 신호는 16-bit PCM WAV여야 합니다. 녹음 파일 대신 재생 원본을 지정하세요.")
        channels, rate, frames = wav.getnchannels(), wav.getframerate(), wav.getnframes()
        if rate != fs or channels not in (1, 2):
            raise ValueError(f"시험 신호 {rate} Hz/{channels}채널: 설정과 같은 {fs} Hz, 1~2채널이 필요합니다.")
        if not fs <= frames <= 300 * fs:
            raise ValueError("시험 신호 길이는 1~300초여야 합니다.")
        pcm = np.frombuffer(wav.readframes(frames), dtype="<i2").reshape(-1, channels)
    if len(pcm) != frames:
        raise ValueError("WAV 헤더와 실제 데이터 길이가 다릅니다.")
    if channels == 2 and not np.array_equal(pcm[:, 0], pcm[:, 1]):
        raise ValueError("좌우가 다른 스테레오 파일입니다. 이번 동일 신호 가정과 맞지 않습니다.")
    x = pcm[:, 0].astype(np.float64) / 32768.0
    if not np.any(x):
        raise ValueError("시험 신호가 전부 무음입니다.")
    info = {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sample_rate_hz": rate, "channels": channels, "pcm_bits": 16,
            "duration_s": frames / rate, "channel_rule": "mono or exactly identical L/R; use channel 1"}
    info["matches_original_06_reference"] = info["sha256"] == REFERENCE_SHA256
    return x, info


def active_blocks(x, fs, np):
    """재생 원본의 10ms 단위 비무음 구간을 찾습니다. 녹음의 음성/SNR 판정은 아닙니다."""
    hop = max(1, round(.01 * fs))
    padded = np.pad(x, (0, (-len(x)) % hop))
    active = np.max(np.abs(padded.reshape(-1, hop)), axis=1) > 0
    edges = np.diff(np.r_[False, active, False].astype(int))
    return [(int(a * hop), min(int(b * hop), len(x)))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if (b - a) * hop >= 3 * fs]


def build_room(scene, order, pra, np, signal=None):
    a = scene["alpha"]
    materials = pra.make_materials(west=a["walls"], east=a["walls"], south=a["walls"],
                                   north=a["walls"], floor=a["floor"], ceiling=a["ceiling"])
    room = pra.ShoeBox(scene["dimensions_m"], fs=scene["fs"], materials=materials,
                       max_order=order, ray_tracing=False, air_absorption=False,
                       use_rand_ism=False, sigma2_awgn=None)
    room.set_sound_speed(scene["speed"])
    for source in scene["sources"]:
        # 추가 지연은 아래 파형 합성에서 정수 샘플로 적용합니다. 전파 지연은 RIR에 있습니다.
        room.add_source(source["position_m"],
                        signal=None if signal is None else signal * source["relative_gain"], delay=0)
    room.add_microphone_array(np.array(scene["receiver"]["position_m"], dtype=float)[:, None])
    room.compute_rir()
    return room


def padded_rows(arrays, np):
    result = np.zeros((len(arrays), max(map(len, arrays))), dtype=np.float64)
    for i, array in enumerate(arrays):
        result[i, :len(array)] = array
    return result


def effective_rirs(room, scene, np):
    return padded_rows([np.pad(np.asarray(h, dtype=float) * s["relative_gain"],
                                (s["extra_delay_samples"], 0))
                        for h, s in zip(room.rir[0], scene["sources"])], np)


def synthesize(x, rirs, fftconvolve, np):
    individual = np.stack([fftconvolve(x, h, mode="full") for h in rirs])
    return individual, individual.sum(axis=0)


def mean_square(x, np):
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def db_ratio(a, b):
    return 10 * math.log10(a / b) if a > 0 and b > 0 else None


def envelope(ax, array, fs, color, np):
    hop = max(1, int(math.ceil(len(array) / 2200)))
    blocks = np.pad(array, (0, (-len(array)) % hop)).reshape(-1, hop)
    t = (np.arange(len(blocks)) * hop + hop / 2) / fs
    ax.fill_between(t, blocks.min(axis=1), blocks.max(axis=1), color=color, alpha=.8, linewidth=.3)


def plot_rirs(scene, raw_rirs, direct_info, filter_delay, plt, np):
    fig, axes = plt.subplots(len(raw_rirs), 2, figsize=(12, 3 * len(raw_rirs) + 1), squeeze=False)
    fs = scene["fs"]
    bound = max(float(np.max(np.abs(h))) for h in raw_rirs) * 1.1
    duration_ms = max(map(len, raw_rirs)) / fs * 1000
    for i, (h, source, timing) in enumerate(zip(raw_rirs, scene["sources"], direct_info)):
        t = np.arange(len(h)) / fs * 1000
        for j in range(2):
            ax = axes[i, j]
            ax.plot(t, h, color=COLORS[i], lw=.65)
            ax.set(ylim=(-bound, bound), xlabel="Raw RIR time (ms)", ylabel="Model amplitude (a.u.)")
            ax.grid(alpha=.18)
        axes[i, 0].set(xlim=(0, duration_ms), title=f"{source['id']} to {scene['receiver']['id']} | order {scene['order']}")
        arrival_ms = timing["raw_rir_expected_direct_peak_ms"]
        axes[i, 1].axvline(arrival_ms, ls="--", color="#444444", lw=1,
                           label=f"Direct: {arrival_ms:.3f} ms (raw time)")
        axes[i, 1].set(xlim=(max(0, arrival_ms - 2), arrival_ms + 15), title="Early arrivals (zoom)")
        axes[i, 1].legend(fontsize=8)
    fig.suptitle(f"{scene['receiver']['id']} RIRs | provisional 3D model", fontsize=15, y=.99)
    fig.text(.5, .018, f"Raw RIR includes {filter_delay / fs * 1000:.3f} ms numerical filter delay. "
             "Each panel uses a unit source; source gain / extra delay are applied later.\n"
             "Uncalibrated model amplitudes. Absorption and source characteristics are assumptions.",
             ha="center", fontsize=8.5, color="#576579")
    fig.tight_layout(rect=(0, .08, 1, .95))
    return fig


def plot_waveforms(scene, x, individual, total, blocks, plt, np):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fs = scene["fs"]
    envelope(axes[0], x, fs, "#667488", np)
    axes[0].set(title="Playback reference | identical signal for both sources", ylabel="Input amplitude")
    for a, b in blocks:
        axes[0].axvspan(a / fs, b / fs, color="#B8D8E1", alpha=.18)
    for i, source in enumerate(scene["sources"]):
        hop = int(round(.1 * fs))
        y = individual[i]
        yblocks = np.pad(y, (0, (-len(y)) % hop)).reshape(-1, hop)
        axes[1].plot((np.arange(len(yblocks)) + .5) * hop / fs,
                     np.sqrt(np.mean(yblocks ** 2, axis=1)), color=COLORS[i],
                     label=f"{source['id']} contribution", lw=1.4)
    axes[1].set(title="Separate contributions | 100 ms RMS", ylabel="Model RMS (a.u.)")
    axes[1].legend(loc="upper right", fontsize=9)
    envelope(axes[2], total, fs, COLORS[2], np)
    axes[2].set(title=f"Summed pressure waveform at {scene['receiver']['id']}", ylabel="Model amplitude (a.u.)")
    start = blocks[0][0] / fs + 2 if blocks else max(0, min(1, len(x) / fs - .03))
    lo, hi = int(round(start * fs)), int(round((start + .02) * fs))
    t = np.arange(lo, hi) / fs
    for i, source in enumerate(scene["sources"]):
        axes[3].plot(t, individual[i, lo:hi], color=COLORS[i], lw=.85, alpha=.8, label=source["id"])
    axes[3].plot(t, total[lo:hi], color=COLORS[2], lw=1.3, label="Sum")
    axes[3].set(title="20 ms detail | signals summed with their timing preserved", ylabel="Model amplitude (a.u.)")
    axes[3].legend(loc="upper right", fontsize=8, ncol=3)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.set_xlabel("Time from playback file start (s)")
    for ax in axes[:3]:
        ax.set_xlim(0, len(total) / fs)
    fig.suptitle(f"{scene['receiver']['id']} waveforms | assumed source gains and absorption", fontsize=15, y=.99)
    fig.text(.5, .015, "Plots show raw simulation values before listening gain. "
             "No phone recording or calibrated SPL is shown.", ha="center", fontsize=9, color="#576579")
    fig.tight_layout(rect=(0, .04, 1, .955))
    return fig


def write_listening_wavs(out, scene, individual, total, np):
    # 모든 청취용 파일에 공통 배율을 적용해 좌우 출력의 상대적 차이를 유지합니다.
    peak = max(float(np.max(np.abs(individual))), float(np.max(np.abs(total))))
    gain = 10 ** (-6 / 20) / peak
    files = []
    for source, values in zip(scene["sources"], individual):
        files.append((f"listen_{source['id']}(06_개별소리_청취용).wav", values))
    files.append(("listen_sum(07_합성소리_청취용).wav", total))
    for name, values in files:
        pcm = np.rint(values * gain * 32767).astype("<i2")
        with wave.open(str(out / name), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(scene["fs"])
            wav.writeframes(pcm.tobytes())
    return {"common_gain": gain, "maximum_peak_target_dbfs": -6,
            "files": [name for name, _ in files],
            "note_ko": "청취용 WAV에만 공통 배율을 적용했습니다. 원시 NPZ와 그래프에는 미적용입니다. WAV 음량은 실측 SPL 비교에 사용하지 않습니다."}


def run(args, np, scipy, pra, plt):
    from scipy.signal import fftconvolve

    config, config_sha = load_json(args.config)
    settings, settings_sha = load_json(args.settings)
    scene = resolve_inputs(config, settings)
    x, reference = read_reference(args.reference, scene["fs"], np)
    blocks = active_blocks(x, scene["fs"], np)
    fs = scene["fs"]
    print(f"(08) 강의실 RIR·파형 계산 v{VERSION}", flush=True)
    print(f"pyroomacoustics {pra.__version__} / {fs} Hz / {scene['receiver']['id']} {scene['receiver']['position_m']}")
    print(f"사용 스피커: {', '.join(s['id'] for s in scene['sources'])}")
    print(f"가정: 흡음률 {scene['alpha']} / 무지향성 / 가구 생략 / 실측 보정 전")
    for s in scene["sources"]:
        print(f"  {s['id']}: 상대 출력={s['relative_gain']}, 추가 지연={s['extra_delay_samples']} samples")
    print(f"재생 원본 확인: {'06번 원본과 동일' if reference['matches_original_06_reference'] else '다른 WAV 사용 — 파일/해시 기록'}")
    if pra.__version__ != "0.10.1":
        print("참고: 이 코드는 pyroomacoustics 0.10.1에서 검사했습니다. 현재 버전을 결과에 기록합니다.")
    if not reference["matches_original_06_reference"]:
        print("참고: 실제 녹음 WAV를 넣으면 녹음된 잔향이 다시 적용됩니다. 재생 원본인지 확인하세요.")

    t0 = time.perf_counter()
    direct = build_room(scene, 0, pra, np)
    filter_delay = pra.constants.get("frac_delay_length") // 2
    timing = []
    for source, h in zip(scene["sources"], direct.rir[0]):
        distance = math.dist(source["position_m"], scene["receiver"]["position_m"])
        travel = distance / scene["speed"]
        peak = int(np.argmax(np.abs(h)))
        error_samples = peak - filter_delay - travel * fs
        passed = abs(error_samples) <= 1.0
        timing.append({"source_id": source["id"], "distance_m": distance,
                       "physical_travel_ms": 1000 * travel,
                       "raw_rir_expected_direct_peak_ms": 1000 * (travel + filter_delay / fs),
                       "direct_only_peak_index": peak, "error_samples": error_samples, "passed": passed})
        print(f"직접음 검사 {source['id']}: {distance:.4f} m / {travel * 1000:.3f} ms / "
              f"{'PASS' if passed else 'FAIL'} (오차 {error_samples:+.3f} sample)")
    if not all(row["passed"] for row in timing):
        raise RuntimeError("직접음 시간 검사가 실패했습니다. 이 실행의 결과를 사용하지 마세요.")

    print(f"RIR 계산 중: 반사 차수 {scene['order']} 및 비교 차수 {scene['comparison_order']}", flush=True)
    room = build_room(scene, scene["order"], pra, np, x)
    comparison_room = build_room(scene, scene["comparison_order"], pra, np)
    raw_rirs = [np.asarray(h, dtype=float) for h in room.rir[0]]
    rirs = effective_rirs(room, scene, np)
    comparison_rirs = effective_rirs(comparison_room, scene, np)
    individual, total = synthesize(x, rirs, fftconvolve, np)
    _, comparison_total = synthesize(x, comparison_rirs, fftconvolve, np)

    # 독립 검사: PRA의 simulate()가 만든 각 기여 신호와 직접 컨볼루션 결과를 비교합니다.
    premix = room.simulate(return_premix=True)
    source_errors = []
    for i, source in enumerate(scene["sources"]):
        expected = np.pad(premix[i, 0], (source["extra_delay_samples"], 0))
        a, b = padded_rows([individual[i], expected], np)
        error = float(np.max(np.abs(a - b)))
        source_errors.append(error)
        if not np.allclose(a, b, rtol=1e-9, atol=1e-10):
            raise RuntimeError("컨볼루션 결과와 PRA의 source별 simulate 결과가 일치하지 않습니다.")
    via_combined_rir = fftconvolve(x, rirs.sum(axis=0))
    sum_error = float(np.max(np.abs(total - via_combined_rir)))
    if not np.allclose(total, via_combined_rir, rtol=1e-9, atol=1e-10):
        raise RuntimeError("두 RIR의 합과 개별 출력의 합이 일치하지 않습니다.")
    print(f"파형 합산 검사: PASS / 최대 수치 차이 {sum_error:.3g}")
    del premix

    core_windows = [(a + fs, b - fs) for a, b in blocks]
    level_rows = []
    for i, (a, b) in enumerate(core_windows, start=1):
        main_power = mean_square(total[a:b], np)
        comparison_power = mean_square(comparison_total[a:b], np)
        row = {"block": i, "start_s": a / fs, "end_s": b / fs,
               "main_model_mean_square": main_power,
               "comparison_model_mean_square": comparison_power,
               "main_minus_comparison_db": db_ratio(main_power, comparison_power)}
        level_rows.append(row)
        change = row["main_minus_comparison_db"]
        print(f"차수 비교 ({scene['comparison_order']}→{scene['order']}): "
              f"{a / fs:g}~{b / fs:g}초 합성 레벨 변화=" + (f"{change:+.4f} dB" if change is not None else "계산 불가"))
    rir_changes = [db_ratio(float(np.sum(a * a)), float(np.sum(b * b)))
                   for a, b in zip(raw_rirs, comparison_room.rir[0])]
    elapsed = time.perf_counter() - t0

    out = args.output_root.resolve() / ("classroom_audio(08_RIR과파형계산)_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "simulation(01_원시RIR과파형).npz",
                        sample_rate_hz=np.int64(fs), source_ids=np.array([s["id"] for s in scene["sources"]]),
                        rir_raw=padded_rows(raw_rirs, np), rir_lengths=np.array(list(map(len, raw_rirs))),
                        rir_effective=rirs, rir_comparison_effective=comparison_rirs,
                        input_signal=x.astype(np.float32), individual_signals=individual.astype(np.float32),
                        summed_signal=total.astype(np.float32), comparison_summed_signal=comparison_total.astype(np.float32),
                        numerical_filter_delay_samples=np.int64(filter_delay))
    figures = [plot_rirs(scene, raw_rirs, timing, filter_delay, plt, np),
               plot_waveforms(scene, x, individual, total, blocks, plt, np)]
    for fig, name in zip(figures, ["rir(02_좌우스피커응답).png", "waveform(03_입력과합성파형).png"]):
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
    listening = write_listening_wavs(out, scene, individual, total, np)
    limits = [
        "흡음률·음속·좌우 상대 출력·추가 지연은 가정값입니다. 07번 측정 조건 파일은 수정하지 않았습니다.",
        "직사각형 3D 방의 거울상 음원법 계산입니다. 가구, 회절, 실제 스피커/휴대폰 응답, AGC, 환경 소음은 포함하지 않았습니다.",
        "원시 진폭은 미보정 모델 단위입니다. Pa, dB SPL 또는 휴대폰 녹음의 dBFS와 동일하지 않습니다.",
        "차수 비교는 지정된 P02/수음점과 시험 신호에서의 민감도 확인입니다. 두 차수의 차이가 작아도 모든 위치/주파수/잔향 시간의 수렴을 보장하지 않습니다.",
        "현재 결과는 음향 계산 과정 확인용입니다. 실측 오차 평가나 기존 2D AI의 강의실 정확도 검증을 수행하지 않았습니다.",
        "NPZ 파형은 float32로 저장하고 RIR은 float64 배열로 저장했습니다. 합산 검사는 저장 전 float64 계산값 기준입니다."
    ]
    metrics = {
        "version": VERSION, "stage": "provisional_3d_rir_and_waveform", "created_at": datetime.now().astimezone().isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pyroomacoustics": pra.__version__},
        "config_sha256": config_sha, "settings_sha256": settings_sha,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_snapshot": config, "assumptions_snapshot": settings, "resolved_scene": scene,
        "reference": reference, "numerical_filter_delay_samples": filter_delay,
        "direct_only_timing_checks": timing,
        "signal_checks": {"pra_simulate_max_abs_errors": source_errors, "sum_linearity_max_abs_error": sum_error, "passed": True},
        "order_comparison": {"main_order": scene["order"], "comparison_order": scene["comparison_order"],
                             "raw_rir_energy_change_db_per_source": rir_changes, "synthesized_signal_windows": level_rows,
                             "convergence_claim": False},
        "simulation_and_checks_seconds": elapsed, "listening_export": listening,
        "measurements_compared": False, "calibrated_spl": False, "limitations_ko": limits,
        "documentation": DOC_URL,
    }
    (out / "metrics(04_가정과계산검사).json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["(08) 강의실 RIR·파형 계산 결과", "", f"수음점: {scene['receiver']['id']} {scene['receiver']['position_m']} m",
             f"활성 스피커: {', '.join(s['id'] for s in scene['sources'])}",
             f"반사 차수: {scene['order']} / 비교: {scene['comparison_order']}",
             "직접음 시간 검사: PASS / 컨볼루션 및 합산 검사: PASS", ""]
    lines += [f"{r['source_id']}: 거리 {r['distance_m']:.4f} m, 물리적 직접음 도달 {r['physical_travel_ms']:.3f} ms" for r in timing]
    lines += ["", "두 음원의 파형을 먼저 더한 후 에너지를 계산합니다. dB 값은 더하지 않습니다.",
              f"청취용 WAV 공통 배율: {listening['common_gain']:.8g}. 실제 재생 음압과 대응하지 않습니다.", ""]
    lines += [f"{r['start_s']:g}~{r['end_s']:g}초: 비교 차수 대비 합성 레벨 변화 {r['main_minus_comparison_db']:+.4f} dB"
              for r in level_rows if r['main_minus_comparison_db'] is not None]
    lines += ["", *limits, "", "먼저 rir(02_좌우스피커응답).png와 waveform(03_입력과합성파형).png를 확인하세요.",
              "listen_sum(07_합성소리_청취용).wav는 합성 결과 청취용이며 다음 현장 실험의 재생 원본으로 쓰지 않습니다."]
    (out / "summary(05_결과읽는법).txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("완료: RIR 그래프 / 합성 파형 / 원시 NPZ / 청취용 WAV 저장")
    print("현재 결과는 가정에 따른 시뮬레이션입니다. 실제 강의실과의 오차를 평가한 결과가 아닙니다.")
    print(f"결과 폴더: {out}")
    return 0


def main():
    project_root = find_project_root(__file__)
    default_reference = project_root / "recordings" / "reference" / REFERENCE_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "classroom" / CONFIG_NAME)
    parser.add_argument("--settings", type=Path, default=project_root / "configs" / "simulation" / SETTINGS_NAME)
    parser.add_argument("--reference", type=Path, default=default_reference)
    parser.add_argument("--output-root", type=Path, default=project_root / "results" / "classroom_3d")
    args = parser.parse_args()
    try:
        import numpy as np
        import scipy
        import pyroomacoustics as pra
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"라이브러리 확인 필요: {exc}\nsound_scape의 .venv Python으로 실행하세요.", file=sys.stderr)
        return 1
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    try:
        return run(args, np, scipy, pra, plt)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, wave.Error) as exc:
        print(f"실행 중단: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

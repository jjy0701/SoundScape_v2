"""(06) WAV 녹음품질검사: 파형, 디지털 레벨, 기준 신호 반복 후보.

프로젝트의 recordings 폴더에 녹음 WAV를 넣고 실행하세요.
    python "check_recording(06_녹음품질검사).py"
파일 직접 지정:
    python "check_recording(06_녹음품질검사).py" --input "recordings/실험.wav"
--reference로 재생 원본 WAV 지정 가능. 생략 시 recordings/reference의
test_signal(06_예비녹음_50초).wav가 있으면 사용합니다.

의존성: numpy, scipy, matplotlib (Pyroomacoustics/PyTorch 학습 환경에서 실행).
정수 PCM WAV 8/16/24/32 bit 지원. 원본 파일과 음량을 변경하지 않습니다.
출력: results/recording_check(06_녹음품질검사)_날짜시간 아래 PNG, JSON, TXT.

주의: dBFS는 녹음 파일의 디지털 크기이며 dB SPL이 아닙니다.
반복 후보/상관 임계값은 탐색용이며 실측의 합격 판정이나 RIR 추정이 아닙니다.
AGC 상태, 사용 스피커 수, 물리 좌표는 녹음 파일만으로 확인할 수 없습니다.
"""

import argparse
import hashlib
import json
import math
import sys
import wave
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    from scipy import signal
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        f"현재 Python 환경에 {exc.name}가 없습니다. 프로젝트 .venv로 실행하세요.\n"
        f'해당 환경에서 설치: python -m pip install {exc.name}'
    )

VERSION = "1.0"
REFERENCE_NAME = "test_signal(06_예비녹음_50초).wav"


def find_project_root(start):
    start = Path(start).resolve()
    start = start.parent if start.is_file() else start
    for candidate in (start, *start.parents):
        if ((candidate / "configs" / "classroom").is_dir()
                and (candidate / "configs" / "simulation").is_dir()
                and (candidate / "results").is_dir()):
            return candidate
    raise RuntimeError("sound_scape 프로젝트 루트를 찾을 수 없습니다.")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def power_db(value):
    """0 에너지는 -inf 대신 JSON null. 계산용 floor를 넣지 않습니다."""
    return float(10 * np.log10(value)) if value > 0 else None


def fmt(value, digits=3):
    return "N/A (무음 포함)" if value is None else f"{value:.{digits}f}"


def read_pcm(path):
    """정수 PCM을 full scale=1인 float64로 변환. 채널은 합치지 않습니다."""
    with wave.open(str(path), "rb") as reader:
        sr, channels = reader.getframerate(), reader.getnchannels()
        width, frames = reader.getsampwidth(), reader.getnframes()
        if reader.getcomptype() != "NONE":
            raise ValueError("압축 WAV는 지원하지 않습니다")
        raw = reader.readframes(frames)
    if frames < 1 or sr < 1 or channels < 1:
        raise ValueError("빈 WAV 또는 잘못된 헤더입니다")
    if len(raw) != frames * channels * width:
        raise ValueError("WAV 헤더보다 실제 데이터가 짧습니다")
    if width == 1:
        integers = np.frombuffer(raw, np.uint8).astype(np.int64) - 128
    elif width in (2, 4):
        integers = np.frombuffer(raw, dtype=f"<i{width}").astype(np.int64)
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int64)
        integers = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        integers = (integers ^ (1 << 23)) - (1 << 23)
    else:
        raise ValueError(f"지원하지 않는 정수 PCM 비트 수: {width * 8}")
    integers = integers.reshape(-1, channels)
    full_scale = 1 << (8 * width - 1)
    x = integers.astype(np.float64) / full_scale
    channel_info = []
    for c in range(channels):
        y = x[:, c]
        rail = (integers[:, c] == -full_scale) | (integers[:, c] == full_scale - 1)
        channel_info.append({
            "channel": c + 1,
            "peak_dbfs": power_db(float(np.max(np.abs(y))) ** 2),
            "whole_file_rms_dbfs": power_db(float(np.mean(y * y))),
            "dc_mean": float(y.mean()),
            "full_scale_code_samples": int(rail.sum()),
            "near_full_scale_samples_ge_minus_0_1_dbfs": int(
                (np.abs(y) >= 10 ** (-0.1 / 20)).sum()
            ),
        })
    info = {
        "file_name": path.name, "sha256": sha256(path),
        "sample_rate_hz": sr, "channels": channels, "pcm_bits": 8 * width,
        "frames": frames, "duration_s": frames / sr, "channel_metrics": channel_info,
    }
    return x, sr, info


def frame_power(x, sr, seconds=0.1):
    step = max(1, round(sr * seconds))
    starts = np.arange(0, len(x), step)
    ends = np.minimum(starts + step, len(x))
    sums = np.add.reduceat(x * x, starts)
    power = sums / (ends - starts)
    return (starts + ends) / (2 * sr), power


def reference_template(path):
    """재생 원본의 첫 1초 이상 비무음 구간. 디지털 무음으로 구분된 신호용."""
    audio, sr, info = read_pcm(path)
    mono = audio[:, 0]  # 스테레오를 평균하지 않고 재생 원본의 채널1을 사용
    step = max(1, round(sr * 0.01))
    _, power = frame_power(mono, sr, 0.01)
    active = power > 1e-14  # 디지털 무음 구분용 고정 RMS -140 dBFS
    changes = np.diff(np.r_[False, active, False].astype(int))
    runs = [(int(a * step), min(int(b * step), len(mono)))
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
            if (b - a) * step >= sr]
    if not runs:
        raise ValueError("기준 WAV에서 1초 이상의 비무음 구간을 찾지 못했습니다")
    a, b = runs[0]
    template = mono[a:b]
    if np.var(template) <= 0:
        raise ValueError("기준 구간이 상수/DC라 상관 비교에 쓸 수 없습니다")
    info.update({
        "analysis_channel": 1, "template_start_s": a / sr, "template_end_s": b / sr,
        "template_duration_s": (b - a) / sr,
        "non_silent_runs_s": [[lo / sr, hi / sr] for lo, hi in runs],
        "other_runs_identical_to_first": [
            bool(np.array_equal(template, mono[lo:hi])) for lo, hi in runs[1:]
        ],
        "stereo_channels_identical": bool(np.array_equal(audio[:, 0], audio[:, 1]))
        if audio.shape[1] == 2 else None,
        "note": "채널1을 기준으로 사용. 음원 상관은 신호 대응 후보이며 스피커 수를 나타내지 않음",
    })
    return template, sr, info


def match_reference(x, sr, template, reference_sr, threshold):
    """샘플레이트를 맞춘 뒤 정규화된 상호상관으로 반복 후보를 찾습니다."""
    common = math.gcd(sr, reference_sr)
    r = signal.resample_poly(template, sr // common, reference_sr // common)
    r = r - r.mean()
    n = len(r)
    if len(x) < n:
        return {"status": "recording_shorter_than_template", "candidates": [],
                "level_spread_db": None, "best_abs_correlation": None}
    energy = float(np.dot(r, r))
    sums = np.r_[0.0, np.cumsum(x, dtype=np.float64)]
    sums2 = np.r_[0.0, np.cumsum(x * x, dtype=np.float64)]
    window_energy = (sums2[n:] - sums2[:-n]) - (sums[n:] - sums[:-n]) ** 2 / n
    dots = signal.correlate(x, r, mode="valid", method="fft")
    denom = np.sqrt(np.maximum(window_energy, 0) * energy)
    corr = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 1e-20)
    corr = np.clip(corr, -1.0, 1.0)
    absolute = np.abs(corr)
    # Padding makes exact start/end matches eligible, too.
    peaks, _ = signal.find_peaks(np.r_[-1.0, absolute, -1.0], height=threshold,
                                 distance=max(1, round(0.8 * n)))
    peaks = (peaks - 1).astype(int)
    # Keep strongest non-overlapping candidates; threshold is an explicit heuristic.
    kept = []
    for index in sorted(peaks, key=lambda i: absolute[i], reverse=True):
        if all(abs(int(index) - j) >= n for j in kept):
            kept.append(int(index))
    candidates = []
    trim = min(sr, round(n * 0.1))
    for start in sorted(kept):
        end = start + n
        lo, hi = start + trim, end - trim
        core = x[lo:hi]
        level = power_db(float(np.mean(core * core)))
        pre_lo, pre_hi = start - 3 * sr, start - sr
        pre = x[pre_lo:pre_hi] if pre_lo >= 0 else np.array([])
        pre_level = power_db(float(np.mean(pre * pre))) if len(pre) else None
        candidates.append({
            "candidate": len(candidates) + 1,
            "start_s": start / sr, "end_s": end / sr,
            "signed_normalized_correlation": float(corr[start]),
            "absolute_normalized_correlation": float(absolute[start]),
            "core_start_s": lo / sr, "core_end_s": hi / sr,
            "core_rms_dbfs": level,
            "preceding_window_start_s": pre_lo / sr if len(pre) else None,
            "preceding_window_end_s": pre_hi / sr if len(pre) else None,
            "preceding_window_rms_dbfs": pre_level,
            "signal_plus_background_minus_preceding_level_db":
                level - pre_level if level is not None and pre_level is not None else None,
        })
    levels = [c["core_rms_dbfs"] for c in candidates if c["core_rms_dbfs"] is not None]
    return {
        "status": "candidates_found" if candidates else "no_confident_match",
        "threshold_abs_correlation": threshold,
        "best_abs_correlation": float(absolute.max()),
        "candidates": candidates,
        "start_intervals_s": np.diff([c["start_s"] for c in candidates]).tolist(),
        "level_spread_db": float(max(levels) - min(levels)) if len(levels) >= 2 else None,
        "note": "상관값은 음질 점수 아님. 지연은 녹음 시작 시차 포함. 앞 구간이 배경소음인지 수동 확인 필요. AGC 유무 판정 불가",
    }


def plot_recording(out, index, x, sr, info, comparison):
    candidates = comparison.get("candidates", [])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), layout="constrained")
    # min/max envelope keeps brief peaks that simple stride sampling would miss.
    step = max(1, math.ceil(len(x) / 5000))
    starts = np.arange(0, len(x), step)
    t = (starts + np.minimum(starts + step, len(x))) / (2 * sr)
    lower, upper = np.minimum.reduceat(x, starts), np.maximum.reduceat(x, starts)
    axes[0].fill_between(t, lower, upper, color="#2266A4", linewidth=0.4)
    peak = max(float(np.max(np.abs(x))), 1e-3)
    axes[0].set(ylim=(-1.12 * peak, 1.12 * peak), ylabel="Amplitude (full scale = 1)",
                title="Waveform envelope (vertical scale fits this recording)")
    time_axis, power = frame_power(x, sr)
    levels = 10 * np.log10(np.maximum(power, 1e-12))
    axes[1].plot(time_axis, levels, color="#2266A4", linewidth=1)
    axes[1].set(ylabel="RMS (dBFS)", ylim=(-100, 0),
                title="100 ms digital RMS | display floor -120 dBFS | not dB SPL")
    for i, candidate in enumerate(candidates):
        for axis in axes[:2]:
            axis.axvspan(candidate["start_s"], candidate["end_s"], color="#16A085", alpha=.12)
        midpoint = (candidate["core_start_s"] + candidate["core_end_s"]) / 2
        axes[1].text(midpoint, -7, f"#{i + 1}: {candidate['core_rms_dbfs']:.3f} dBFS",
                     ha="center", fontsize=9)
        axes[1].hlines(candidate["core_rms_dbfs"], candidate["core_start_s"],
                       candidate["core_end_s"], color="#087F5B", linewidth=2)
    nperseg = min(1024, len(x))
    freq, time_spec, psd = signal.spectrogram(x, fs=sr, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, detrend=False, scaling="density", mode="psd")
    keep = freq <= min(8000, sr / 2)
    spec_db = 10 * np.log10(np.maximum(psd[keep], 1e-12))
    image = axes[2].pcolormesh(time_spec, freq[keep], spec_db, shading="auto",
                              vmin=-110, vmax=-40, cmap="viridis")
    axes[2].set(xlabel="Time from start of recording (s)", ylabel="Frequency (Hz)",
                title="Spectrogram | colors can saturate outside fixed scale")
    fig.colorbar(image, ax=axes[2], label="dB re 1 FS²/Hz", pad=.01)
    for axis in axes:
        axis.set_xlim(0, len(x) / sr)
    fig.suptitle(f"Recording {index:02d} | {sr:,} Hz | {info['pcm_bits']}-bit PCM | "
                 f"channel {info['analysis_channel']}/{info['channels']} | {len(x)/sr:.2f} s")
    path = out / f"recording_{index:02d}(파형과구간).png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def report_text(report):
    lines = ["(06) 녹음품질검사 결과", "", "dBFS = 파일의 디지털 크기. 절대 SPL이나 AI 오차가 아닙니다.",
             "이 결과는 자동 검출 후보이며 녹음 조건은 별도 확인해야 합니다.", ""]
    for item in report["recordings"]:
        if item.get("error"):
            lines.extend([f"{item['file_name']}: 읽기 실패: {item['error']}", ""])
            continue
        lines.extend([f"파일: {item['file_name']}",
                      f"형식: {item['sample_rate_hz']} Hz / {item['channels']}채널 / "
                      f"{item['pcm_bits']} bit / {item['duration_s']:.2f}초"])
        for channel in item["channel_metrics"]:
            lines.append(f"  채널 {channel['channel']}: peak {fmt(channel['peak_dbfs'])} dBFS / "
                         f"디지털 최대·최소 코드 {channel['full_scale_code_samples']}개")
        comparison = item["reference_comparison"]
        found = comparison.get("candidates", [])
        lines.append(f"기준 신호 대응 후보: {len(found)}개 (상태: {comparison['status']})")
        for c in found:
            lines.append(f"  #{c['candidate']}: 약 {c['start_s']:.3f}~{c['end_s']:.3f}초, "
                         f"중앙 구간 RMS {fmt(c['core_rms_dbfs'])} dBFS")
        if comparison.get("level_spread_db") is not None:
            lines.append(f"  후보 중앙 구간 레벨 최대-최소: {comparison['level_spread_db']:.4f} dB")
        if item["channels"] > 1:
            lines.append(f"파형/반복 분석은 채널 {item['analysis_channel']}만 사용했습니다. 채널 평균을 내지 않았습니다.")
        lines.extend([f"그림: {item['plot_file']}", ""])
    lines.extend([
        "해석 범위",
        "- 최대·최소 코드가 0개라도 상류 장비의 포화·리미터·AGC가 없었다는 증명은 아닙니다.",
        "- 반복 간 레벨이 비슷해도 AGC(자동 음량 보정)가 꺼졌다고 결론낼 수 없습니다.",
        "- 기준 신호와 일치하지 않음은 다른 신호/부분 녹음/환경 영향 등일 수 있으며 파일 불량 판정이 아닙니다.",
        "- 두 파일은 녹음 설정·재생 음량·대역·위치가 확인되기 전 서로의 RMS로 공간 감쇠를 비교하지 않습니다.",
        "- 재생 원본에 포함된 모든 소리는 실제 소음과 섞여 녹음됩니다. 소음 제거/음량 정규화는 수행하지 않았습니다.",
        "- RIR 에너지 heatmap과 직접 비교하려면 재생 신호·분석 대역·레벨 기준을 먼저 맞춰야 합니다.",
        "확인할 실험 기록: 재생 파일, 켠 스피커 수, 음량, receiver x/y/z, 휴대폰 방향/앱 처리 설정.",
        "AGC 근거: https://developer.android.com/reference/android/media/audiofx/AutomaticGainControl",
        "WAV 지원 형식: https://docs.python.org/3/library/wave.html",
    ])
    return "\n".join(lines) + "\n"


def main():
    project_root = find_project_root(__file__)
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, help="녹음 WAV 한 개 이상")
    parser.add_argument("--reference", type=Path, help="스피커로 재생한 원본 WAV")
    parser.add_argument("--no-reference", action="store_true", help="기본 기준 신호 자동 사용 안 함")
    parser.add_argument("--channel", type=int, default=1, help="분석할 녹음 채널 (1부터, 기본 1)")
    parser.add_argument("--min-correlation", type=float, default=0.10,
                        help="대응 후보 탐색용 절대 상관 임계값, 품질 합격선 아님")
    parser.add_argument("--output-root", type=Path, default=project_root / "results" / "classroom_3d")
    args = parser.parse_args()
    if args.channel < 1 or not 0 < args.min_correlation <= 1:
        parser.error("channel>=1, 0<min-correlation<=1이어야 합니다")
    if args.no_reference and args.reference:
        parser.error("--reference와 --no-reference는 함께 쓸 수 없습니다")
    reference = args.reference
    reference_root = project_root / "recordings" / "reference"
    if reference is None and not args.no_reference and (reference_root / REFERENCE_NAME).is_file():
        reference = reference_root / REFERENCE_NAME
    paths = args.input
    if not paths:
        directory = project_root / "recordings" / "measurements"
        if not directory.is_dir():
            directory = base
        paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav"
                       and p.name != REFERENCE_NAME
                       and (reference is None or p.resolve() != reference.resolve()))
    if not paths:
        parser.error('녹음 파일이 없습니다. recordings 폴더에 WAV를 넣거나 --input "파일.wav"를 사용하세요.')
    out = args.output_root.resolve() / (
        "recording_check(06_녹음품질검사)_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    ref = reference_template(reference.resolve()) if reference else None
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "version": VERSION, "created_at": datetime.now().astimezone().isoformat(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "amplitude_reference": "digital PCM full scale=1; not pressure in pascals",
        "preprocessing": "no gain change, no denoising, no resampling of recordings; reference only resampled for correlation",
        "reference": ref[2] if ref else None,
        "recordings": [],
    }
    print("(06) 녹음품질검사 v" + VERSION, flush=True)
    failures = 0
    for index, path in enumerate(paths, 1):
        print(f"분석 중: {path.name}", flush=True)
        try:
            audio, sr, info = read_pcm(path.resolve())
            if args.channel > audio.shape[1]:
                raise ValueError(f"{audio.shape[1]}채널 파일에서 {args.channel}채널을 요청했습니다")
            x = audio[:, args.channel - 1]
            info["analysis_channel"] = args.channel
            comparison = match_reference(x, sr, ref[0], ref[1], args.min_correlation) if ref else {
                "status": "reference_not_provided", "candidates": [], "level_spread_db": None,
            }
            info["reference_comparison"] = comparison
            info["plot_file"] = plot_recording(out, index, x, sr, info, comparison)
            report["recordings"].append(info)
        except (OSError, ValueError, wave.Error, EOFError) as exc:
            report["recordings"].append({"file_name": path.name, "error": str(exc)})
            failures += 1
    report["files_failed"] = failures
    (out / "metrics(01_녹음검사수치).json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    text = report_text(report)
    (out / "summary(02_결과설명).txt").write_text(text, encoding="utf-8-sig")
    print(text)
    print(f"결과 폴더: {out}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, wave.Error, EOFError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)

"""Run only model C/order 8 with one receiver per killable process.

Existing order-7 and comparison result files are never modified.  A completed
receiver is checkpointed immediately; a timed-out receiver is recorded and the
remaining receivers are not started.
"""
from pathlib import Path
import argparse, importlib.util, json, multiprocessing as mp, os, time
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve

ROOT = Path(__file__).resolve().parent
IDS = ("P01", "P02", "P03")

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def load_compare():
    spec = importlib.util.spec_from_file_location("cmp09", ROOT / "compare_models_09_3models.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def worker(payload, queue):
    """Compute one receiver.  Parent may terminate this process at any time."""
    c = load_compare()
    scene = payload["scene"]
    order = payload["order"]
    signal = np.asarray(payload["signal"], dtype=float)
    windows = payload["windows"]
    fs = payload["fs"]
    sos = np.asarray(payload["sos"], dtype=float)
    sid = payload["receiver"]
    started = time.perf_counter()
    room = c.make_room(scene, "C", order, signal)
    parts = [fftconvolve(signal, np.asarray(r, float) * s["gain"], mode="full")
             for r, s in zip(room.rir[0], scene["sources"])]
    total = np.zeros(max(len(p) for p in parts), dtype=float)
    for p in parts: total[:len(p)] += p
    y = sosfiltfilt(sos, total)
    vals = []
    for a, b in windows:
        z = y[round(a*fs):round(b*fs)]
        vals.append(10*np.log10(float(np.mean(z*z))))
    level = float(10*np.log10(np.mean(10**(np.asarray(vals)/10))))
    row = {
        "order": order, "model": "C", "receiver": sid,
        "band_hz": [250, 4000], "level_db": level,
        "seconds": time.perf_counter()-started,
        "rir_samples_first_source": len(np.asarray(room.rir[0][0])),
        "rir_samples_all_sources": [len(np.asarray(r)) for r in room.rir[0]],
        "synthesized_samples": len(total), "wall_count": len(room.walls),
        "path_diagnostics": c.path_diagnostics(room, scene),
    }
    queue.put({"status": "completed", "row": row})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-seconds", type=float, default=120.0)
    ap.add_argument("--output-dir", type=Path, default=ROOT/"results"/"three_model_comparison_order8_C")
    ap.add_argument("--order7", type=Path, default=ROOT/"results"/"three_model_comparison_order78"/"metrics_order_7.json")
    args = ap.parse_args()
    if args.order7.exists():
        old7 = load(args.order7)
        rows7 = old7.get("rows", [])
    else:
        raise FileNotFoundError(args.order7)
    # Load the same configs/signal/window definitions as the existing script.
    c = load_compare()
    ss = importlib.util.spec_from_file_location("stage08", c.STAGE08)
    stage = importlib.util.module_from_spec(ss); ss.loader.exec_module(stage)
    cfg, settings = load(c.CONFIG), load(c.SETTINGS)
    fs = settings["sample_rate_hz"]
    signal, _ = stage.read_reference(c.REFERENCE, fs, np)
    blocks = stage.active_blocks(signal, fs, np)
    windows = [((a+fs)/fs, (b-fs)/fs) for a,b in blocks]
    sos = butter(4, [250,4000], btype="bandpass", fs=fs, output="sos")
    base = {"dims": [cfg["room"]["width_m"], cfg["room"]["length_m"], 2.67],
            "fs": fs, "speed": settings["speed_of_sound_m_s"], "alpha": .2}
    scenes = {}
    for sid in IDS:
        rec = next(v for v in cfg["receivers"] if v["id"] == sid)
        sc = {**base, "receiver": rec["position_m"], "sources": []}
        for src in cfg["sources"]:
            x = src["side_wall_distance_m"] if src["side_wall"] == "left" else base["dims"][0] - src["side_wall_distance_m"]
            sc["sources"].append({"position_m": [x, src["front_wall_distance_m"], src["height_m"]],
                                   "gain": settings["sources"][src["id"]]["relative_gain"]})
        scenes[sid] = sc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "metrics_order_8_C.json"
    result = {"pyroomacoustics": "0.10.1", "order": 8, "model": "C",
              "timeout_seconds": args.timeout_seconds, "receivers": list(IDS),
              "orders_completed": [], "rows": [], "status": "running",
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    ctx = mp.get_context("spawn")
    for sid in IDS:
        q = ctx.Queue(); payload = {"scene": scenes[sid], "order": 8,
            "signal": signal.tolist(), "windows": windows, "fs": fs,
            "sos": sos.tolist(), "receiver": sid}
        p = ctx.Process(target=worker, args=(payload, q)); p.start(); p.join(args.timeout_seconds)
        if p.is_alive():
            p.terminate(); p.join(10)
            result["status"] = "timed_out"
            result["timed_out_receiver"] = sid
            result["stopped_before"] = [x for x in IDS if x not in result["orders_completed"] and x != sid]
            checkpoint.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2)); return
        msg = q.get() if not q.empty() else None
        if not msg or msg.get("status") != "completed":
            result["status"] = "failed"; result["failed_receiver"] = sid
            checkpoint.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2)); return
        result["rows"].append(msg["row"]); result["orders_completed"].append(sid)
        # Save after every completed receiver.
        checkpoint.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    p02 = next(r["level_db"] for r in result["rows"] if r["receiver"] == "P02")
    for r in result["rows"]: r["relative_to_P02_db"] = r["level_db"] - p02
    for r in result["rows"]:
        prev = next((x for x in rows7 if x.get("model") == "C" and x.get("receiver") == r["receiver"]), None)
        if prev:
            r["relative_level_change_vs_previous_order_db"] = r["relative_to_P02_db"] - prev["relative_to_P02_db"]
            r["band_level_change_vs_previous_order_db"] = r["level_db"] - prev["level_db"]
    result["status"] = "completed"; result["p02_level_db"] = p02
    checkpoint.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()

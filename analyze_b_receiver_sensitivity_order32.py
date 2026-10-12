"""Receiver-position sensitivity for model B at order 32.

Uses the existing stage-08 signal and stage-09 analysis procedure.  Only model
B is evaluated; no C or high-order comparison result is touched.
"""
from pathlib import Path
import importlib.util, json, math, time
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IDS = ("P01", "P02", "P03")
ORDER = 32
OUT = ROOT / "results" / "receiver_sensitivity_B_order32"

def load(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def load_cmp():
    sp = importlib.util.spec_from_file_location("cmp09", ROOT / "compare_models_09_3models.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

def band_level(signal, fs, windows, sos):
    y = sosfiltfilt(sos, signal); vals=[]
    for a,b in windows:
        z=y[round(a*fs):round(b*fs)]
        vals.append(10*math.log10(float(np.mean(z*z))))
    return float(10*math.log10(np.mean(10**(np.asarray(vals)/10))))

def main():
    c=load_cmp(); OUT.mkdir(parents=True, exist_ok=True)
    ss=importlib.util.spec_from_file_location("stage08", c.STAGE08)
    stage=importlib.util.module_from_spec(ss); ss.loader.exec_module(stage)
    cfg, st=load(c.CONFIG), load(c.SETTINGS); fs=st["sample_rate_hz"]
    x,_=stage.read_reference(c.REFERENCE,fs,np); blocks=stage.active_blocks(x,fs,np)
    windows=[((a+fs)/fs,(b-fs)/fs) for a,b in blocks]
    sos=butter(4,[250,4000],btype="bandpass",fs=fs,output="sos")
    w,l=cfg["room"]["width_m"],cfg["room"]["length_m"]
    sources=[]
    for s in cfg["sources"]:
        xx=s["side_wall_distance_m"] if s["side_wall"]=="left" else w-s["side_wall_distance_m"]
        sources.append({"position_m":[xx,s["front_wall_distance_m"],s["height_m"]],
                        "gain":st["sources"][s["id"]]["relative_gain"]})
    originals={r["id"]:list(r["position_m"]) for r in cfg["receivers"] if r["id"] in IDS}
    offsets=[("x",-0.10),("x",-0.05),("x",0.05),("x",0.10),
             ("y",-0.10),("y",-0.05),("y",0.05),("y",0.10)]
    conditions=[]
    conditions.append({"condition":"original","moved_receiver":None,"axis":None,"offset_m":0.0,"positions":originals})
    for sid in IDS:
        for axis,off in offsets:
            pos={k:list(v) for k,v in originals.items()}; idx=0 if axis=="x" else 1
            pos[sid][idx]+=off
            conditions.append({"condition":f"{sid}_{axis}_{off:+.2f}m","moved_receiver":sid,"axis":axis,"offset_m":off,"positions":pos})
    rows=[]; started_all=time.perf_counter()
    for cond in conditions:
        t=time.perf_counter(); room=c.pra.ShoeBox([w,l,2.67],fs=fs,materials=c.pra.make_materials(
            west=.2,east=.2,south=.2,north=.2,floor=.2,ceiling=.2),max_order=ORDER,
            ray_tracing=False,air_absorption=False,use_rand_ism=False,sigma2_awgn=None)
        room.set_sound_speed(st["speed_of_sound_m_s"])
        for s in sources: room.add_source(s["position_m"])
        mic=np.asarray([cond["positions"][sid] for sid in IDS],dtype=float).T
        room.add_microphone_array(mic); room.compute_rir()
        levels={}
        for i,sid in enumerate(IDS):
            parts=[fftconvolve(x,np.asarray(r,float)*src["gain"],mode="full")
                   for r,src in zip(room.rir[i],sources)]
            total=np.zeros(max(map(len,parts)))
            for p in parts: total[:len(p)]+=p
            levels[sid]=band_level(total,fs,windows,sos)
        ref=levels["P02"]
        rel={sid:levels[sid]-ref for sid in IDS}
        rows.append({**cond,"levels_db":levels,"relative_db":rel,
                     "relative_change_from_original_db":None,
                     "seconds":time.perf_counter()-t})
        # checkpoint after every condition
        (OUT/"metrics_partial.json").write_text(json.dumps({"order":ORDER,"model":"B","rows":rows},ensure_ascii=False,indent=2),encoding="utf-8")
    base=rows[0]
    for r in rows:
        r["relative_change_from_original_db"]={sid:r["relative_db"][sid]-base["relative_db"][sid] for sid in ("P01","P03")}
    result={"order":ORDER,"model":"B","absorption":.2,"microphone_order":list(IDS),
            "original_positions_m":originals,"rows":rows,
            "total_seconds":time.perf_counter()-started_all,
            "note":"Order-32 baseline is independent of prior order-7 results; no C calculation performed."}
    (OUT/"metrics_receiver_sensitivity.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    # CSV-style summary for convenient inspection.
    lines=["condition,moved_receiver,axis,offset_m,P01_minus_P02_db,P03_minus_P02_db,dP01_db,dP03_db,P02_level_db,seconds"]
    for r in rows:
        lines.append(",".join(map(str,[r["condition"],r["moved_receiver"] or "",r["axis"] or "",r["offset_m"],r["relative_db"]["P01"],r["relative_db"]["P03"],r["relative_change_from_original_db"]["P01"],r["relative_change_from_original_db"]["P03"],r["levels_db"]["P02"],r["seconds"]])))
    (OUT/"summary.csv").write_text("\n".join(lines),encoding="utf-8")
    # Two-panel graph: relative levels and changes from the order-32 original.
    labels=[r["condition"] for r in rows]; xx=np.arange(len(rows));
    fig,ax=plt.subplots(2,1,figsize=(16,9),sharex=True)
    for sid,color in (("P01","tab:blue"),("P03","tab:orange")):
        ax[0].plot(xx,[r["relative_db"][sid] for r in rows],".-",label=sid,color=color)
        ax[1].plot(xx,[r["relative_change_from_original_db"][sid] for r in rows],".-",label=sid,color=color)
    ax[0].axhline(2.3554424604,color="tab:blue",ls="--",alpha=.35,label="P01 measured")
    ax[0].axhline(-2.1292925638,color="tab:orange",ls="--",alpha=.35,label="P03 measured")
    ax[0].set_ylabel("relative level (dB)"); ax[1].set_ylabel("change from original (dB)")
    ax[1].set_xlabel("condition"); ax[0].legend(ncol=4); ax[1].legend(); ax[0].grid(alpha=.2); ax[1].grid(alpha=.2)
    ax[1].set_xticks(xx); ax[1].set_xticklabels(labels,rotation=75,ha="right",fontsize=7)
    fig.suptitle("Model B receiver-position sensitivity, order 32")
    fig.tight_layout(); fig.savefig(OUT/"receiver_sensitivity_order32.png",dpi=160); plt.close(fig)
    print(json.dumps({"output_dir":str(OUT),"conditions":len(rows),"baseline":base["levels_db"],"total_seconds":result["total_seconds"]},ensure_ascii=False,indent=2))

if __name__=="__main__": main()

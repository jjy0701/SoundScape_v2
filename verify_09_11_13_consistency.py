"""Independent consistency check for stages 09, 11 and 13.

This file only reads the project inputs and writes a new verification folder.
It does not modify the original scripts or existing result folders.
"""
from pathlib import Path
import csv, importlib.util, json, math, sys, wave
import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfreqz, fftconvolve
import pyroomacoustics as pra

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "consistency_check_09_11_13"
FS = 48000
POINTS = {
    "P01": np.array([3.75, 2.5, 0.706]),
    "P02": np.array([3.75, 5.0, 0.706]),
    "P03": np.array([3.75, 7.5, 0.706]),
}

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def read_wav(path):
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes()); ch = w.getnchannels()
        assert w.getframerate() == FS and w.getsampwidth() == 2
    a = np.frombuffer(raw, dtype="<i2").reshape(-1, ch)
    assert ch in (1, 2) and (ch == 1 or np.array_equal(a[:,0], a[:,1]))
    return a[:,0].astype(float) / 32768.0

def active_core(x):
    hop = 480; p = np.pad(x, (0, (-len(x)) % hop))
    active = np.max(np.abs(p.reshape(-1, hop)), axis=1) > 0
    edges = np.diff(np.r_[False, active, False].astype(int))
    blocks = [(int(a*hop), min(int(b*hop), len(x))) for a,b in
              zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1))
              if (b-a)*hop >= 3*FS]
    a,b = blocks[0]; return x[a+FS:b-FS], (a+FS,b-FS)

def room_at(mics, config, order=32, alpha=.2):
    dims = [config["room"][k] for k in ("width_m","length_m","height_m")]
    mat = pra.make_materials(west=alpha,east=alpha,south=alpha,north=alpha,floor=alpha,ceiling=alpha)
    room = pra.ShoeBox(dims, fs=FS, materials=mat, max_order=order,
                       ray_tracing=False, air_absorption=False, use_rand_ism=False,
                       sigma2_awgn=None)
    room.set_sound_speed(343.0)
    for s in config["sources"]:
        if s["side_wall"] == "left": x = s["side_wall_distance_m"]
        else: x = dims[0] - s["side_wall_distance_m"]
        room.add_source([x,s["front_wall_distance_m"],s["height_m"]])
    room.add_microphone_array(np.asarray(mics).T); room.compute_rir(); return room

def rms_db(y): return 10*np.log10(np.mean(np.asarray(y,float)**2))

def freq_energy(h, core, nfft, sos):
    X = np.fft.rfft(core,n=nfft); f=np.fft.rfftfreq(nfft,1/FS)
    _, H=sosfreqz(sos,worN=f,fs=FS); parse=np.ones(len(f)); parse[1:-1]=2
    w=np.abs(X)**2*np.abs(H)**4*parse/nfft
    return float(np.sum(w*np.abs(np.fft.rfft(h,n=nfft))**2))

def main():
    config=json.loads((ROOT/"configs/classroom/classroom_config(07_강의실측정조건).json").read_text(encoding="utf-8"))
    ref=read_wav(ROOT/"recordings/reference/test_signal(06_예비녹음_50초).wav")
    core,(ca,cb)=active_core(ref)
    sos=butter(4,[250,4000],btype="bandpass",fs=FS,output="sos")
    room=room_at(list(POINTS.values()),config)
    nfft=1 << (len(core)+max(len(h) for m in room.rir for h in m)-2).bit_length()
    rows=[]; all_e={}
    for i,(pid,_) in enumerate(POINTS.items()):
        # Pyroomacoustics may return different RIR lengths per source.
        # Keep source RIRs as a list; convolution outputs are padded by NumPy
        # when summed below, rather than forcing a ragged list into an array.
        hs=[np.asarray(h,float) for h in room.rir[i]]
        ys=[fftconvolve(ref,h,mode="full") for h in hs]
        total=np.zeros(max(map(len,ys)), dtype=float)
        for y in ys: total[:len(y)] += y
        filtered=sosfiltfilt(sos,total)
        # 09's three 8 s core windows; identical value per repeat for this deterministic model.
        vals=[]
        for start in (6,21,36):
            a,b=round(start*FS),round((start+8)*FS); vals.append(rms_db(filtered[a:b]))
        eL=freq_energy(hs[0],core,nfft,sos); eR=freq_energy(hs[1],core,nfft,sos)
        hsum=np.zeros(max(map(len,hs)), dtype=float)
        for h in hs: hsum[:len(h)] += h
        eB=freq_energy(hsum,core,nfft,sos)
        all_e[pid]=(eL,eR,eB)
        rows.append({"point":pid,"09_db":float(np.mean(vals)),"09_repeat_spread_db":float(np.ptp(vals),),
                     "11_db":10*np.log10(eB),"13_baseline_db":10*np.log10(eB),
                     "p02_interaction_db":10*np.log10(eB/(eL+eR))})
    anchor={k: next(r for r in rows if r["point"]=="P02")[k] for k in ("09_db","11_db","13_baseline_db")}
    for r in rows:
        r["09_relative_db"]=r["09_db"]-anchor["09_db"]
        r["11_relative_db"]=r["11_db"]-anchor["11_db"]
        r["13_relative_db"]=r["13_baseline_db"]-anchor["13_baseline_db"]
        r["09_vs_11_difference_db"]=r["09_relative_db"]-r["11_relative_db"]
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    report={"conditions":{"sample_rate_hz":FS,"band_hz":[250,4000],"absorption":.2,"max_order":32,"gains":[1,1],"extra_delay_samples":0,"coordinates":{k:v.tolist() for k,v in POINTS.items()},"reference_core_samples":[ca,cb]},"rows":rows,"p02_interaction_comparison":{"exact_coordinate_db":rows[1]["p02_interaction_db"],"old_interpolated_db":1.041060954862688,"difference_db":rows[1]["p02_interaction_db"]-1.041060954862688,"metric_note":"This is coherent two-speaker interaction relative to incoherent left+right energy; it is not the P02-normalized relative level, whose value is 0 dB by definition."},"methods":{"09":"Full reference convolved with each RIR, sum waveforms, sosfiltfilt on full sum, then 8 s windows at 6-14/21-29/36-44 s; mean-square -> 10log10.","11_13":"Core 6-14 s spectrum, Butterworth response magnitude^4 (forward-backward power), Parseval weighted RIR transfer energy; coherent sum before energy; P02 both-speaker energy normalization."},"diagnosis":{"same_rir":True,"coordinates_exact":True,"interpolation_used":False,"likely_difference":"09 includes finite-window convolution/filter edge and the full reference timing; 11/13 use steady-state frequency-domain energy and omit finite-window transient/tail effects. Filter definitions are otherwise the same 4th-order Butterworth bandpass, but 11/13 apply its idealized |H|^4 weighting."}}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(rows,ensure_ascii=False,indent=2)); print("OUTPUT",OUT)

if __name__=="__main__": main()

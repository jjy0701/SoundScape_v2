"""Validate one multi-microphone Room against the preserved single-mic method.

Only orders 2 and 7 are tested. This script does not run order 8 and does not
modify the original comparison script or its result folders.
"""
from pathlib import Path
import importlib.util, json, math, time
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve

ROOT=Path(__file__).resolve().parent
IDS=("P01","P02","P03")
SRC=ROOT/"compare_models_09_3models.py"

def load(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def pad_diff(a,b):
    n=max(len(a),len(b)); aa=np.pad(np.asarray(a,float),(0,n-len(a))); bb=np.pad(np.asarray(b,float),(0,n-len(b)))
    d=aa-bb
    return {"max_abs":float(np.max(np.abs(d))),"rms":float(np.sqrt(np.mean(d*d))),"relative_rms":float(np.sqrt(np.mean(d*d))/(np.sqrt(np.mean(bb*bb))+1e-30))}

def main():
    s=importlib.util.spec_from_file_location("orig",SRC); o=importlib.util.module_from_spec(s); s.loader.exec_module(o)
    ss=importlib.util.spec_from_file_location("stage08",o.STAGE08); stage=importlib.util.module_from_spec(ss); ss.loader.exec_module(stage)
    cfg=o.load(o.CONFIG); st=o.load(o.SETTINGS); fs=st["sample_rate_hz"]
    x,_=stage.read_reference(o.REFERENCE,fs,np); blocks=stage.active_blocks(x,fs,np)
    windows=[((a+fs)/fs,(b-fs)/fs) for a,b in blocks]; sos=butter(4,[250,4000],btype="bandpass",fs=fs,output="sos")
    scenes={}
    for model,h in (("A",2.63),("B",2.67),("C",2.67)):
        for sid in IDS:
            r=next(v for v in cfg["receivers"] if v["id"]==sid)
            sc={"dims":[cfg["room"]["width_m"],cfg["room"]["length_m"],h],"fs":fs,"speed":st["speed_of_sound_m_s"],"alpha":.2,"receiver":r["position_m"],"sources":[]}
            for q in cfg["sources"]:
                xx=q["side_wall_distance_m"] if q["side_wall"]=="left" else sc["dims"][0]-q["side_wall_distance_m"]
                sc["sources"].append({"position_m":[xx,q["front_wall_distance_m"],q["height_m"]],"gain":st["sources"][q["id"]]["relative_gain"]})
            scenes[model,sid]=sc
    result={"orders":[2,7],"microphone_order":list(IDS),"models":{},"criterion_db":.001,"notes":["Order 8 is intentionally not executed.","No reusable full RIR arrays were found in the existing metrics JSON; waveform validation uses fresh preserved-method vs multi-microphone calculations.","All RIR comparisons are t=0 aligned and right-zero-padded only."]}
    for order in (2,7):
        if order == 7 and not all(result["models"].get("2",{}).get(m,{}).get("pass",False) for m in ("A","B","C")):
            result["order_7_status"]="SKIPPED: order-2 validation failed"
            break
        result["models"][str(order)]={}
        for model in ("A","B","C"):
            old_rows={}; old_seconds=0.0
            for sid in IDS:
                t=time.perf_counter(); room=o.make_room(scenes[model,sid],model,order,x); old_seconds+=time.perf_counter()-t
                old_rows[sid]={'room':room,'rir':room.rir[0]}
            t=time.perf_counter(); sc=scenes[model,"P02"]; w,l,h=sc["dims"]
            if model in ("A","B"):
                a=sc["alpha"]; room=o.pra.ShoeBox([w,l,h],fs=fs,materials=o.pra.make_materials(west=a,east=a,south=a,north=a,floor=a,ceiling=a),max_order=order,ray_tracing=False,air_absorption=False,use_rand_ism=False,sigma2_awgn=None)
            else: room=o.extruded_polygon_room(sc,order,o.air_floor_polygon(w,l),add_microphone=False)
            room.set_sound_speed(sc["speed"])
            for q in sc["sources"]: room.add_source(q["position_m"])
            # Register one explicit (dim, n_mics) array. This preserves the
            # documented microphone order P01, P02, P03 across Room/RIR axes.
            mic_locs=np.asarray([scenes[model,sid]["receiver"] for sid in IDS],dtype=float).T
            room.add_microphone_array(mic_locs)
            actual_mics=np.asarray(room.mic_array.R,dtype=float).T.tolist()
            if room.mic_array.R.shape != (3,3) or not np.allclose(room.mic_array.R,mic_locs):
                raise RuntimeError(f"{model}/order={order}: microphone registration mismatch: shape={room.mic_array.R.shape}, coords={actual_mics}")
            t_image=time.perf_counter(); room.image_source_model(); image_seconds=time.perf_counter()-t_image
            t_rir=time.perf_counter(); room.compute_rir(); rir_seconds=time.perf_counter()-t_rir
            new_total=time.perf_counter()-t
            comparisons=[]
            old_levels={}; new_levels={}
            for mi,sid in enumerate(IDS):
                parts_old=[]; parts_new=[]
                for si,q in enumerate(sc["sources"]):
                    comparisons.append({"receiver":sid,"source_index":si,"old_length":len(old_rows[sid]["rir"][si]),"new_length":len(room.rir[mi][si]),**pad_diff(old_rows[sid]["rir"][si],room.rir[mi][si])})
                    parts_old.append(fftconvolve(x,np.asarray(old_rows[sid]["rir"][si])*q["gain"],mode="full")); parts_new.append(fftconvolve(x,np.asarray(room.rir[mi][si])*q["gain"],mode="full"))
                def mix(ps):
                    y=np.zeros(max(map(len,ps)))
                    for p in ps:y[:len(p)]+=p
                    return y
                old_levels[sid]=o.level(mix(parts_old),fs,windows,sos); new_levels[sid]=o.level(mix(parts_new),fs,windows,sos)
            old_rel={sid:old_levels[sid]-old_levels["P02"] for sid in IDS}; new_rel={sid:new_levels[sid]-new_levels["P02"] for sid in IDS}
            level_rows=[{"receiver":sid,"old_band_level_db":old_levels[sid],"new_band_level_db":new_levels[sid],"band_level_difference_db":new_levels[sid]-old_levels[sid],"old_relative_db":old_rel[sid],"new_relative_db":new_rel[sid],"relative_difference_db":new_rel[sid]-old_rel[sid]} for sid in IDS]
            result["models"][str(order)][model]={"old_single_mic_seconds":old_seconds,"new_multimic_seconds":new_total,"new_image_source_seconds":image_seconds,"new_rir_seconds":rir_seconds,"microphone_order":list(IDS),"registered_microphone_count":int(room.mic_array.R.shape[1]),"registered_microphone_coordinates_m":actual_mics,"rir_index_mapping":{"rir[0]":"P01","rir[1]":"P02","rir[2]":"P03"},"rir":comparisons,"levels":level_rows,"pass":max([abs(v["relative_difference_db"]) for v in level_rows]+[0])<=.001}
    out=ROOT/"results"/"multimic_validation_09"; out.mkdir(parents=True,exist_ok=True)
    (out/"metrics_multimic_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    available=[o for o in result["orders"] if str(o) in result["models"]]
    summary={o:{m:{"old_s":result["models"][str(o)][m]["old_single_mic_seconds"],"new_s":result["models"][str(o)][m]["new_multimic_seconds"],"image_s":result["models"][str(o)][m]["new_image_source_seconds"],"rir_s":result["models"][str(o)][m]["new_rir_seconds"],"pass":result["models"][str(o)][m]["pass"]} for m in result["models"][str(o)]} for o in available}
    print(json.dumps({"orders_requested":result["orders"],"orders_completed":available,"status":result.get("order_7_status","OK"),"summary":summary},ensure_ascii=False,indent=2))
if __name__=="__main__": main()

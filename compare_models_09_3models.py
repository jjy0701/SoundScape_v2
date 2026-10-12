"""Compare the three classroom geometries using the stage-09 signal procedure.

Model A: 7.5 x 10.25 x 2.63 m ShoeBox
Model B: 7.5 x 10.25 x 2.67 m ShoeBox
Model C: 2.67 m custom 3-D Room with the four measured wall-connected blocks.

The protrusions are acoustic walls, not plotting-only objects.  The script
never changes the existing configs or results and records every requested
image-source order.  Model C is intentionally not silently downgraded.
"""
from pathlib import Path
import argparse, importlib.util, json, math, platform, time
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve
import pyroomacoustics as pra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IDS = ("P01", "P02", "P03")
ROOT = Path(__file__).resolve().parent
STAGE08 = ROOT / "support" / "simulate_classroom(08_RIR과파형계산).py"
CONFIG = ROOT / "support" / "classroom_config(07_강의실측정조건).json"
SETTINGS = ROOT / "configs" / "simulation" / "simulation_settings(08_음향가정).json"
REFERENCE = ROOT / "recordings" / "reference" / "test_signal(06_예비녹음_50초).wav"

def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def wall(corners, alpha, name):
    c = np.asarray(corners, dtype=float).T
    return pra.wall_factory(c, np.array([alpha], dtype=float), np.array([0.0]), name)

def outer_walls(w, l, h, alpha):
    # Counter-clockwise rectangular faces; each is a true 3-D Wall.
    return [
        wall([[0,0,0],[0,l,0],[0,l,h],[0,0,h]], alpha, "outer_west"),
        wall([[w,0,0],[w,0,h],[w,l,h],[w,l,0]], alpha, "outer_east"),
        wall([[0,0,0],[0,0,h],[w,0,h],[w,0,0]], alpha, "outer_south"),
        wall([[0,l,0],[w,l,0],[w,l,h],[0,l,h]], alpha, "outer_north"),
        wall([[0,0,0],[w,0,0],[w,l,0],[0,l,0]], alpha, "floor"),
        wall([[0,0,h],[0,l,h],[w,l,h],[w,0,h]], alpha, "ceiling"),
    ]

def protrusion_walls(alpha, h):
    # Only the three exposed faces are added for each block. The wall-attached
    # face coincides with the outer wall and is deliberately not duplicated.
    rects = [("left_front",0,.42,2.74,3.465),
             ("right_front",6.80,7.50,2.74,3.465),
             ("left_back",0,.42,9.20,10.25),
             ("right_back",6.80,7.50,9.20,10.25)]
    out=[]
    for name,x0,x1,y0,y1 in rects:
        out += [wall([[x1,y0,0],[x1,y1,0],[x1,y1,h],[x1,y0,h]], alpha, name+"_x"),
                wall([[x0,y0,0],[x1,y0,0],[x1,y0,h],[x0,y0,h]], alpha, name+"_front"),
                wall([[x1,y1,0],[x0,y1,0],[x0,y1,h],[x1,y1,h]], alpha, name+"_back")]
    return out

def air_floor_polygon(w, l):
    """CCW plan of the air domain; the four wall-attached blocks are not air."""
    return np.asarray([[0,0], [w,0], [w,2.74], [6.80,2.74],
                       [6.80,3.465], [w,3.465], [w,9.20],
                       [6.80,9.20], [6.80,l], [.42,l],
                       [.42,9.20], [0,9.20], [0,3.465],
                       [.42,3.465], [.42,2.74], [0,2.74]], dtype=float).T

def extruded_polygon_room(scene, order, polygon=None, add_microphone=True):
    w,l,h=scene["dims"]; a=scene["alpha"]
    if polygon is None: polygon=np.asarray([[0,0],[w,0],[w,l],[0,l]],dtype=float).T
    mat=pra.make_materials(west=a,east=a,south=a,north=a,floor=a,ceiling=a)
    # from_corners normalizes polygon orientation; extrude creates one closed
    # air-domain boundary, with no coincident hidden wall faces.
    room=pra.Room.from_corners(polygon,fs=scene["fs"],max_order=order,
                               materials=[mat["west"]]*polygon.shape[1],
                               ray_tracing=False,air_absorption=False)
    room.extrude(h,materials={"floor":mat["floor"],"ceiling":mat["ceiling"]})
    room.set_sound_speed(scene["speed"])
    for s in scene["sources"]: room.add_source(s["position_m"])
    if add_microphone:
        room.add_microphone_array(np.asarray(scene["receiver"],dtype=float)[:,None])
    if add_microphone:
        room.compute_rir()
    return room

def make_room(scene, model, order, signal=None):
    w,l,h = scene["dims"]
    a = scene["alpha"]
    if model in ("A", "B"):
        room = pra.ShoeBox([w,l,h], fs=scene["fs"], materials=pra.make_materials(
            west=a,east=a,south=a,north=a,floor=a,ceiling=a), max_order=order,
            ray_tracing=False, air_absorption=False, use_rand_ism=False, sigma2_awgn=None)
    else:
        room = extruded_polygon_room(scene,order,air_floor_polygon(w,l))
    if model in ("A","B"):
        room.set_sound_speed(scene["speed"])
        for s in scene["sources"]:
            room.add_source(s["position_m"], signal=None if signal is None else signal*s["gain"])
        room.add_microphone_array(np.asarray(scene["receiver"], dtype=float)[:,None])
        room.compute_rir()
    return room

def level(x, fs, windows, sos):
    y=sosfiltfilt(sos,x)
    vals=[]
    for a,b in windows:
        z=y[round(a*fs):round(b*fs)]
        vals.append(10*math.log10(float(np.mean(z*z))))
    return float(10*math.log10(np.mean(10**(np.asarray(vals)/10))))

def path_diagnostics(room, scene):
    """Capture path/order evidence before interpreting level changes."""
    out=[]
    visibility=getattr(room, "visibility", None)
    for i,(src,rir) in enumerate(zip(scene["sources"], room.rir[0])):
        h=np.asarray(rir,dtype=float)
        orders=np.asarray(getattr(room.sources[i], "orders", []))
        vis=None if visibility is None else np.asarray(visibility[i]).astype(int).tolist()
        out.append({"source_index":i,"source_position_m":src["position_m"],
                    "rir_samples":int(h.size),"nonzero_samples":int(np.count_nonzero(h)),
                    "peak_sample":int(np.argmax(np.abs(h))) if h.size else None,
                    "peak_abs":float(np.max(np.abs(h))) if h.size else 0.0,
                    "image_order_counts":{str(int(o)):int(np.count_nonzero(orders==o)) for o in np.unique(orders)},
                    "visible_path_count":None if vis is None else int(np.count_nonzero(vis)),
                    "orders_head":orders[:20].astype(int).tolist()})
    return out

def geometry_diagnostics(w,l,h,alpha):
    """Check the custom Room surface inventory and face orientation metadata."""
    faces=outer_walls(w,l,h,alpha)+protrusion_walls(alpha,h)
    rows=[]
    for f in faces:
        c=np.asarray(f.corners,dtype=float)
        normal=np.cross(c[:,1]-c[:,0],c[:,2]-c[:,0])
        rows.append({"name":f.name,"corner_count":int(c.shape[1]),
                     "normal_unit":(normal/np.linalg.norm(normal)).round(6).tolist(),
                     "area_proxy":float(np.linalg.norm(normal)),
                     "corners_m":c.T.tolist()})
    return {"wall_count":len(faces),"outer_wall_count":6,"protrusion_exposed_face_count":12,
            "coincident_outer_faces_retained":True,
            "closed_boundary_note":"Outer ShoeBox faces close the domain; each attached protrusion uses its three exposed faces and shares the outer wall as its fourth face.",
            "faces":rows}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--orders", nargs="+", type=int, default=[0,1,2])
    ap.add_argument("--output-dir", type=Path, default=ROOT/"results"/"three_model_comparison")
    ap.add_argument("--previous-results", type=Path, default=ROOT/"results"/"three_model_comparison"/"metrics_three_models.json")
    ap.add_argument("--timeout-seconds", type=float, default=120.0)
    args=ap.parse_args()
    if pra.__version__ != "0.10.1": raise RuntimeError(f"Expected pyroomacoustics 0.10.1, got {pra.__version__}")
    spec=importlib.util.spec_from_file_location("stage08", STAGE08); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    cfg=load(CONFIG); st=load(SETTINGS); fs=st["sample_rate_hz"]
    x,_=m.read_reference(REFERENCE,fs,np); blocks=m.active_blocks(x,fs,np)
    windows=[((a+fs)/fs,(b-fs)/fs) for a,b in blocks]
    if len(windows)!=3 or any(abs(b-a-8)>1e-9 for a,b in windows): raise RuntimeError("09-compatible three 8-second windows not found")
    sos=butter(4,[250,4000],btype="bandpass",fs=fs,output="sos")
    base={"dims":[cfg["room"]["width_m"],cfg["room"]["length_m"],2.63],"fs":fs,"speed":st["speed_of_sound_m_s"],"alpha":.2}
    scenes={}
    for model,h in (("A",2.63),("B",2.67),("C",2.67)):
        for sid in IDS:
            r=next(v for v in cfg["receivers"] if v["id"]==sid); scenes[model,sid]={**base,"dims":[base["dims"][0],base["dims"][1],h],"receiver":r["position_m"],"sources":[]}
            for s in cfg["sources"]:
                xx=s["side_wall_distance_m"] if s["side_wall"]=="left" else base["dims"][0]-s["side_wall_distance_m"]
                scenes[model,sid]["sources"].append({"position_m":[xx,s["front_wall_distance_m"],s["height_m"]],"gain":st["sources"][s["id"]]["relative_gain"]})
    rows=[]
    for order in args.orders:
        vals={}
        for model in ("A","B","C"):
            for sid in IDS:
                t=time.perf_counter(); room=make_room(scenes[model,sid],model,order,x)
                h=np.asarray(room.rir[0][0],float)
                # RIR lengths can differ by source (especially for the custom
                # 3-D Room). Convolve independently, then align at t=0 and
                # zero-pad before coherent summation.
                parts=[fftconvolve(x,np.asarray(r,float)*s["gain"],mode="full")
                       for r,s in zip(room.rir[0],scenes[model,sid]["sources"])]
                total=np.zeros(max(len(p) for p in parts),dtype=float)
                for p in parts: total[:len(p)] += p
                vals[model,sid]=level(total,fs,windows,sos)
                elapsed=time.perf_counter()-t
                if elapsed > args.timeout_seconds:
                    raise TimeoutError(f"{model}/{sid}/order={order} exceeded {args.timeout_seconds:.1f}s")
                rows.append({"order":order,"model":model,"receiver":sid,"band_hz":[250,4000],"level_db":vals[model,sid],"seconds":elapsed,"rir_samples_first_source":len(h),"rir_samples_all_sources":[len(np.asarray(q)) for q in room.rir[0]],"synthesized_samples":len(total),"wall_count":len(room.walls),"path_diagnostics":path_diagnostics(room,scenes[model,sid])})
        for model in ("A","B","C"):
            anchor=vals[model,"P02"]
            for r in rows:
                if r["order"]==order and r["model"]==model: r["relative_to_P02_db"]=r["level_db"]-anchor
        # Checkpoint each completed order so a later interruption does not
        # erase the completed order's result.
        args.output_dir.mkdir(parents=True,exist_ok=True)
        checkpoint={"pyroomacoustics":pra.__version__,"orders_completed":[order],"geometry":"current acoustic polygon","rows":[r for r in rows if r["order"]==order]}
        (args.output_dir/f"metrics_order_{order}.json").write_text(json.dumps(checkpoint,ensure_ascii=False,indent=2),encoding="utf-8")
    previous_rows=[]
    if args.previous_results.exists() and args.previous_results.resolve()!= (args.output_dir/"metrics_three_models.json").resolve():
        previous_rows=load(args.previous_results).get("rows",[])
    all_for_delta=previous_rows+rows
    for r in rows:
        prior=[q for q in all_for_delta if q["model"]==r["model"] and q["receiver"]==r["receiver"] and q["order"]==r["order"]-1]
        if prior:
            r["relative_level_change_vs_previous_order_db"]=r["relative_to_P02_db"]-prior[-1]["relative_to_P02_db"]
            r["band_level_change_vs_previous_order_db"]=r["level_db"]-prior[-1]["level_db"]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # Required control: same 2.67 m geometry through general Room without protrusions.
    control={}
    control_diffs={}
    for order in args.orders:
        for sid in IDS:
            sc=scenes["B",sid]; t=time.perf_counter()
            rr=extruded_polygon_room(sc,order)
            control[str(order)+":"+sid]={"rir_samples_all_sources":[len(np.asarray(q)) for q in rr.rir[0]],"paths":path_diagnostics(rr,sc),"seconds":time.perf_counter()-t}
            shoe=make_room(sc,"B",order,x)
            diffs=[]
            for rh,sh in zip(rr.rir[0],shoe.rir[0]):
                n=max(len(rh),len(sh)); a=np.pad(np.asarray(rh,float),(0,n-len(rh))); b=np.pad(np.asarray(sh,float),(0,n-len(sh)))
                diffs.append({"max_abs":float(np.max(np.abs(a-b))),"rms":float(np.sqrt(np.mean((a-b)**2))),"relative_rms":float(np.sqrt(np.mean((a-b)**2))/(np.sqrt(np.mean(b*b))+1e-30))})
            control_diffs[str(order)+":"+sid]=diffs
    poly=air_floor_polygon(7.5,10.25)
    polygon_area=float(0.5*abs(np.dot(poly[0],np.roll(poly[1],-1))-np.dot(poly[1],np.roll(poly[0],-1))))
    area_expected=7.5*10.25-(.42+.70)*(.725+1.05)
    area_pass=abs(polygon_area-area_expected)<1e-9
    fig,ax=plt.subplots(figsize=(8,6)); closed=np.column_stack([poly,poly[:,0]])
    ax.plot(closed[0],closed[1],"-o",ms=3); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Acoustic air polygon | area={polygon_area:.3f} m²"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(args.output_dir/"polygon_acoustic_floorplan.png",dpi=160); plt.close(fig)
    result={"pyroomacoustics":pra.__version__,"analysis":"09-compatible 250-4000 Hz, Butterworth 4 sosfiltfilt, three 8 s windows","absorption":.2,"orders":args.orders,"models":{"A":"2.63 m ShoeBox","B":"2.67 m ShoeBox","C":"2.67 m extruded air polygon with four wall-connected recess boundaries"},"control_B_height_general_Room":control,"control_B_vs_ShoeBox_rir_differences":control_diffs,"polygon_area_check":{"computed_m2":polygon_area,"expected_m2":area_expected,"absolute_difference_m2":abs(polygon_area-area_expected),"status":"PASS" if area_pass else "FAIL"},"custom_geometry_diagnostics":{"method":"single_air_floor_polygon_then_extrude","polygon_xy_m":air_floor_polygon(7.5,10.25).T.tolist(),"coincident_outer_faces_retained":False,"right_front_exposed_x_m":6.80,"right_back_exposed_x_m":6.80,"visualization":"polygon_acoustic_floorplan.png"},"rows":rows,"limitations":["C uses general 3-D Room, not ShoeBox fast ISM; compare order and runtime explicitly.","The four measured blocks are represented as air-domain recess boundaries by one closed plan polygon and extrusion.","High-order calculation remains on hold until the order-0/1/2 control, numerical RIR comparison and area checks pass."],"source_config":str(CONFIG),"settings":str(SETTINGS)}
    (args.output_dir/"metrics_three_models.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    summary={
        "polygon_xy_m": result["custom_geometry_diagnostics"]["polygon_xy_m"],
        "area_check": result["polygon_area_check"],
        "general_room_vs_shoebox_rir_max_abs_by_source": {
            k: max(d["max_abs"] for d in v) for k,v in control_diffs.items()
        },
        "relative_levels_db": {
            m: {r["receiver"]: r["relative_to_P02_db"] for r in rows
                if r["model"]==m and r["receiver"] in ("P01","P03")}
            for m in ("A","B","C")
        },
        "relative_level_change_vs_previous_order_db": {
            f"order_{o}": {m: {r["receiver"]: r.get("relative_level_change_vs_previous_order_db")
                                for r in rows if r["order"]==o and r["model"]==m and r["receiver"] in ("P01","P03")}
                         for m in ("A","B","C")}
            for o in args.orders
        },
        "p02_band_levels_db": {
            f"order_{o}": {m: next(r["level_db"] for r in rows if r["order"]==o and r["model"]==m and r["receiver"]=="P02") for m in ("A","B","C")}
            for o in args.orders
        },
        "p02_band_level_change_vs_previous_order_db": {
            f"order_{o}": {m: next((r.get("band_level_change_vs_previous_order_db") for r in rows if r["order"]==o and r["model"]==m and r["receiver"]=="P02"), None) for m in ("A","B","C")}
            for o in args.orders
        },
        "calculation_seconds": {
            f"order_{o}": {m: {r["receiver"]: r["seconds"] for r in rows
                                if r["order"]==o and r["model"]==m}
                         for m in ("A","B","C")}
            for o in args.orders
        }
    }
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()

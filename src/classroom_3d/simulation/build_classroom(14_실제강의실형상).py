"""(14) 실제 강의실 형상만 구축하고 평면도·3D 배치도·좌표 검사를 생성한다."""
import argparse, json, math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/classroom/classroom_config(14_실제강의실형상).json"
OUT = ROOT / "results/classroom_3d"

def inside(p, dims): return all(0 <= p[i] <= dims[i] for i in range(3))
def in_rect(p, r): return r["x_range_m"][0] <= p[0] <= r["x_range_m"][1] and r["y_range_m"][0] <= p[1] <= r["y_range_m"][1]
def ranges_overlap(a, b): return a[0] < b[1] and b[0] < a[1]
def check(scene):
    dims = [scene["room"][k] for k in ("width_m","length_m","height_m")]
    rows=[]
    for kind, items in (("source",scene["sources"]),("receiver",scene["receivers"])):
        for ident, obj in items.items():
            p=obj["position_m"]; obstacle=next((r["id"] for r in scene["wall_connected_protrusions"] if in_rect(p,r)), None)
            rows.append({"id":ident,"kind":kind,"position_m":p,"inside_room":inside(p,dims),"inside_wall_connected_protrusion":obstacle or False,"status":"PASS" if inside(p,dims) and not obstacle else "FAIL"})
    for top in scene["tops_and_sills"]:
        overlaps = [r["id"] for r in scene["wall_connected_protrusions"]
                    if top["y_range_m"] is not None
                    and ranges_overlap(top["x_range_m"], r["x_range_m"])
                    and ranges_overlap(top["y_range_m"], r["y_range_m"])]
        rows.append({"id": top["id"], "kind": "top_or_sill", "temporary_extent": top["status"] == "temporary_extent_assumption",
                     "y_range_m": top["y_range_m"], "overlaps_wall_connected_protrusion": overlaps,
                     "status": "PASS" if not overlaps else "FAIL"})
    return rows
def box(ax, x0,x1,y0,y1,z0,z1,color,alpha,label):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v=[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    f=[[v[i] for i in q] for q in ((0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))]
    ax.add_collection3d(Poly3DCollection(f,facecolors=color,edgecolors=color,alpha=alpha,label=label))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--show",action="store_true"); a=ap.parse_args()
    scene=json.loads(CONFIG.read_text(encoding="utf-8")); checks=check(scene); assert all(x["status"]=="PASS" for x in checks)
    import matplotlib
    if not a.show: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for candidate in ("Malgun Gothic", "Noto Sans CJK KR", "DejaVu Sans"):
        if any(candidate.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = candidate; break
    plt.rcParams["axes.unicode_minus"] = False
    from matplotlib.patches import Rectangle
    out=OUT/("classroom(14_실제강의실형상)_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")); out.mkdir(parents=True)
    w,l,h=scene["room"]["width_m"],scene["room"]["length_m"],scene["room"]["height_m"]
    fig,ax=plt.subplots(figsize=(9,11)); ax.add_patch(Rectangle((0,0),w,l,fc="#f7fafc",ec="#25364d",lw=2))
    for r in scene["wall_connected_protrusions"]: ax.add_patch(Rectangle((r["x_range_m"][0],r["y_range_m"][0]),r["x_range_m"][1]-r["x_range_m"][0],r["y_range_m"][1]-r["y_range_m"][0],fc="#8b9aad",ec="#344054",alpha=.75))
    wood=scene["tops_and_sills"][0]; sill=scene["tops_and_sills"][1]
    ax.add_patch(Rectangle((wood["x_range_m"][0],wood["y_range_m"][0]),.5,wood["y_range_m"][1]-wood["y_range_m"][0],fc="#c98b45",ec="#754c24",alpha=.85)); ax.add_patch(Rectangle((sill["x_range_m"][0],sill["y_range_m"][0]),.2,sill["y_range_m"][1]-sill["y_range_m"][0],fc="#e3bd7d",ec="#8b6b3f",alpha=.8))
    for ident,o in scene["sources"].items(): x,y,z=o["position_m"]; ax.scatter(x,y,marker="*",s=260,c="#d06a20",edgecolors="k",zorder=5); ax.annotate(f"{ident}\n({x:.2f},{y:.2f},{z:.2f})",(x,y),xytext=(7,7),textcoords="offset points")
    for ident,o in scene["receivers"].items(): x,y,z=o["position_m"]; ax.scatter(x,y,s=95,facecolors="white",edgecolors="#087f8c",lw=2,zorder=5); ax.annotate(f"{ident}\n({x:.2f},{y:.2f},{z:.3f})",(x,y),xytext=(7,-22),textcoords="offset points",color="#087f8c")
    ax.text(w/2,-.45,"칠판/앞벽 y=0",ha="center"); ax.text(w/2,l+.35,"뒷벽 y=10.25",ha="center"); ax.text(7.54,l/2,"오른쪽 창문 유리면 x=7.50",rotation=90,va="center",color="#247a9b")
    ax.set(xlim=(-.5,w+.7),ylim=(-.8,l+.7),xlabel="x (m)",ylabel="y (m)",title=f"실제 강의실 평면 배치도 | {w:.2f} × {l:.2f} m"); ax.set_aspect("equal"); ax.grid(alpha=.2); fig.text(.5,.02,"회색=벽 연결 돌출부(바닥~천장) / 갈색=상판·창턱 임시 범위 / 창문 정확한 유리 구간은 미확인\n책상·의자 생략. 배치도이며 음압 분포·음향 계산·실측 검증 결과가 아님.",ha="center",fontsize=9); fig.savefig(out/"classroom(02_평면배치도).png",dpi=170,bbox_inches="tight"); plt.close(fig)
    fig=plt.figure(figsize=(11,8)); ax=fig.add_subplot(111,projection="3d"); box(ax,0,w,0,l,0,.02,"#aab6c4",.15,"floor")
    for r in scene["wall_connected_protrusions"]: box(ax,*r["x_range_m"],*r["y_range_m"],*r["z_range_m"],"#718096",.65,"wall-connected protrusion")
    box(ax,*wood["x_range_m"],*wood["y_range_m"],.70,wood["z_top_m"],"#bd7b32",.85,"wood top (temporary extent)"); box(ax,*sill["x_range_m"],*sill["y_range_m"],.70,sill["z_top_m"],"#d7a85e",.75,"narrow sill (temporary extent)")
    ax.plot([7.5,7.5],[0,l],[1.37,1.37],color="#2b90b9",alpha=.35) # visual cue only
    for ident,o in scene["sources"].items(): x,y,z=o["position_m"]; ax.scatter(x,y,z,marker="*",s=190,c="#d06a20"); ax.plot([x,x],[y,y],[0,z],"--",c="#d06a20",lw=1); ax.text(x,y,z+.06,ident)
    for ident,o in scene["receivers"].items(): x,y,z=o["position_m"]; ax.scatter(x,y,z,s=60,facecolors="white",edgecolors="#087f8c"); ax.text(x,y,z+.07,ident,color="#087f8c")
    ax.set(xlim=(0,w),ylim=(0,l),zlim=(0,h),xlabel="x (m)",ylabel="y (m)",zlabel="z (m)",title=f"실제 강의실 3D 배치도 | 천장 높이 {h:.2f} m"); ax.set_box_aspect((w,l,h)); ax.view_init(25,-58); fig.text(.5,.02,"회색 돌출부는 바닥부터 천장까지 연결. 상판·창턱은 높이 차 확인용 임시 표현이며 하부를 고체로 확정하지 않음.\n창문 유리·블라인드·천장 흡음률 미확정; 책상·의자 생략. 음향 결과 아님.",ha="center",fontsize=9); fig.savefig(out/"classroom(03_3D배치도).png",dpi=170,bbox_inches="tight"); plt.close(fig)
    (out/"classroom(01_좌표검사와설정요약).json").write_text(json.dumps({"config":str(CONFIG.relative_to(ROOT)),"checks":checks,"changes":["height 2.63m -> 2.67m","four wall-connected protrusions added","tops/sills represented with explicit uncertainty","window glass segments not invented"],"not_in_scope":["acoustic calculation","absorption correction","new recording","measurement validation"]},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"PASS: {out}")
if __name__=="__main__": main()

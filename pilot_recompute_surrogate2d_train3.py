from pathlib import Path
import csv,json,hashlib,platform,sys,time,importlib.metadata as md
import numpy as np
import pyroomacoustics as pra

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'/'surrogate_2d'/'dataset'; GEN=ROOT/'src'/'surrogate_2d'/'dataset'/'heatmap_obstacle.py'; OUT=ROOT/'results'/'surrogate_2d_pilot_recompute_train3_20261012'
FS=16000; ROOM_X=8.; ROOM_Y=10.; ORDER=8; GRID=20; WALL_ALPHA=.2; OBSTACLE_ALPHA=.1
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def wall(p1,p2,a,name): return pra.wall_factory(np.array([p1,p2],dtype=np.float32).T,np.array([a],dtype=np.float32),np.array([0.],dtype=np.float32),name)
def main():
 t0=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=False); manifest=json.loads((DATA/'splits/split_manifest.json').read_text(encoding='utf-8')); selected=sorted(manifest['splits']['train'])[:3]
 with (DATA/'metadata.csv').open(encoding='utf-8-sig',newline='') as f: meta={r['sample_id']:r for r in csv.DictReader(f)}
 x=np.linspace(.5,ROOM_X-.5,GRID); y=np.linspace(.5,ROOM_Y-.5,GRID); X,Y=np.meshgrid(x,y); scene_reports=[]
 for sid in selected:
  r=meta[sid]; sx,sy,ox,oy,ow,ol=[float(r[k]) for k in ('speaker_x','speaker_y','obstacle_x','obstacle_y','obstacle_w','obstacle_l')]; invalid=(X>=ox)&(X<=ox+ow)&(Y>=oy)&(Y<=oy+ol); valid=~invalid; pos=[]; ij=[]
  for i in range(GRID):
   for j in range(GRID):
    if valid[i,j]: pos.append([X[i,j],Y[i,j]]); ij.append([i,j])
  walls=[]; outer=[[0,0],[ROOM_X,0],[ROOM_X,ROOM_Y],[0,ROOM_Y]]
  for i in range(4): walls.append(wall(outer[i],outer[(i+1)%4],WALL_ALPHA,f'outer_{i}'))
  ob=[[ox,oy],[ox,oy+ol],[ox+ow,oy+ol],[ox+ow,oy]]
  for i in range(4): walls.append(wall(ob[i],ob[(i+1)%4],OBSTACLE_ALPHA,f'obstacle_{i}'))
  room=pra.Room(walls,fs=FS,max_order=ORDER); room.add_source(np.array([sx,sy],dtype=np.float32)); room.add_microphone_array(pra.MicrophoneArray(np.asarray(pos,dtype=np.float32).T,FS)); room.compute_rir()
  raw_rirs=[np.asarray(room.rir[k][0],dtype=np.float64) for k in range(len(pos))]; rir_len=np.array([len(z) for z in raw_rirs],dtype=np.int64); rir=np.zeros((len(raw_rirs),int(rir_len.max())),dtype=np.float64); [rir.__setitem__((k,slice(0,len(z))),z) for k,z in enumerate(raw_rirs)]; raw_energy=np.sum(rir**2,axis=1); floor=raw_energy<1e-12; floored=np.maximum(raw_energy,1e-12); db=10*np.log10(floored); max_db=float(np.max(db)); norm=db-max_db; regenerated=np.full((GRID,GRID),np.nan,dtype=np.float32); regenerated[valid]=norm.astype(np.float32)
  orig=np.load(DATA/'npy'/f'{sid}.npy',allow_pickle=False); mask_match=bool(np.array_equal(np.isnan(orig),invalid)); e=regenerated[valid]-orig[valid]; comp={'sample_id':sid,'grid_shape_match':bool(orig.shape==(20,20)),'axis_order':'y,x','valid_mask_match':mask_match,'valid_receiver_count':len(pos),'invalid_obstacle_cells':int(invalid.sum()),'empty_rir_count':int(np.sum(np.array([len(z)==0 for z in rir]))),'zero_energy_count':int(np.sum(raw_energy==0)),'zero_or_empty_locations':[ij[k] for k,z in enumerate(raw_energy) if z==0],'floor_count':int(floor.sum()),'max_pre_normalization_db':max_db,'mae_db':float(np.mean(abs(e))),'rmse_db':float(np.sqrt(np.mean(e**2))),'max_abs_db':float(np.max(abs(e)))}; scene_reports.append(comp)
  np.savez_compressed(OUT/f'{sid}_raw.npz',sample_id=sid,rir=rir,rir_lengths=rir_len,receiver_positions=np.asarray(pos),receiver_indices=np.asarray(ij,dtype=np.int64),valid_mask=valid,obstacle_mask=invalid,raw_energy_before_floor=raw_energy,floor_applied=floor,pre_normalization_db=db,scene_max_db=np.array(max_db),normalized_map=regenerated,original_map=orig)
 (OUT/'metrics.json').write_text(json.dumps({'selected_train_ids':selected,'selection_basis':'sorted existing manifest train IDs; validation/test unused','settings':{'room_m':[ROOM_X,ROOM_Y],'grid_shape':[20,20],'grid_axis_order':'y,x','grid_coordinates':'0.5..7.5 and 0.5..9.5','fs_hz':FS,'max_order':ORDER,'wall_absorption':WALL_ALPHA,'obstacle_absorption':OBSTACLE_ALPHA,'energy':'sum(rir**2)','floor':1e-12,'normalization':'10log10(max(floor(energy),1e-12)) then subtract scene maximum'},'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pyroomacoustics':md.version('pyroomacoustics'),'generator_sha256':sha(GEN),'manifest_sha256':sha(DATA/'splits/split_manifest.json')},'scenes':scene_reports,'elapsed_s':time.perf_counter()-t0,'model_shape_check':{'existing_training_model_input_per_receiver':'(400, 10)','existing_training_model_output_per_scene':'(400)','evidence':'train_baseline(04_기초모델학습).py lines 59-61, 75-80, 111; AcousticDataset docstring'},'distance_baseline':{'formula':'-20*log10(distance/dmin)','status':'existing distance reference only; not asserted physically equivalent to 2D simulator'}},ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'README.md').write_text('# 2D surrogate train-3 recomputation pilot\n\n기존 manifest의 정렬된 train 첫 3개만 metadata 형상으로 재계산했습니다. 새 장면 생성, validation/test 사용, 주파수 필터, AI 학습은 하지 않았습니다. 원본 NPY는 변경하지 않았습니다. 각 NPZ에는 raw RIR, 수음점 좌표·인덱스, 마스크, floor 전 에너지, floor 여부, 정규화 전 dB, 장면 최대값, 정규화 지도, 원본 지도가 포함됩니다. 비교 오차는 유효 격자에서만 계산했습니다.\n\n실행: `.venv\\Scripts\\python.exe pilot_recompute_surrogate2d_train3.py`\n',encoding='utf-8')
 print(json.dumps({'output':str(OUT),'selected':selected,'elapsed_s':time.perf_counter()-t0,'reports':scene_reports},ensure_ascii=False))
if __name__=='__main__': main()

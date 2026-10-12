from pathlib import Path
import json, math, csv, hashlib
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyroomacoustics as pra

ROOT=Path(__file__).resolve().parent; FS=48000
BANDS=[(250,500),(500,1000),(1000,2000),(2000,4000)]
GAINS={'250-500 Hz':1.1969358634407834,'500-1000 Hz':0.9009453609884894,'1000-2000 Hz':0.6476255960846105,'2000-4000 Hz':0.3910747361628081}
MODELS={'B0':(1.0,1),'B1':(None,1),'B2':(None,-1)}
TRIAL=ROOT/'recordings'/'measurements'/'20261009_P02_trial2'
FILES={'left_only':TRIAL/'2차실험 왼쪽 스피커.wav','right_only':TRIAL/'2차실험 오른쪽.wav','both':TRIAL/'2차실험 양쪽 스피커.wav','both_end':TRIAL/'2차실험 종료 양쪽스피커.wav'}
REF=next((p for p in (ROOT/'recordings'/'reference').glob('*.wav') if 'test_signal' in p.name),None)
CFG=next((p for p in (ROOT/'configs'/'classroom').glob('classroom_config(07_*.json')),None)
ST=next((p for p in (ROOT/'configs'/'simulation').glob('simulation_settings(08_*.json')),None)
CACHE=ROOT/'results'/'20261009_P02_trial2_delay_polarity_search_20261009_172020'/'P02_B_order32_rir_cache.npz'
OUT=ROOT/'results'/'fixed_B_models_P01_P02_P03_20261012'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 import wave
 with wave.open(str(p),'rb') as w:
  a=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').reshape(-1,w.getnchannels())
  if w.getframerate()!=FS or w.getsampwidth()!=2: raise ValueError(p)
  return a[:,0].astype(float)/32768
def energy(x,a,b): return float(np.mean(x[a:b]**2))
def level(e): return 10*math.log10(e)
def wav(rir,ref,gain=1): return fftconvolve(ref,rir*gain,mode='full')
def make_rir(pos,src,st):
 room=pra.ShoeBox([7.5,10.25,2.67],fs=FS,materials=pra.make_materials(west=.2,east=.2,south=.2,north=.2,floor=.2,ceiling=.2),max_order=32,ray_tracing=False,air_absorption=False,use_rand_ism=False,sigma2_awgn=None); room.set_sound_speed(st['speed_of_sound_m_s'])
 for sid in ('S_L','S_R'): room.add_source(src[sid]['position_m'])
 room.add_microphone_array(np.asarray(pos,float)[:,None]); room.compute_rir(); return {sid:np.asarray(room.rir[0][i],float)*src[sid]['gain'] for i,sid in enumerate(('S_L','S_R'))}
def main():
 OUT.mkdir(parents=True,exist_ok=True); cfg=json.loads(CFG.read_text(encoding='utf-8-sig')); st=json.loads(ST.read_text(encoding='utf-8-sig')); ref=read(REF)
 src={}
 for s in cfg['sources']:
  x=s['side_wall_distance_m'] if s['side_wall']=='left' else 7.5-s['side_wall_distance_m']; sid=s['id']; src[sid]={'position_m':[x,s['front_wall_distance_m'],s['height_m']],'gain':st['sources'][sid]['relative_gain'],'delay_samples':st['sources'][sid]['extra_delay_samples']}
 if set(src)!= {'S_L','S_R'} or any(x['delay_samples']!=0 for x in src.values()): raise ValueError(src)
 pos={r['id']:r['position_m'] for r in cfg['receivers']}; rirs={}
 if CACHE.exists():
  z=np.load(CACHE); rirs['P02']={'S_L':z['rir_L'],'S_R':z['rir_R']}; cache_used=['P02']
 else: cache_used=[]
 for pid in ('P01','P02','P03'):
  if pid not in rirs: rirs[pid]=make_rir(pos[pid],src,st)
  if pid not in cache_used: np.savez(OUT/f'{pid}_B_order32_rir_cache.npz',**{f'rir_{s}':rirs[pid][s] for s in ('S_L','S_R')})
 raw={k:read(v) for k,v in FILES.items()}; wins=[(6*FS,14*FS),(21*FS,29*FS),(36*FS,44*FS)]; refwins=wins
 data={}; rows=[]; checks={}
 for lo,hi in BANDS:
  key=f'{lo}-{hi} Hz'; sos=butter(4,[lo,hi],btype='bandpass',fs=FS,output='sos'); data[key]={};
  measured={k:sosfiltfilt(sos,v) for k,v in raw.items()}; data[key]['models']={}
  for model,(g,pol) in MODELS.items():
   g=GAINS[key] if model!='B0' else 1.0; data[key]['models'][model]={}
   for pid in ('P01','P02','P03'):
    L=wav(rirs[pid]['S_L'],ref); R=wav(rirs[pid]['S_R'],ref,gain=g)*pol; lf=sosfiltfilt(sos,L); rf=sosfiltfilt(sos,R); both=np.zeros(max(len(lf),len(rf))); both[:len(lf)]+=lf; both[:len(rf)]+=rf
    esL=np.mean([energy(lf,*w) for w in refwins]); esR=np.mean([energy(rf,*w) for w in refwins]); esB=np.mean([energy(both,*w) for w in refwins]); I=level(esB/(esL+esR))
    data[key]['models'][model][pid]={'E_left':esL,'E_right':esR,'E_both':esB,'level_left_db':level(esL),'level_right_db':level(esR),'level_both_db':level(esB),'left_minus_right_db':level(esL)-level(esR),'I_db':I}
   p02=data[key]['models'][model]['P02']['level_both_db']
   for pid in ('P01','P02','P03'):
    d=data[key]['models'][model][pid]; d['relative_to_P02_both_db']=d['level_both_db']-p02; d['relative_level_check_db']=d['level_both_db']-p02-d['relative_to_P02_both_db']
  checks[key]={'B1_B2_single_energy_max_abs_db':max(abs(10*math.log10(data[key]['models']['B1'][p][q]/data[key]['models']['B2'][p][q])) for p in ('P01','P02','P03') for q in ('E_left','E_right')),'polarity_identity_max_abs_relative':max(abs(data[key]['models']['B1'][p]['E_both']+data[key]['models']['B2'][p]['E_both']-2*(data[key]['models']['B1'][p]['E_left']+data[key]['models']['B1'][p]['E_right']))/(data[key]['models']['B1'][p]['E_both']+data[key]['models']['B2'][p]['E_both']) for p in ('P01','P02','P03'))}
  for m in MODELS:
   for p in ('P01','P02','P03'):
    d=data[key]['models'][m][p]; rows.append([key,m,p,d['level_left_db'],d['level_right_db'],d['level_both_db'],d['left_minus_right_db'],d['I_db'],d['relative_to_P02_both_db'],d['relative_level_check_db']])
  # separate measured reference summaries, no measured-based fitting
  data[key]['measured_energy_levels_db']={k:level(np.mean([energy(v,*w) for w in wins])) for k,v in measured.items()}
 with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f: csv.writer(f).writerows([['band','model','point','left_level_db','right_level_db','both_level_db','left_minus_right_db','I_db','relative_to_P02_both_db','relative_level_check_db'],*rows])
 # plots
 x=np.arange(3); labels=['P01','P02','P03']
 for field,name in [('relative_to_P02_both_db','relative_levels'),('I_db','I')]:
  fig,axs=plt.subplots(2,2,figsize=(12,8));
  for ax,(key,d) in zip(axs.flat,data.items()):
   for m in MODELS: ax.plot(x,[d['models'][m][p][field] for p in labels],'o-',label=m)
   ax.set_title(key); ax.set_xticks(x,labels); ax.set_ylabel('dB'); ax.grid(alpha=.25)
  axs.flat[0].legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/f'{name}_by_band.png',dpi=180); plt.close(fig)
 result={'fixed_conditions':{'models':'B0 gain 1/polarity +1; B1 fixed right gains; B2 B1 with right polarity -1; all delays 0','right_gain_by_band':GAINS,'left_gain':1,'excluded_delay_candidates':[-1],'geometry_m':[7.5,10.25,2.67],'order':32,'absorption':.2,'speed_of_sound_m_s':st['speed_of_sound_m_s'],'receiver_positions':pos,'source_mapping':src,'bands_hz':BANDS,'filter':'Butterworth 4th order sosfiltfilt','windows_s':[[6,14],[21,29],[36,44]],'no_file_normalization':True,'coherent_sum_before_energy':True},'input_identification':{'reference':str(REF),'reference_sha256':sha(REF),'files':{k:{'path':str(v),'sha256':sha(v)} for k,v in FILES.items()},'cache':str(CACHE),'cache_used_for':['P02']+['P01','P03'] if CACHE.exists() else ['P01','P02','P03']},'data':data,'validation':checks,'prior_P02_B2_I_db_expected':[1.27856,.98514,.95233,.19917],'prior_P02_B2_I_db_actual':[data[f'{lo}-{hi} Hz']['models']['B2']['P02']['I_db'] for lo,hi in BANDS],'notes':['Fixed prediction only; no new-measurement optimization.','Previous P01/P03 recordings were not used for model selection or correction.','This is not confirmation of actual wiring polarity or an independent validation.','Each band is independent; no broadband result is formed by combining bands.']}
 (OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'README.md').write_text('# Fixed B0/B1/B2 predictions\n\nP02에서 고정한 대역별 오른쪽 진폭 보정을 동일하게 P01/P02/P03에 적용한 예측입니다. 새 실측 최적화와 파일별 음량 정규화는 하지 않았고, 6–14, 21–29, 36–44초를 사용했습니다. B1/B2 단독 에너지와 극성 반전 항등식을 검사했습니다. 이 결과는 향후 측정을 위한 고정 예측이며 실제 배선 극성 확인이나 독립 검증 결과가 아닙니다. 이전 P01/P03 녹음은 모델 선택·보정에 사용하지 않았습니다.\n',encoding='utf-8')
 print(OUT)
if __name__=='__main__': main()

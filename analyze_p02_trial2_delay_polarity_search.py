from pathlib import Path
import json, math, csv, hashlib, importlib.util
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; FS=48000; MAXD=240
BASE=ROOT/'analyze_p02_three_speaker_conditions_order32.py'
PREV=ROOT/'results'/'20261009_P02_trial2_frequency_correction_B_order32_20261009_170209'/'metrics.json'
INPUT=ROOT/'recordings'/'measurements'/'20261009_P02_trial2'
FILES={'start_both':INPUT/'2차실험 양쪽 스피커.wav','end_both':INPUT/'2차실험 종료 양쪽스피커.wav'}
BANDS=[(250,500),(500,1000),(1000,2000),(2000,4000)]
spec=importlib.util.spec_from_file_location('base',BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

def shift_zero(x,d):
 out=np.zeros_like(x)
 if d>0: out[d:]=x[:-d]
 elif d<0: out[:d]=x[-d:]
 else: out[:]=x
 return out
def energy(x,a,b): return float(np.mean(x[a:b]**2))
def compute_I(L,R,a,b):
 el=energy(L,a,b); er=energy(R,a,b); eb=energy(L+R,a,b); return 10*math.log10(eb/(el+er))
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

def main():
 old=json.loads(PREV.read_text(encoding='utf-8')); stamp=__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S'); out=ROOT/'results'/f'20261009_P02_trial2_delay_polarity_search_{stamp}'; out.mkdir()
 # Use the already validated trial2 alignment windows and fixed P02 correction multipliers.
 wins={n:[tuple(x) for x in old['input_files'][n]['analysis_windows_s']] for n in ('start_both','end_both','left_only','right_only')}
 wins={n:[(round(a*FS),round(b*FS)) for a,b in v] for n,v in wins.items()}
 ref_wins=[(6*FS,14*FS),(21*FS,29*FS),(36*FS,44*FS)]
 raw={n:base.read_wav(p) for n,p in FILES.items()}
 cfg=json.loads((ROOT/'configs'/'classroom'/'classroom_config(07_강의실측정조건).json').read_text(encoding='utf-8-sig')); st=json.loads((ROOT/'configs'/'simulation'/'simulation_settings(08_음향가정).json').read_text(encoding='utf-8-sig')); w,l=cfg['room']['width_m'],cfg['room']['length_m']; src={}
 for s in cfg['sources']:
  x=s['side_wall_distance_m'] if s['side_wall']=='left' else w-s['side_wall_distance_m']; src[s['id']]={'position_m':[x,s['front_wall_distance_m'],s['height_m']],'gain':st['sources'][s['id']]['relative_gain'],'delay_samples':st['sources'][s['id']]['extra_delay_samples']}
 rec=next(r for r in cfg['receivers'] if r['id']=='P02')['position_m']
 room=base.pra.ShoeBox([w,l,2.67],fs=FS,materials=base.pra.make_materials(west=.2,east=.2,south=.2,north=.2,floor=.2,ceiling=.2),max_order=32,ray_tracing=False,air_absorption=False,use_rand_ism=False,sigma2_awgn=None); room.set_sound_speed(st['speed_of_sound_m_s']); room.add_source(src['S_L']['position_m']); room.add_source(src['S_R']['position_m']); room.add_microphone_array(np.asarray(rec,float)[:,None]); room.compute_rir(); ref=base.read_wav(base.REFERENCE,allow_identical_stereo=True)
 # Cache matching P02 RIR and the unfiltered coherent source components for reproducibility/reuse.
 np.savez(out/'P02_B_order32_rir_cache.npz', rir_L=np.asarray(room.rir[0][0]), rir_R=np.asarray(room.rir[0][1]), source_L=fftconvolve(ref,np.asarray(room.rir[0][0])*src['S_L']['gain'],mode='full'), source_R=fftconvolve(ref,np.asarray(room.rir[0][1])*src['S_R']['gain'],mode='full'))
 corr={f'{lo}-{hi} Hz':old['measured'][f'{lo}-{hi} Hz']['correction']['right_amplitude_multiplier'] for lo,hi in BANDS}
 signals={}
 for lo,hi in BANDS:
  key=f'{lo}-{hi} Hz'; sos=butter(4,[lo,hi],btype='bandpass',fs=FS,output='sos'); signals[key]={}
  for n,y in raw.items(): signals[key][n]=(sosfiltfilt(sos,y),)
  Lm=sosfiltfilt(sos,fftconvolve(ref,np.asarray(room.rir[0][0])*src['S_L']['gain'],mode='full')); Rm=sosfiltfilt(sos,fftconvolve(ref,np.asarray(room.rir[0][1])*src['S_R']['gain']*corr[key],mode='full')); signals[key]['model']=(Lm,Rm)
 left_raw=base.read_wav(INPUT/'2차실험 왼쪽 스피커.wav'); right_raw=base.read_wav(INPUT/'2차실험 오른쪽.wav'); both_raw={n:base.read_wav(FILES[n]) for n in ('start_both','end_both')}
 def target(n,key):
  lo,hi=map(float,key.replace(' Hz','').split('-')); sos=butter(4,[lo,hi],btype='bandpass',fs=FS,output='sos'); left=sosfiltfilt(sos,left_raw); right=sosfiltfilt(sos,right_raw); both=sosfiltfilt(sos,both_raw[n]); vals=[]
  for i in range(3):
   aL,bL=wins['left_only'][i]; aR,bR=wins['right_only'][i]; aB,bB=wins[n][i]; vals.append(10*math.log10(energy(both,aB,bB)/(energy(left,aL,bL)+energy(right,aR,bR))))
  return vals
 targets={key:{n:target(n,key) for n in ('start_both','end_both')} for key in corr}; target_start=np.array([np.mean(targets[k]['start_both']) for k in corr]); target_end=np.array([np.mean(targets[k]['end_both']) for k in corr])
 rows=[]; bypol={}
 for pol in (1,-1):
  vals=[]
  for d in range(-MAXD,MAXD+1):
   modelI=[]; startI=[]; endI=[]
   for key in corr:
    L,R=signals[key]['model']; R2=shift_zero(R,d)*pol; a,b=ref_wins[1]; modelI.append(compute_I(L,R2,a,b))
   e=rmse(modelI,target_start); vals.append((d,e,modelI)); rows.append([d,pol,e,*modelI])
  bypol[pol]=vals
 best=min(rows,key=lambda r:r[2]); bd,bp,be=best[:3]; bestI=best[3:]
 # Direct recheck uses the exact selected delayed summed waveform and all repeats.
 def direct(key,d,pol):
  L,R=signals[key]['model']; R2=shift_zero(R,d)*pol; return [compute_I(L,R2,a,b) for a,b in ref_wins]
 direct_best=[]
 for key in corr:
  L,R=signals[key]['model']; R2=shift_zero(R,bd)*bp; direct_best.append(compute_I(L,R2,*ref_wins[1]))
 direct_end=direct_best[:]
 near=sorted(rows,key=lambda r:r[2])[:11]
 with (out/'candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
  wr=csv.writer(f); wr.writerow(['delay_samples','delay_ms','polarity','rmse_start_db','I_250_500','I_500_1000','I_1000_2000','I_2000_4000']); [wr.writerow([r[0],r[0]/FS*1000,r[1],*r[2:]]) for r in rows]
 fig,ax=plt.subplots(figsize=(11,5)); ds=np.arange(-MAXD,MAXD+1); [ax.plot(ds,[v[1] for v in bypol[p]],label=f'polarity {p:+d}') for p in (1,-1)]; ax.axvline(0,color='k',lw=.7); ax.axvline(bd,color='tab:red',ls='--'); ax.set(xlabel='relative delay (samples)',ylabel='RMSE of 4-band I (dB)',title='Common delay/polarity search'); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'rmse_by_delay_polarity.png',dpi=180); plt.close(fig)
 baseI=[]; bestvals=[]
 for key in corr:
  L,R=signals[key]['model']; baseI.append(compute_I(L,R,*ref_wins[1])); bestvals.append(direct_best[len(bestvals)])
 x=np.arange(4); fig,ax=plt.subplots(figsize=(11,5)); ax.plot(x,target_start,'o-',label='Measured start'); ax.plot(x,baseI,'o-',label='Model baseline'); ax.plot(x,bestvals,'o-',label='Model best'); ax.set_xticks(x,list(corr)); ax.set_ylabel('I (dB)'); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'baseline_best_band_I.png',dpi=180); plt.close(fig)
 result={'analysis':'P02 trial2 common relative delay/polarity exploratory search','basis_result':str(PREV),'search_delay_samples':[-MAXD,MAXD],'search_delay_ms':[-5,5],'delay_step_samples':1,'polarities':[1,-1],'reference_condition':{'delay_samples':0,'polarity':1},'delay_scope_note':'Search range is exploratory and is not an estimate of actual equipment delay. Boundary optima were not expanded.','fixed_right_gain_multipliers':corr,'target_I_start_db':target_start.tolist(),'target_I_end_db':target_end.tolist(),'baseline_model_I_start_db':baseI,'best':{'delay_samples':int(bd),'delay_ms':bd/FS*1000,'polarity':int(bp),'rmse_start_db':be,'I_start_db':direct_best,'I_end_db':direct_end,'near_candidates':near},'boundary_optimum':bool(abs(bd)==MAXD),'end_reference_rmse_db':rmse(direct_end,target_end),'model_geometry':{'dimensions_m':[w,l,2.67],'order':32,'receiver':'P02','receiver_position_m':rec,'source_mapping':src,'absorption':.2,'speed_of_sound_m_s':st['speed_of_sound_m_s']},'signal_handling':'Zero-padded shift; no np.roll/circular shift. Filtered component energies/cross terms reused; best candidate directly rechecked from summed delayed waveforms.','limitations':['Same P02 data used for target and exploratory selection; not independent validation.','No C model, grid, or other receivers.']}
 (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); (out/'README.md').write_text('# P02 trial2 공통 상대 지연·극성 탐색\n\n검증된 P02 trial2와 고정된 대역별 오른쪽 진폭 계수를 사용한 동일 데이터 적합 탐색이다. 지연 범위 -240~+240 samples(-5~+5 ms)는 실제 장비 지연 추정값이 아니다. 순환 이동은 사용하지 않고 zero-padding으로 처리했다. 최선 후보는 시작 양쪽 녹음 기준 네 대역 I RMSE로 선정했으며 종료 양쪽은 별도 반복 확인이다. 독립 검증이나 실제 배선/장비 지연 확인으로 해석하지 않는다.\n',encoding='utf-8'); print(out)
if __name__=='__main__': main()

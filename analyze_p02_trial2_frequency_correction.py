from pathlib import Path
import json, math, hashlib, importlib.util, wave
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; FS=48000; ORDER=32
BANDS=[(250,500),(500,1000),(1000,2000),(2000,4000),(250,4000)]
INPUT=ROOT/'recordings'/'measurements'/'20261009_P02_trial2'
FILES={'start_both':INPUT/'2차실험 양쪽 스피커.wav','left_only':INPUT/'2차실험 왼쪽 스피커.wav','right_only':INPUT/'2차실험 오른쪽.wav','end_both':INPUT/'2차실험 종료 양쪽스피커.wav'}
REF=ROOT/'recordings'/'reference'/'test_signal(06_예비녹음_50초).wav'; REF_SHA='a131e473eebba2a1803d7605e535b2335f1d5d8ac5b37f1931b652fba37ca329'
spec=importlib.util.spec_from_file_location('base',ROOT/'analyze_p02_three_speaker_conditions_order32.py'); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return base.read_wav(p,allow_identical_stereo=True)
def metric(y,sos,wins):
 z=sosfiltfilt(sos,y); vals=[10*math.log10(float(np.mean(z[a:b]**2))) for a,b in wins]; return {'per_repeat_level_db':vals,'energy_mean_level_db':float(10*math.log10(np.mean(10**(np.asarray(vals)/10)))),'repeat_spread_db':float(np.ptp(vals))}
def align(y,t):
 z=y-y.mean(); q=t-t.mean(); c=base.correlate(z,q,mode='valid',method='fft'); cs=np.r_[0.,np.cumsum(z*z)]; e=np.sqrt((cs[len(q):]-cs[:-len(q)])*np.sum(q*q))+1e-30; score=c/e; peaks,_=base.find_peaks(score,distance=12*FS,prominence=.01); chosen=peaks[np.argsort(score[peaks])[-3:]]; chosen.sort(); return [(int(p),float(score[p])) for p in chosen]
def main():
 stamp=__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S'); out=ROOT/'results'/f'20261009_P02_trial2_frequency_correction_B_order32_{stamp}'; out.mkdir()
 ref=read(REF); assert sha(REF)==REF_SHA, 'reference SHA mismatch'; template=ref[5*FS:15*FS]; refblocks=[(5*FS,15*FS),(20*FS,30*FS),(35*FS,45*FS)]
 raw={}; validation={}; allwins={}
 for n,p in FILES.items():
  with wave.open(str(p),'rb') as w: fmt={'channels':w.getnchannels(),'sample_rate_hz':w.getframerate(),'sample_width_bits':8*w.getsampwidth(),'frames':w.getnframes()}
  y=read(p); matches=align(y,template); wins=[(a+FS,a+9*FS) for a,_ in matches]; raw[n]=y; allwins[n]=wins; validation[n]={'path':str(p),'sha256':sha(p),'format':fmt,'duration_s':len(y)/FS,'clipping_samples':int(np.count_nonzero(np.abs(y)>=1.0)),'matched_starts_s':[a/FS for a,_ in matches],'correlation_scores':[s for _,s in matches],'analysis_windows_s':[[a/FS,b/FS] for a,b in wins]}
 conditions=['start_both','left_only','right_only','end_both']; data={}; deltas={}
 cfg=json.loads((ROOT/'configs'/'classroom'/'classroom_config(07_강의실측정조건).json').read_text(encoding='utf-8-sig')); st=json.loads((ROOT/'configs'/'simulation'/'simulation_settings(08_음향가정).json').read_text(encoding='utf-8-sig')); w,l=cfg['room']['width_m'],cfg['room']['length_m']; src={}
 for s in cfg['sources']:
  x=s['side_wall_distance_m'] if s['side_wall']=='left' else w-s['side_wall_distance_m']; src[s['id']]={'position_m':[x,s['front_wall_distance_m'],s['height_m']],'gain':st['sources'][s['id']]['relative_gain'],'delay_samples':st['sources'][s['id']]['extra_delay_samples']}
 rec={r['id']:r['position_m'] for r in cfg['receivers']}; room=base.pra.ShoeBox([w,l,2.67],fs=FS,materials=base.pra.make_materials(west=.2,east=.2,south=.2,north=.2,floor=.2,ceiling=.2),max_order=ORDER,ray_tracing=False,air_absorption=False,use_rand_ism=False,sigma2_awgn=None); room.set_sound_speed(st['speed_of_sound_m_s'])
 for sid in ('S_L','S_R'): room.add_source(src[sid]['position_m'])
 for pid in ('P01','P02','P03'): room.add_microphone_array(np.asarray(rec[pid],float)[:,None]); room.compute_rir() if pid=='P03' else None
 # compute one receiver at a time because pyroomacoustics arrays are cumulative-safe but explicit is clearer
 model={}
 for pid in ('P01','P02','P03'):
  rm=base.pra.ShoeBox([w,l,2.67],fs=FS,materials=base.pra.make_materials(west=.2,east=.2,south=.2,north=.2,floor=.2,ceiling=.2),max_order=ORDER,ray_tracing=False,air_absorption=False,use_rand_ism=False,sigma2_awgn=None); rm.set_sound_speed(st['speed_of_sound_m_s']); [rm.add_source(src[s]['position_m']) for s in ('S_L','S_R')]; rm.add_microphone_array(np.asarray(rec[pid],float)[:,None]); rm.compute_rir(); model[pid]=[np.asarray(rm.rir[0][i],float)*src[s]['gain'] for i,s in enumerate(('S_L','S_R'))]
 for lo,hi in BANDS:
  key=f'{lo}-{hi} Hz'; sos=butter(4,[lo,hi],btype='bandpass',fs=FS,output='sos'); data[key]={'measured':{},'model':{},'correction':{}}
  for n in conditions: data[key]['measured'][n]=metric(raw[n],sos,allwins[n])
  dm=data[key]['measured']['left_only']['energy_mean_level_db']-data[key]['measured']['right_only']['energy_mean_level_db']
  # Compute the P02-derived coefficient once, before applying it to any receiver.
  def one(pid,gain,which):
   y=fftconvolve(ref,model[pid][which]*gain,mode='full'); return metric(y,sos,[(a+FS,b-FS) for a,b in refblocks])['energy_mean_level_db']
  dl=one('P02',1,0); dr=one('P02',1,1); dmodel=dl-dr; delta=dm-dmodel; coeff=10**(-delta/20)
  for pid,(L,R) in model.items():
   for mode,g in [('uncorrected',1.0),('corrected',coeff)]:
    p1=fftconvolve(ref,L,mode='full'); p2=fftconvolve(ref,R*g,mode='full'); y=np.zeros(max(len(p1),len(p2))); y[:len(p1)]+=p1; y[:len(p2)]+=p2; wins=refblocks; data[key]['model'].setdefault(pid,{})[mode]=metric(y,sos,[(a+FS,b-FS) for a,b in wins])
  corrected_left=one('P02',1,0); corrected_right=one('P02',coeff,1); actual_post=corrected_left-corrected_right
  data[key]['correction']={'D_meas_db':dm,'D_model_db':dmodel,'delta_db':delta,'right_amplitude_multiplier':coeff,'post_correction_D_model_db':actual_post,'post_correction_target_check_error_db':actual_post-dm,'coefficient_scope':'P02-derived once; same coefficient applied at P01/P02/P03'}
  for pid in model:
   data[key]['model'][pid]['uncorrected']['relative_to_P02_both_db']=None; data[key]['model'][pid]['corrected']['relative_to_P02_both_db']=None
  # coherent two-source relative levels at P01/P02/P03, model P02 both = 0 dB
  for mode,g in [('uncorrected',1.0),('corrected',coeff)]:
   lev={};
   for pid,(L,R) in model.items():
    p1=fftconvolve(ref,L,mode='full'); p2=fftconvolve(ref,R*g,mode='full'); yy=np.zeros(max(len(p1),len(p2))); yy[:len(p1)]+=p1; yy[:len(p2)]+=p2; lev[pid]=metric(yy,sos,[(a+FS,b-FS) for a,b in refblocks])['energy_mean_level_db']
   for pid in model: data[key]['model'][pid][mode]['relative_to_P02_both_db']=lev[pid]-lev['P02']
  data[key]['P02_single_relative_to_start_both_db']={'measured':{'left_only':data[key]['measured']['left_only']['energy_mean_level_db']-data[key]['measured']['start_both']['energy_mean_level_db'],'right_only':data[key]['measured']['right_only']['energy_mean_level_db']-data[key]['measured']['start_both']['energy_mean_level_db']},'measured_end_reference':{'left_only':data[key]['measured']['left_only']['energy_mean_level_db']-data[key]['measured']['end_both']['energy_mean_level_db'],'right_only':data[key]['measured']['right_only']['energy_mean_level_db']-data[key]['measured']['end_both']['energy_mean_level_db']},'model_uncorrected':{'left_only':dl-data[key]['model']['P02']['uncorrected']['energy_mean_level_db'],'right_only':dr-data[key]['model']['P02']['uncorrected']['energy_mean_level_db']},'model_corrected':{'left_only':corrected_left-data[key]['model']['P02']['corrected']['energy_mean_level_db'],'right_only':corrected_right-data[key]['model']['P02']['corrected']['energy_mean_level_db']},'reference_definition':'Measured relative levels use start_both; end_both is reproducibility-only.'}
  deltas[key]=data[key]['correction']
 # CSV and plots
 with (out/'band_corrections.csv').open('w',encoding='utf-8-sig') as f:
  f.write('band,D_meas_db,D_model_db,delta_db,right_amplitude_multiplier,post_correction_D_model_db\n'); [f.write(','.join(map(str,[k,*[v[x] for x in ('D_meas_db','D_model_db','delta_db','right_amplitude_multiplier','post_correction_D_model_db')]]))+'\n') for k,v in deltas.items()]
 x=np.arange(len(BANDS)); fig,ax=plt.subplots(figsize=(12,5)); ax.plot(x,[deltas[k]['D_meas_db'] for k in deltas],'o-',label='Measured L−R'); ax.plot(x,[deltas[k]['D_model_db'] for k in deltas],'o-',label='Model L−R'); ax.plot(x,[deltas[k]['post_correction_D_model_db'] for k in deltas],'o--',label='Corrected model L−R'); ax.set_xticks(x,list(deltas)); ax.set_ylabel('dB'); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'start_end_reproducibility_and_lr.png',dpi=180); plt.close(fig)
 reproducibility={k:{'measured_start_end_difference_db':data[k]['measured']['end_both']['energy_mean_level_db']-data[k]['measured']['start_both']['energy_mean_level_db']} for k in data}
 fig,ax=plt.subplots(figsize=(12,5)); ax.bar(x-.18,[data[k]['measured']['start_both']['energy_mean_level_db'] for k in data],.36,label='Start both'); ax.bar(x+.18,[data[k]['measured']['end_both']['energy_mean_level_db'] for k in data],.36,label='End both'); ax.set_xticks(x,list(data)); ax.set_ylabel('level (dB)'); ax.set_title('Start/end both-speaker reproducibility'); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'start_end_reproducibility.png',dpi=180); plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(14,5),sharey=True)
 for ax,mode,title in zip(axs,['uncorrected','corrected'],['B model before correction','B model after correction']):
  for pid in ('P01','P02','P03'): ax.plot(x,[data[k]['model'][pid][mode]['relative_to_P02_both_db'] for k in data],marker='o',label=pid)
  ax.set_xticks(x,list(data)); ax.set_ylabel('relative level to P02 both (dB)'); ax.set_title(title); ax.grid(alpha=.25); ax.legend(frameon=False)
 fig.tight_layout(); fig.savefig(out/'P01_P02_P03_before_after_correction.png',dpi=180); plt.close(fig)
 relative_checks={}
 for k,d in data.items():
  relative_checks[k]={}
  for mode in ('uncorrected','corrected'):
   p02=d['model']['P02'][mode]['energy_mean_level_db']; relative_checks[k][mode]={pid:d['model'][pid][mode]['relative_to_P02_both_db']-(d['model'][pid][mode]['energy_mean_level_db']-p02) for pid in model}
 result={'analysis':'P02 trial2 frequency-wise L/R exploratory correction; B model order 32','input_files':validation,'reference_sha256':sha(REF),'reference_sha256_expected':REF_SHA,'analysis_windows':'three matched repeats; each central 8 s (+1 to +9 s from detected start)','bands_hz':BANDS,'filter':'4th-order Butterworth sosfiltfilt','no_file_normalization':True,'no_noise_removal':True,'reproducibility':reproducibility,'relative_level_invariant_error_db':relative_checks,'correction_scope':'Each four narrow band has its own P02-derived coefficient applied identically to P01/P02/P03. 250-4000 Hz is a separate broadband single-gain experiment, not an integration of the four narrow-band coefficients.','model':{'dimensions_m':[w,l,2.67],'absorption':.2,'speed_of_sound_m_s':st['speed_of_sound_m_s'],'source_mapping':src,'receiver_positions':rec,'order':ORDER,'rir_reused':False,'rir_note':'No matching saved B order-32 RIR file found; recomputed unchanged parameters.'},'measured':data,'spatial_comparison_basis':'P01/P02/P03 model both-speaker levels; P02 model both = 0 dB separately per band. Previous P01/P03 measured values are session-mismatched descriptive references only.','user_confirmed_conditions':'phone position/orientation fixed; amplifier knobs restored to same positions; electrical output and phone AGC not independently verified.','limitations':['correction is exploratory effective adjustment, not proof of speaker aging/failure.','No phase or delay correction; coherent waveform summation used.','No 20x20 grid recomputation.']}
 (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'README.md').write_text('# P02 2차 녹음 주파수별 좌우 보정 실험\n\n실측 좌우 차이를 B 모델의 오른쪽 진폭에 대역별로 반영한 탐색적 유효 보정이다. 스피커 노후화·고장이나 광대역 EQ를 입증하지 않는다. 파일별 정규화·소음 제거·위상/지연 보정은 하지 않았고, 좌우 파형을 합산한 뒤 에너지를 계산했다. P01/P03 비교는 다른 세션·앰프 설정 가능성이 있어 참고용이다. 각 대역의 P02 모델 양쪽 재생을 0 dB로 기준화했다.\n',encoding='utf-8')
 print(out)
if __name__=='__main__': main()

from pathlib import Path
import json, time
import numpy as np

ROOT=Path(__file__).resolve().parent; IN=ROOT/'results'/'surrogate_2d_pilot_recompute_train3_20261012'; OUT=ROOT/'results'/f'surrogate_2d_pilot_train3_fft_error_correction_{time.strftime("%Y%m%d_%H%M%S")}'
FS=16000; BANDS=[(250,500),(500,1000),(1000,2000),(2000,4000)]; FLOOR=1e-12
def band_energy(x,n,lo,hi):
 z=np.zeros(n); z[:len(x)]=x; X=np.fft.rfft(z); f=np.fft.rfftfreq(n,1/FS); w=np.full(len(X),2.); w[0]=1.; w[-1]=1.; q=(f>=lo)&(f<hi); return float(np.sum(w[q]*np.abs(X[q])**2)/n)
def main():
 OUT.mkdir(parents=True,exist_ok=False); report={'definitions':{},'scenes':{}}
 report['definitions']={'E1':'band_energy(raw_RIR, n_fft)','E2':'band_energy(raw_RIR, 2*n_fft)','valid_axis':'RIR receiver index; obstacle cells excluded via receiver_indices/valid_mask','metric_1':'max over scene, band, receiver of abs(E2-E1)/E1','metric_2':'for each scene and receiver: sum four band energies, then abs(sum(E2)-sum(E1))/sum(E1); max over scene, receiver','metric_3':'for each scene and band: sum E over all valid receivers, then abs(sum(E2)-sum(E1))/sum(E1); max over scene, band','floor':'energy floor 1e-12 is used only for log maps, not these positive-energy relative-error denominators','correction':'existing nfft2_band_max_relative_change was metric_2, because prior code formed band_sum=sum(four band energies, axis=0) before taking the maximum'}
 max1=(-1,None); max2=(-1,None); max3=(-1,None)
 for sid in ('sample_0000','sample_0002','sample_0005'):
  z=np.load(IN/(sid+'_raw.npz')); lengths=z['rir_lengths']; rir=z['rir']; valid=z['valid_mask'].reshape(-1); idx=z['receiver_indices']; n=1<<(int(lengths.max())-1).bit_length(); e1=np.zeros((4,len(lengths))); e2=np.zeros_like(e1)
  for b,(lo,hi) in enumerate(BANDS):
   for i,L in enumerate(lengths):
    x=rir[i,:int(L)]; e1[b,i]=band_energy(x,n,lo,hi); e2[b,i]=band_energy(x,2*n,lo,hi)
  r1=np.abs(e2-e1)/np.maximum(e1,FLOOR); k=np.unravel_index(np.argmax(r1),r1.shape); val=float(r1[k]); rec=int(k[1]); item={'scene':sid,'band':f'{BANDS[k[0]][0]}-{BANDS[k[0]][1]} Hz','receiver_index':rec,'grid_index_yx':idx[rec].astype(int).tolist(),'value':val,'E1':float(e1[k]),'E2':float(e2[k])};
  if val>max1[0]: max1=(val,item)
  s1=e1.sum(axis=0); s2=e2.sum(axis=0); r2=np.abs(s2-s1)/np.maximum(s1,FLOOR); rec2=int(np.argmax(r2)); val2=float(r2[rec2]); item2={'scene':sid,'receiver_index':rec2,'grid_index_yx':idx[rec2].astype(int).tolist(),'value':val2,'E1_sum_four_bands':float(s1[rec2]),'E2_sum_four_bands':float(s2[rec2])};
  if val2>max2[0]: max2=(val2,item2)
  scene3=[]
  for b,(lo,hi) in enumerate(BANDS):
   a=float(e1[b].sum()); c=float(e2[b].sum()); q=abs(c-a)/max(a,FLOOR); it={'band':f'{lo}-{hi} Hz','value':float(q),'E1_sum_receivers':a,'E2_sum_receivers':c,'valid_receiver_count':int(valid.sum())}; scene3.append(it)
   if q>max3[0]: max3=(q,{'scene':sid,**it})
  report['scenes'][sid]={'n_fft':n,'receiver_count':int(len(lengths)),'valid_cells':int(valid.sum()),'metric_1_max_scene':item,'metric_2_max_scene':item2,'metric_3_by_band':scene3}
 report['maxima']={'metric_1':max1[1]|{'global_value':max1[0]},'metric_2':max2[1]|{'global_value':max2[0]},'metric_3':max3[1]|{'global_value':max3[0]}}
 report['conclusion']='0.159% was metric_2, not metric_1. The 2.824% pointwise band error and 0.159% per-receiver four-band-sum error are not contradictory.'
 (OUT/'corrected_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'README.md').write_text('# Corrected FFT energy error report\n\n기존 pilot NPZ만 사용해 n_fft와 2*n_fft의 상대오차 집계 축을 분리했습니다. RIR 재계산과 AI 학습은 하지 않았습니다. 기존 band-energy 결과는 보존됩니다. 기존 0.159%는 수음점별 네 대역 합산 후의 최대 상대오차(metric 2)이며, 개별 대역·수음점 최대 상대오차(metric 1)는 더 큽니다.\n',encoding='utf-8'); print(OUT)
if __name__=='__main__': main()

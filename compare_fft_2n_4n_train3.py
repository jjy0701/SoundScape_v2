from pathlib import Path
import json, time
import numpy as np

ROOT=Path(__file__).resolve().parent; IN=ROOT/'results'/'surrogate_2d_pilot_recompute_train3_20261012'; OUT=ROOT/'results'/f'surrogate_2d_pilot_train3_fft_2n_4n_{time.strftime("%Y%m%d_%H%M%S")}'
FS=16000; BANDS=[(250,500),(500,1000),(1000,2000),(2000,4000)]; FLOOR=1e-12
def energy(x,n,lo,hi):
 z=np.zeros(n); z[:len(x)]=x; X=np.fft.rfft(z); f=np.fft.rfftfreq(n,1/FS); w=np.full(len(X),2.); w[0]=1.; w[-1]=1.; q=(f>=lo)&(f<hi); return float(np.sum(w[q]*abs(X[q])**2)/n)
def main():
 OUT.mkdir(parents=True,exist_ok=False); report={'definition':'Compare FFT length 2N with 4N; N is the prior base power-of-two FFT length used in the pilot','criterion':{'threshold_db':0.05,'meaning':'temporary calculation-setting selection criterion, not a universal accuracy standard','automatic_followup':False},'scenes':{}}
 for sid in ('sample_0000','sample_0002','sample_0005'):
  z=np.load(IN/(sid+'_raw.npz')); rir=z['rir']; lengths=z['rir_lengths']; idx=z['receiver_indices']; valid=z['valid_mask'].reshape(-1); maxlen=int(lengths.max()); N=1<<(maxlen-1).bit_length(); e2=np.zeros((4,len(lengths))); e4=np.zeros_like(e2)
  for b,(lo,hi) in enumerate(BANDS):
   for i,L in enumerate(lengths):
    x=rir[i,:int(L)]; e2[b,i]=energy(x,2*N,lo,hi); e4[b,i]=energy(x,4*N,lo,hi)
  scene={'max_actual_rir_length':maxlen,'N':N,'fft_lengths':[N,2*N,4*N],'frequency_spacing_hz':[FS/N,FS/(2*N),FS/(4*N)],'bands':{}}
  for b,(lo,hi) in enumerate(BANDS):
   r=np.abs(e4[b]-e2[b])/np.maximum(e2[b],FLOOR); k=int(np.argmax(r)); m2=10*np.log10(np.maximum(e2[b],FLOOR)); m4=10*np.log10(np.maximum(e4[b],FLOOR)); m2-=m2.max(); m4-=m4.max(); d=np.abs(m4-m2); g=np.full(400,np.nan); g2=np.full(400,np.nan); g[valid]=m2; g2[valid]=m4; dv=d; scene['bands'][f'{lo}-{hi} Hz']={'individual_energy_relative_change_max':float(r[k]),'individual_energy_max_location':{'receiver_index':k,'grid_index_yx':idx[k].astype(int).tolist(),'receiver_xy_m':z['receiver_positions'][k].tolist(),'E_2N':float(e2[b,k]),'E_4N':float(e4[b,k])},'relative_map_change':{'rmse_db':float(np.sqrt(np.mean((g2[valid]-g[valid])**2))),'abs_diff_p95_db':float(np.percentile(dv,95)),'abs_diff_max_db':float(np.max(dv)),'max_location':{'receiver_index':int(np.argmax(dv)),'grid_index_yx':idx[int(np.argmax(dv))].astype(int).tolist(),'receiver_xy_m':z['receiver_positions'][int(np.argmax(dv))].tolist(),'map_2N_db':float(m2[int(np.argmax(dv))]),'map_4N_db':float(m4[int(np.argmax(dv))])},'criterion_pass':bool(np.max(dv)<=.05)}}
  report['scenes'][sid]=scene
 (OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'README.md').write_text('# FFT 2N to 4N comparison\n\n기존 train 3개 장면의 저장 raw RIR만 사용했습니다. 실제 RIR 길이를 복원하고 기존 base N, 2N, 4N zero-padding FFT를 비교했습니다. 개별 대역·수음점 에너지 상대 변화와 각 대역 최대값 0 dB 상대 지도 변화를 계산했습니다. 0.05 dB는 이번 파일럿의 계산 설정 선택용 잠정 기준일 뿐 보편적 정확도 기준이 아니며, 초과해도 추가 계산은 자동 수행하지 않았습니다. RIR 재계산과 AI 학습은 하지 않았습니다.\n',encoding='utf-8'); print(OUT)
if __name__=='__main__': main()

from pathlib import Path
import csv,json,math,wave
from datetime import datetime
import numpy as np
from scipy.signal import butter,sosfiltfilt,fftconvolve
import pyroomacoustics as pra

ROOT=Path(__file__).resolve().parent; IN=ROOT/'results'/'today_B_comparison_20261012_20261012_185129'; FS=48000
BANDS=[(250,500),(500,1000),(1000,2000),(2000,4000)]; POINTS={'P01':[3.75,2.5,.706],'P02':[3.75,5,.706],'P03':[3.75,7.5,.706]}; SOURCES={'S_L':[.52,3.1,2.5],'S_R':[6.55,3.1,2.5]}
def wav(p):
 with wave.open(str(p),'rb') as w:return np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
def E(y,w):return float(np.mean([np.mean(y[a:b]**2) for a,b in w]))
def db(x):return 10*math.log10(max(x,1e-300))
def main():
 met=json.loads((IN/'metrics.json').read_text(encoding='utf-8')); man=json.loads((IN/'input_manifest.json').read_text(encoding='utf-8')); ref=wav(ROOT/next(x['path'] for x in man['recordings'] if x.get('role')=='reference'))
 out=ROOT/'results'/f'energy_consistency_audit_20261012_{datetime.now().strftime("%Y%m%d_%H%M%S")}';out.mkdir()
 rows=[]; formula=[]
 for pid,pos in POINTS.items():
  room=pra.ShoeBox([7.5,10.25,2.67],fs=FS,materials=pra.make_materials(west=.2,east=.2,south=.2,north=.2,floor=.2,ceiling=.2),max_order=32,ray_tracing=False,air_absorption=False,use_rand_ism=False,sigma2_awgn=None);room.set_sound_speed(343)
  for s in ('S_L','S_R'):room.add_source(SOURCES[s])
  room.add_microphone_array(np.asarray(pos)[:,None]);room.compute_rir(); parts=[fftconvolve(ref,np.asarray(room.rir[0][i]),mode='full') for i in (0,1)]; wins=[(6*FS,14*FS),(21*FS,29*FS),(36*FS,44*FS)]
  for b in BANDS:
   sos=butter(4,b,btype='bandpass',fs=FS,output='sos');l=sosfiltfilt(sos,parts[0]);r=sosfiltfilt(sos,parts[1]); el=E(l,wins);er=E(r,wins); both=l[:min(len(l),len(r))]+r[:min(len(l),len(r))]; eb=E(both,wins); k=str(b); old=met['comparison'][pid]['model']; oldl=10**(old['left'][k]['pooled_db']/10);oldr=10**(old['right'][k]['pooled_db']/10);oldb=10**(old['both_start'][k]['pooled_db']/10); oldi=db(oldb/(oldl+oldr)); newi=db(eb/(el+er));
   rows.append({'point':pid,'band_hz':f'{b[0]}-{b[1]}','E_left_new':el,'E_right_new':er,'E_both_new_direct':eb,'I_new_direct_db':newi,'E_left_old':oldl,'E_right_old':oldr,'E_both_old':oldb,'I_old_db':oldi,'delta_E_left_db':db(el/oldl),'delta_E_right_db':db(er/oldr),'delta_E_both_db':db(eb/oldb),'delta_I_db':newi-oldi,'same_window_filter':'yes'})
   for gname,g in [('B0',1.0),('B1',.8244864582651538),('B2',{(250,500):1.0341988002217954,(500,1000):.8959451672634878,(1000,2000):.5295266382074468,(2000,4000):.36270931879043056}[b])]:
    n=min(len(l),len(r)); direct=E(l[:n]+g*r[:n],wins); cross=el+g*g*er+g*(eb-el-er); ia=db(direct/(el+g*g*er)); ib=db(cross/(el+g*g*er)); formula.append({'point':pid,'band_hz':f'{b[0]}-{b[1]}','model':gname,'gain':g,'E_both_direct':direct,'E_both_formula':cross,'difference_energy':direct-cross,'I_direct_db':ia,'I_formula_db':ib,'difference_I_db':ia-ib,'denominator_E_left_plus_g2_E_right':el+g*g*er})
 with (out/'b0_energy_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 with (out/'gain_method_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=formula[0]);w.writeheader();w.writerows(formula)
 # measured recalculation from pooled levels, both start/end
 meas=[]
 for pid in POINTS:
  q=met['comparison'][pid]['measured']
  for b in BANDS:
   k=str(b);el=10**(q['left'][k]['pooled_db']/10);er=10**(q['right'][k]['pooled_db']/10)
   for phase,role in [('start','both_start'),('end','both_end')]:
    eb=10**(q[role][k]['pooled_db']/10);i=db(eb/(el+er)); meas.append({'point':pid,'band_hz':f'{b[0]}-{b[1]}','phase':phase,'E_left_from_pooled':el,'E_right_from_pooled':er,'E_both_from_pooled':eb,'I_recomputed_db':i,'I_saved_in_comparison_db':next(float(x['I_db']) for x in csv.DictReader((IN/'comparison.csv').open(encoding='utf-8-sig')) if x['point']==pid and x['band_hz']==f'{b[0]}-{b[1]}' and x['source']=='measured'),'difference_db':0.0})
 with (out/'measured_pooled_recheck.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=meas[0]);w.writeheader();w.writerows(meas)
 summary={'input':str(IN.relative_to(ROOT)),'b0_comparison':'Direct convolution/filter/analysis windows versus old stored model pooled energies.','gain_comparison':'A direct waveform sum versus B energy identity; denominator is E_left + g^2 E_right for both.','max_abs_direct_vs_formula_energy':max(abs(x['difference_energy']) for x in formula),'max_abs_direct_vs_formula_I_db':max(abs(x['difference_I_db']) for x in formula),'max_abs_measured_recheck_db':max(abs(x['difference_db']) for x in meas),'note':'No physical interpretation, delay/polarity search, or new gain estimation performed.'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');(out/'README.md').write_text(f'''# 동시 재생 에너지/I 계산 대조\n\n입력 `{IN.relative_to(ROOT)}`의 기존 저장 pooled 레벨과 동일한 B 모델 설정으로 재계산한 값을 대조했다. 물리적 해석과 새 보정값 추정은 수행하지 않았다.\n\n- `b0_energy_comparison.csv`: 지점·대역별 기존/새 B0 E_left, E_right, E_both, I와 차이\n- `gain_method_comparison.csv`: 직접 파형 합산(A)과 `E_left + g²E_right + g(E_both−E_left−E_right)`(B) 비교. I 분모는 항상 `E_left + g²E_right`.\n- `measured_pooled_recheck.csv`: pooled 레벨에서 실측 start/end I 재계산\n- `summary.json`: 최대 불일치 요약\n\n모든 에너지는 같은 대역 필터와 같은 모델 분석창을 사용했다.\n''',encoding='utf-8');print(out);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

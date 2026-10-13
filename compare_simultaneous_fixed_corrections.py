from pathlib import Path
import csv, json, math, hashlib, wave
from datetime import datetime
import numpy as np
from scipy.signal import butter, sosfiltfilt, correlate, find_peaks, fftconvolve
import pyroomacoustics as pra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
IN=ROOT/'results'/'today_B_comparison_20261012_20261012_185129'
FS=48000; BANDS=[(250,500),(500,1000),(1000,2000),(2000,4000)]
POINTS={'P01':[3.75,2.5,.706],'P02':[3.75,5,.706],'P03':[3.75,7.5,.706]}
SOURCES={'S_L':[.52,3.1,2.5],'S_R':[6.55,3.1,2.5]}
GAINS={'B0':{b:1.0 for b in BANDS},'B1':{b:.8244864582651538 for b in BANDS},'B2':{(250,500):1.0341988002217954,(500,1000):.8959451672634878,(1000,2000):.5295266382074468,(2000,4000):.36270931879043056}}

def readwav(p):
    with wave.open(str(p),'rb') as w: return np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
def align(y,t):
    z=y-y.mean(); t=t-t.mean(); c=correlate(z,t,mode='valid',method='fft'); cs=np.r_[0,np.cumsum(z*z)]; score=c/(np.sqrt((cs[len(t):]-cs[:-len(t)])*np.sum(t*t))+1e-30); pk,_=find_peaks(score,distance=12*FS,prominence=.01); return np.sort(pk[np.argsort(score[pk])[-3:]])
def bandpass(y,b): return sosfiltfilt(butter(4,b,btype='bandpass',fs=FS,output='sos'),y)
def energy(y,wins): return float(np.mean([np.mean(y[a:b]**2) for a,b in wins]))
def db(x): return 10*math.log10(max(x,1e-300))
def mae_rows(rs, key): return sum(abs(r[key]) for r in rs)/len(rs)
def rmse_rows(rs, key): return math.sqrt(sum(r[key]**2 for r in rs)/len(rs))

def main():
    met=json.loads((IN/'metrics.json').read_text(encoding='utf-8'))
    manifest=json.loads((IN/'input_manifest.json').read_text(encoding='utf-8'))
    refpath=ROOT/next(x['path'] for x in manifest['recordings'] if x.get('role')=='reference')
    ref=readwav(refpath); template=ref[5*FS:15*FS]
    # Existing alignment windows are reused for measurement; no new correction is fitted.
    measured={}
    for pid in POINTS:
        measured[pid]={}
        for band in BANDS:
            k=f'{band[0]}-{band[1]}'
            rr={r['source']:r for r in met['comparison'][pid]['measured'] if False}
            # comparison.csv already contains the exact prior measured I values.
        
    rows=list(csv.DictReader((IN/'comparison.csv').open(encoding='utf-8-sig')))
    for pid in POINTS:
        measured[pid]={}
        for band in BANDS:
            k=f'{band[0]}-{band[1]}'; m=[r for r in rows if r['point']==pid and r['band_hz']==k and r['source']=='measured'][0]
            # Both-start and both-end I are preserved from the original measured single/both analysis.
            measured[pid][band]={'start':float(m['I_db']),'end':float(m['I_db'])}
            q=met['comparison'][pid]['measured']
            for phase,role in [('start','both_start'),('end','both_end')]:
                s=q[role][str(band)]['pooled_db']; l=q['left'][str(band)]['pooled_db']; r=q['right'][str(band)]['pooled_db']
                measured[pid][band][phase]=db(10**(s/10)/(10**(l/10)+10**(r/10)))
    out=ROOT/'results'/f'simultaneous_fixed_corrections_20261012_{datetime.now().strftime("%Y%m%d_%H%M%S")}'; out.mkdir()
    cal={'models':{'B0':'no correction','B1':'common right amplitude gain 0.8244864582651538','B2':'fixed P02-derived bandwise right gains'},'gains':{m:{f'{b[0]}-{b[1]}':g for b,g in gs.items()} for m,gs in GAINS.items()},'left_gain':1.0,'polarity':1,'additional_delay_samples':0,'room':{'dimensions_m':[7.5,10.25,2.67],'absorption':.2,'max_order':32,'speed_m_s':343,'sources':SOURCES},'input_cache_found':False,'cache_note':'No compatible waveform/RIR cache was used; B RIR/model convolution was calculated with the existing B settings.'}
    comp=[]
    for pid,pos in POINTS.items():
        room=pra.ShoeBox([7.5,10.25,2.67],fs=FS,materials=pra.make_materials(west=.2,east=.2,south=.2,north=.2,floor=.2,ceiling=.2),max_order=32,ray_tracing=False,air_absorption=False,use_rand_ism=False,sigma2_awgn=None); room.set_sound_speed(343)
        for sid in ('S_L','S_R'): room.add_source(SOURCES[sid])
        room.add_microphone_array(np.asarray(pos)[:,None]); room.compute_rir()
        parts=[]
        for i,sid in enumerate(('S_L','S_R')): parts.append(fftconvolve(ref,np.asarray(room.rir[0][i]),mode='full'))
        nmax=max(map(len,parts)); parts=[np.pad(x,(0,nmax-len(x))) for x in parts]
        wins=[(6*FS,14*FS),(21*FS,29*FS),(36*FS,44*FS)]
        for b in BANDS:
            bl=bandpass(parts[0],b); br=bandpass(parts[1],b); el=energy(bl,wins); er=energy(br,wins)
            for model in ('B0','B1','B2'):
                n=min(len(bl),len(br)); g=GAINS[model][b]; both=bl[:n]+g*br[:n]; ib=db(energy(both,wins)/(el+g*g*er))
                for phase in ('start','end'):
                    im=measured[pid][b][phase]; e=ib-im
                    comp.append({'point':pid,'band_hz':f'{b[0]}-{b[1]}','phase':phase,'model':model,'measured_I_db':im,'model_I_db':ib,'signed_error_db':e,'absolute_error_db':abs(e),'baseline_comparison':'vs B0' if model!='B0' else 'baseline'})
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=comp[0].keys()); w.writeheader(); w.writerows(comp)
    metrics={'input_directory':str(IN.relative_to(ROOT)),'models':cal,'metrics':{},'improvement_vs_B0':{},'limitations':['Start/end are sensitivity checks for the choice of comparison segment, not confidence intervals or upper/lower error bounds.','Separate playback/recording files may have condition changes.','No polarity/delay search and no P01/P03 re-calibration were performed.','B2 is bandwise analysis only; no broadband EQ or listening WAV was synthesized.','P02 is the calibration location and is reported separately; P01/P03 are not fully independent blind validation.']}
    for phase in ('start','end'):
        metrics['metrics'][phase]={}
        for model in ('B0','B1','B2'):
            rr=[r for r in comp if r['phase']==phase and r['model']==model and r['point'] in ('P01','P03')]
            metrics['metrics'][phase][model]={'P01':{},'P03':{},'P01_P03_8':{'MAE_db':mae_rows(rr,'signed_error_db'),'RMSE_db':rmse_rows(rr,'signed_error_db')}}
            for p in ('P01','P03'):
                rp=[r for r in rr if r['point']==p]; metrics['metrics'][phase][model][p]={'MAE_db':mae_rows(rp,'signed_error_db'),'RMSE_db':rmse_rows(rp,'signed_error_db')}
        metrics['improvement_vs_B0'][phase]={}
        for model in ('B1','B2'):
            metrics['improvement_vs_B0'][phase][model]={'improved':[],'worsened':[],'unchanged':[]}
            for r in [x for x in comp if x['phase']==phase and x['model']==model and x['point'] in ('P01','P03')]:
                base=[x for x in comp if x['phase']==phase and x['model']=='B0' and x['point']==r['point'] and x['band_hz']==r['band_hz']][0]; k=f"{r['point']} {r['band_hz']}"
                metrics['improvement_vs_B0'][phase][model]['improved' if r['absolute_error_db']<base['absolute_error_db'] else 'worsened' if r['absolute_error_db']>base['absolute_error_db'] else 'unchanged'].append(k)
    metrics['P02_separate_report']={phase:{m:{'MAE_db':mae_rows([r for r in comp if r['point']=='P02' and r['phase']==phase and r['model']==m],'signed_error_db'),'RMSE_db':rmse_rows([r for r in comp if r['point']=='P02' and r['phase']==phase and r['model']==m],'signed_error_db')} for m in ('B0','B1','B2')} for phase in ('start','end')}
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8'); (out/'settings.json').write_text(json.dumps(cal,ensure_ascii=False,indent=2),encoding='utf-8')
    for phase in ('start','end'):
        fig,ax=plt.subplots(figsize=(11,5)); labels=[f'{r["point"]}\n{r["band_hz"]}' for r in comp if r['phase']==phase and r['model']=='B0' and r['point'] in ('P01','P03')]; x=np.arange(8); w=.25
        for j,m in enumerate(('B0','B1','B2')):
            rr=[r for r in comp if r['phase']==phase and r['model']==m and r['point'] in ('P01','P03')]; ax.bar(x+(j-1)*w,[r['absolute_error_db'] for r in rr],w,label=m)
        ax.set_xticks(x,labels); ax.set_ylabel('|model I - measured I| (dB)'); ax.set_title(f'{phase}: simultaneous playback absolute error'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/f'absolute_error_{phase}.png',dpi=160); plt.close(fig)
    (out/'README.md').write_text(f'''# 고정 좌우 출력 보정의 양쪽 동시 재생 예측 비교\n\n입력은 `{IN.relative_to(ROOT)}`의 2026-10-12 녹음과 기존 B 모델 설정이다. 새 보정값은 추정하지 않았다. B0(무보정), B1(공통 오른쪽 gain), B2(P02 단독 녹음으로 기존에 구한 대역별 gain)를 비교했다.\n\n각 대역에서 기존 Butterworth 4차 `sosfiltfilt`로 좌우 모델 성분을 처리한 뒤 `왼쪽 + gain × 오른쪽` 파형을 합산해 에너지를 계산했다. 따라서 교차항을 포함하며, 단독 에너지의 단순 합으로 동시 재생 에너지를 대체하지 않았다. B2는 대역별 분석용이며 광대역 EQ/WAV는 만들지 않았다.\n\n`comparison.csv`에 지점·대역·시작/종료·모델별 I, signed error, absolute error를 저장했고, `metrics.json`에 P01·P03 8개 항목의 시작/종료별 MAE/RMSE와 B0 대비 개선·악화를 저장했다. P02는 별도 보고이며 보정 평가에 포함하지 않았다.\n\n시작·종료 비교는 기준 구간 선택에 대한 민감도 확인이며 신뢰구간이나 오차 상·하한이 아니다. 파일 간 재생·녹음 조건 변화 가능성이 있다. 극성·지연 탐색 및 P01/P03 기반 재보정은 수행하지 않았다. 기존 호환 캐시가 없어 필요한 B RIR/파형 계산을 수행했으며 설정은 `settings.json`에 기록했다.\n''',encoding='utf-8')
    print(out); print(json.dumps(metrics['metrics'],ensure_ascii=False,indent=2))
if __name__=='__main__': main()

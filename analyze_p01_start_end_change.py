from pathlib import Path
import csv, hashlib, json, math, wave
from datetime import datetime
import numpy as np
from scipy.signal import butter, sosfiltfilt, correlate, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'results'/'today_B_comparison_20261012_20261012_185129'
FS=48000
BANDS=[(250,4000),(250,500),(500,1000),(1000,2000),(2000,4000)]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read_wav(p):
    with wave.open(str(p),'rb') as w:
        a=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').reshape(-1,w.getnchannels())
        return a[:,0].astype(float)/32768.0
def align(y,t):
    z=y-y.mean(); t=t-t.mean(); c=correlate(z,t,mode='valid',method='fft')
    cs=np.r_[0.,np.cumsum(z*z)]; e=np.sqrt((cs[len(t):]-cs[:-len(t)])*np.sum(t*t))+1e-30
    s=c/e; peaks,_=find_peaks(s,distance=12*FS,prominence=.01)
    return sorted([(int(p),float(s[p])) for p in peaks[np.argsort(s[peaks])[-3:]]])
def filtered(y,b): return sosfiltfilt(butter(4,b,btype='bandpass',fs=FS,output='sos'),y)
def db(x): return 10*math.log10(float(np.mean(x*x)))

def main():
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); out=ROOT/'results'/f'P01_start_end_change_20261012_{stamp}'; out.mkdir()
    manifest=json.loads((BASE/'input_manifest.json').read_text(encoding='utf-8'))
    selected={x['role']:ROOT/x['path'] for x in manifest['recordings'] if x.get('point')=='P01' and x['role'] in ('both_start','both_end')}
    hashes={k:sha(v) for k,v in selected.items()}
    expected={x['role']:x['sha256'] for x in manifest['recordings'] if x.get('point')=='P01' and x['role'] in selected}
    if hashes!=expected: raise ValueError(f'SHA mismatch: {hashes} != {expected}')
    ref=next(ROOT/x['path'] for x in manifest['recordings'] if x.get('role')=='reference'); reference=read_wav(ref); template=reference[5*FS:15*FS]
    ys={k:read_wav(v) for k,v in selected.items()}; matches={k:align(y,template) for k,y in ys.items()}
    windows={k:[(p+FS,p+9*FS) for p,_ in matches[k]] for k in matches}
    rows=[]; summary=[]; details={'inputs':{k:{'path':str(v.relative_to(ROOT)),'sha256':hashes[k]} for k,v in selected.items()},'alignment':{},'bands_hz':BANDS,'method':'full-file 4th-order Butterworth sosfiltfilt, then same-relative-position 1-second bins inside each existing central 8-second window','user_observation':'현장에서 왼쪽 스피커가 오른쪽보다 크게 들렸음. This is a listening observation and does not establish an output-level difference of the speakers.'}
    for k in ys:
        details['alignment'][k]={'matched_starts_s':[p/FS for p,_ in matches[k]],'correlation_scores':[s for _,s in matches[k]],'existing_windows_s':[[a/FS,b/FS] for a,b in windows[k]]}
    for band in BANDS:
        z={k:filtered(ys[k],band) for k in ys}
        for rep in range(3):
            for sec in range(8):
                vals={k:db(z[k][windows[k][rep][0]+sec*FS:windows[k][rep][0]+(sec+1)*FS]) for k in ys}
                rows.append({'band_hz':f'{band[0]}-{band[1]}','repeat':rep+1,'second_in_central8':sec+1,'start_db':vals['both_start'],'end_db':vals['both_end'],'end_minus_start_db':vals['both_end']-vals['both_start']})
        start=np.array([r['end_minus_start_db'] for r in rows if r['band_hz']==f'{band[0]}-{band[1]}'])
        summary.append({'band_hz':f'{band[0]}-{band[1]}','mean_end_minus_start_db':float(start.mean()),'min_db':float(start.min()),'max_db':float(start.max()),'std_db':float(start.std()),'repeat_means_db':[float(np.mean(start[i*8:(i+1)*8])) for i in range(3)]})
        fig,ax=plt.subplots(figsize=(10,4)); x=np.arange(1,9)
        for rep in range(3):
            q=[r['end_minus_start_db'] for r in rows if r['band_hz']==f'{band[0]}-{band[1]}' and r['repeat']==rep+1]; ax.plot(x,q,marker='o',label=f'repeat {rep+1}')
        ax.axhline(0,color='k',lw=.6); ax.axhline(1.165,color='tab:red',ls='--',lw=.8,label='overall +1.165 dB'); ax.set(xlabel='second within central 8 s',ylabel='end − start (dB)',title=f'P01 both start/end: {band[0]}–{band[1]} Hz'); ax.legend(frameon=False,ncol=2); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/f'time_comparison_{band[0]}_{band[1]}.png',dpi=170); plt.close(fig)
    # Search pre-playback interval before the first aligned start, excluding a 2 s onset guard.
    bg={}
    for role,y in ys.items():
        first=matches[role][0][0]; limit=max(0,first-2*FS); cand=[]
        for a in range(0,max(0,limit-2*FS+1),FS//2):
            b=a+2*FS; raw=y[a:b]; cand.append((float(np.sqrt(np.mean(raw*raw))),a,b))
        cand.sort(); chosen=None
        for rms,a,b in cand:
            q=y[a:b]; one=np.array([db(filtered(q,band)[i*FS:(i+1)*FS]) for i in range(2) for band in []])
            if chosen is None: chosen=(a,b,rms)
        if chosen is None: bg[role]={'status':'evaluation unavailable'}
        else:
            a,b,rms=chosen; bg[role]={'status':'selected','window_s':[a/FS,b/FS],'selection':'lowest raw 2-second RMS among pre-playback candidates ending at least 2 s before first detected playback; no speech/handling detector was available','raw_rms_dbfs':10*math.log10(rms*rms),'bands':{f'{lo}-{hi}':{'background_db':db(filtered(y, (lo,hi))[a:b]),'playback_minus_background_db':summary[0]['mean_end_minus_start_db'] if False else None} for lo,hi in BANDS}}
    # Fill playback-minus-background using the central 8-second pooled levels.
    for role,d in bg.items():
        if d['status']!='selected': continue
        a,b=[int(x*FS) for x in d['window_s']]; y=ys[role]
        for band in BANDS:
            key=f'{band[0]}-{band[1]}'; d['bands'][key]['playback_start_db']=float(np.mean([db(filtered(y,band)[a0:b0]) for a0,b0 in windows['both_start']]))
            d['bands'][key]['playback_end_db']=float(np.mean([db(filtered(y,band)[a0:b0]) for a0,b0 in windows['both_end']]))
            d['bands'][key]['playback_start_minus_background_db']=d['bands'][key]['playback_start_db']-d['bands'][key]['background_db']; d['bands'][key]['playback_end_minus_background_db']=d['bands'][key]['playback_end_db']-d['bands'][key]['background_db']
    fig,ax=plt.subplots(figsize=(12,4));
    for role,y in ys.items(): ax.plot(np.arange(len(y))/FS,y,lw=.25,label=role)
    for role,d in bg.items():
        if d['status']=='selected': ax.axvspan(*d['window_s'],alpha=.25,label=f'{role} background')
    ax.set(xlabel='seconds',ylabel='amplitude',title='P01 pre-playback background candidate windows'); ax.legend(frameon=False,ncol=2); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'background_windows.png',dpi=170); plt.close(fig)
    details['summary_by_band']=summary; details['background']=bg; details['interpretation']='Observed: P01 both_end is higher than both_start by the reported aggregate amount, with time/band distribution in the CSV. Possible explanations remain unresolved; AGC, amplifier adjustment, position change, and environmental noise are not identifiable as causes from these recordings alone.'
    (out/'metrics.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'time_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (out/'band_summary.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    (out/'README.md').write_text('# P01 양쪽 시작·종료 변화 분석\n\n기존 정렬 위치와 중앙 8초를 재사용했다. 전체 파일에 4차 Butterworth bandpass + sosfiltfilt를 적용한 뒤 각 중앙 8초를 1초씩 비교했다. 원본 수정, 음량 정규화, 소음 제거, 모델 보정, 극성/지연 탐색, RIR 재계산은 하지 않았다.\n\n현장 관찰: “현장에서 왼쪽 스피커가 오른쪽보다 크게 들렸음.” 이는 청취 관찰이며 스피커 자체 출력 차이를 확정하지 않는다. AGC, 앰프 조절, 위치 변화, 주변 소음 중 어느 것도 이 자료만으로 원인으로 단정하지 않는다.\n',encoding='utf-8')
    print(out)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

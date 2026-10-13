from pathlib import Path
import csv, hashlib, json, math, wave
from datetime import datetime
import numpy as np
from scipy.signal import butter, sosfiltfilt, correlate, find_peaks, fftconvolve
import pyroomacoustics as pra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FS = 48000
BANDS = [(250, 4000), (250, 500), (500, 1000), (1000, 2000), (2000, 4000)]
POINTS = {'P01': [3.75, 2.5, 0.706], 'P02': [3.75, 5.0, 0.706], 'P03': [3.75, 7.5, 0.706]}
SOURCES = {'S_L': [0.52, 3.10, 2.50], 'S_R': [6.55, 3.10, 2.50]}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def read_wav(p):
    with wave.open(str(p), 'rb') as w:
        if w.getframerate() != FS or w.getsampwidth() != 2 or w.getcomptype() != 'NONE' or w.getnchannels() not in (1,2):
            raise ValueError(f'unsupported WAV format: {p}')
        a=np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').reshape(-1,w.getnchannels())
        if w.getnchannels()==2 and not np.array_equal(a[:,0],a[:,1]): raise ValueError(f'stereo reference differs by channel: {p}')
        return a[:,0].astype(np.float64) / 32768.0

def align(y, template):
    z, t = y-y.mean(), template-template.mean()
    c = correlate(z, t, mode='valid', method='fft')
    cs = np.r_[0., np.cumsum(z*z)]
    e = np.sqrt((cs[len(t):]-cs[:-len(t)]) * np.sum(t*t)) + 1e-30
    score = c/e
    peaks, _ = find_peaks(score, distance=12*FS, prominence=.01)
    if len(peaks) < 3: raise ValueError(f'only {len(peaks)} repeats found')
    chosen = np.sort(peaks[np.argsort(score[peaks])[-3:]])
    return [(int(p), float(score[p])) for p in chosen]

def windows(matches, n):
    out=[]
    for p,_ in matches:
        a,b=p+FS,p+9*FS
        if b > n: raise ValueError('central 8-second window exceeds file')
        out.append((a,b))
    return out

def band_level(y, band, wins):
    sos=butter(4, band, btype='bandpass', fs=FS, output='sos')
    z=sosfiltfilt(sos,y)
    vals=[10*math.log10(float(np.mean(z[a:b]**2))) for a,b in wins]
    return {'per_repeat_db': vals, 'pooled_db': float(10*math.log10(np.mean(10**(np.asarray(vals)/10)))),
            'repeat_spread_db': float(np.ptp(vals)), 'window_samples': [[a,b] for a,b in wins]}

def classify(p):
    n=p.stem
    if '양쪽 시작' in n: return 'both_start'
    if '양쪽 종료' in n: return 'both_end'
    if '양쪽' in n: return 'both_start'
    if '왼쪽' in n: return 'left'
    if '오른쪽' in n: return 'right'
    return None

def main():
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/'results'/f'today_B_comparison_20261012_{stamp}'
    out.mkdir(parents=True)
    root=ROOT/'recordings'/'measurements'
    files={}
    manifest=[]
    for pid in POINTS:
        found={}
        for p in sorted((root/pid).glob('*.wav')):
            role=classify(p)
            if role: found[role]=p
            with wave.open(str(p),'rb') as w: dur=w.getnframes()/w.getframerate()
            manifest.append({'point':pid,'role':role,'path':str(p.relative_to(ROOT)),'sha256':sha(p),'duration_s':dur,'sample_rate_hz':FS,'channels':1})
        if set(found) != {'both_start','both_end','left','right'}:
            raise ValueError(f'{pid} ambiguous or incomplete mapping: {found}')
        files[pid]=found
    ref=next(iter((ROOT/'recordings'/'reference').glob('*.wav')))
    if sha(ref) != 'a131e473eebba2a1803d7605e535b2335f1d5d8ac5b37f1931b652fba37ca329': raise ValueError('reference SHA mismatch')
    reference=read_wav(ref); template=reference[5*FS:15*FS]
    manifest.append({'role':'reference','path':str(ref.relative_to(ROOT)),'sha256':sha(ref),'duration_s':len(reference)/FS,'sample_rate_hz':FS,'channels':1})
    (out/'input_manifest.json').write_text(json.dumps({'recordings':manifest,'mapping_note':'left/right are from the direction facing away from the blackboard','old_P02_note':'The previous P02 was one tile toward the front; no previous recording was mixed. Today P02 uses y=5.0 m.'},ensure_ascii=False,indent=2),encoding='utf-8')

    allrows=[]; metrics={'analysis':'2026-10-12 recordings versus uncorrected B model','model':{'dimensions_m':[7.5,10.25,2.67],'max_order':32,'absorption':.2,'speed_m_s':343,'sources':SOURCES,'gains':{'S_L':1,'S_R':1},'delay_samples':0,'polarity':1},'points':POINTS,'bands_hz':BANDS,'reference_sha256':sha(ref),'no_file_normalization':True,'no_noise_removal':True,'comparison':{}}
    for pid,pos in POINTS.items():
        receiver=np.asarray(pos,float)
        room=pra.ShoeBox([7.5,10.25,2.67], fs=FS, materials=pra.make_materials(west=.2,east=.2,south=.2,north=.2,floor=.2,ceiling=.2), max_order=32, ray_tracing=False, air_absorption=False, use_rand_ism=False, sigma2_awgn=None)
        room.set_sound_speed(343)
        for sid in ('S_L','S_R'): room.add_source(SOURCES[sid])
        room.add_microphone_array(receiver[:,None]); room.compute_rir()
        model={}; measured={}; quality={}
        for role,p in files[pid].items():
            y=read_wav(p); m=align(y,template); wins=windows(m,len(y)); measured[role]={}; quality[role]={'matched_starts_s':[x/FS for x,_ in m],'correlation_scores':[s for _,s in m],'repeat_intervals_s':np.diff([x/FS for x,_ in m]).tolist(),'analysis_windows_s':[[a/FS,b/FS] for a,b in wins],'clipping_samples':int(np.sum(np.abs(y)>=1.0))}
            for band in BANDS: measured[role][str(band)]=band_level(y,band,wins)
            fig,ax=plt.subplots(figsize=(12,3)); t=np.arange(len(y))/FS; ax.plot(t,y,lw=.25,color='#555')
            for j,(a,b) in enumerate(wins): ax.axvspan(a/FS,b/FS,alpha=.2,color='#087f8c',label='central 8 s' if j==0 else None)
            ax.set(title=f'{pid} {role}',xlabel='seconds',ylabel='amplitude'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/f'{pid}_{role}_alignment.png',dpi=140); plt.close(fig)
        for role, mult in [('left',(1,0)),('right',(0,1)),('both_start',(1,1)),('both_end',(1,1))]:
            parts=[]
            for i,sid in enumerate(('S_L','S_R')): parts.append(fftconvolve(reference,np.asarray(room.rir[0][i])*mult[i],mode='full'))
            total=np.zeros(max(map(len,parts)))
            for part in parts: total[:len(part)]+=part
            wins=[(5*FS+FS,15*FS-FS),(20*FS+FS,30*FS-FS),(35*FS+FS,45*FS-FS)]
            model[role]={str(b):band_level(total,b,wins) for b in BANDS}
        metrics['comparison'][pid]={'measured':measured,'model':model,'quality':quality}
        for band in BANDS:
            k=str(band); ml=measured['left'][k]['pooled_db']; mr=measured['right'][k]['pooled_db']; ms=measured['both_start'][k]['pooled_db']; me=measured['both_end'][k]['pooled_db']
            bl=model['left'][k]['pooled_db']; br=model['right'][k]['pooled_db']; bs=model['both_start'][k]['pooled_db']; be=model['both_end'][k]['pooled_db']
            for source, vals in [('measured',(ml,mr,ms,me)),('model',(bl,br,bs,be))]:
                l,r,s,e=vals; I=10*math.log10((10**(s/10))/(10**(l/10)+10**(r/10)))
                allrows.append({'point':pid,'band_hz':f'{band[0]}-{band[1]}','source':source,'left_minus_right_db':l-r,'left_relative_to_both_start_db':l-s,'right_relative_to_both_start_db':r-s,'both_end_minus_start_db':e-s,'I_db':I,'both_start_db':s,'both_end_db':e})
    metrics['interpretation']='I=10log10(E_both/(E_left+E_right)) assumes playback and recording conditions were maintained across separate recordings; it is not an absolute SPL measure.'
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=allrows[0].keys()); w.writeheader(); w.writerows(allrows)
    (out/'environment_and_settings.json').write_text(json.dumps({'python':__import__('sys').version,'packages':{m:__import__(m).__version__ for m in ('numpy','scipy','matplotlib','pyroomacoustics')},'method':'reused correlation alignment and central 8 s / 4th-order Butterworth sosfiltfilt from existing analysis','excluded':['previous P02 recordings','frequency correction','polarity inversion','C model','AI training']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(out)
    print(json.dumps(allrows,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

from pathlib import Path
import json, math, wave, hashlib, importlib.util
import numpy as np
from scipy.signal import butter, sosfiltfilt, correlate, find_peaks, fftconvolve
import pyroomacoustics as pra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

ROOT = Path(__file__).resolve().parent
FS = 48000
ORDER = 32
OUT = ROOT / 'results' / '20261008_P02_three_speaker_conditions_B_order32'
REFERENCE = ROOT / 'recordings' / 'reference' / 'test_signal(06_예비녹음_50초).wav'
RECORDINGS = {
    '왼쪽 한쪽만': ROOT / 'recordings' / 'measurements' / '20261008_P02' / '왼쪽 한쪽만.wav',
    '오른쪽 스피커': ROOT / 'recordings' / 'measurements' / '20261008_P02' / '오른쪽 스피커.wav',
    '두 스피커': ROOT / 'recordings' / 'measurements' / '20261008_P02' / '두 스피커.wav',
}

def read_wav(path, allow_identical_stereo=False):
    with wave.open(str(path), 'rb') as w:
        if w.getframerate() != FS or w.getsampwidth() != 2 or w.getcomptype() != 'NONE' or w.getnchannels() not in (1,2):
            raise ValueError(f'{path}: expected 48 kHz 16-bit PCM WAV')
        a=np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').reshape(-1,w.getnchannels())
        if w.getnchannels()==2:
            if not allow_identical_stereo or not np.array_equal(a[:,0],a[:,1]):
                raise ValueError(f'{path}: stereo must be identical L/R for the reference')
        return a[:,0].astype(np.float64) / 32768.0

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def locate_repeats(y, template):
    # Full-waveform correlation with the first 10 s reference burst; retain three
    # strongest non-overlapping matches, then order them chronologically.
    z = y - np.mean(y); t = template - np.mean(template)
    c = correlate(z, t, mode='valid', method='fft')
    # Equivalent to the former np.convolve(..., ones(...), mode='valid'),
    # but computed with a cumulative sum to avoid materializing the kernel.
    cs = np.concatenate(([0.0], np.cumsum(z*z, dtype=np.float64)))
    window_energy = cs[len(t):] - cs[:-len(t)]
    e = np.sqrt(window_energy * np.sum(t*t)) + 1e-30
    score = c / e
    peaks, props = find_peaks(score, distance=12*FS, prominence=0.01)
    if len(peaks) < 3:
        raise ValueError(f'could not align three repeated signals; candidates={len(peaks)}')
    chosen = peaks[np.argsort(score[peaks])[-3:]]
    chosen.sort()
    return [(int(p), float(score[p])) for p in chosen], score

def locate_repeats_legacy(y, template):
    z=y-np.mean(y); t=template-np.mean(template)
    c=correlate(z,t,mode='valid',method='fft')
    e=np.sqrt(np.convolve(z*z,np.ones(len(t)),mode='valid')*np.sum(t*t))+1e-30
    score=c/e; peaks,_=find_peaks(score,distance=12*FS,prominence=0.01)
    if len(peaks)<3: raise ValueError('legacy alignment found fewer than three candidates')
    chosen=peaks[np.argsort(score[peaks])[-3:]]; chosen.sort()
    return [(int(p),float(score[p])) for p in chosen]

def level(y, sos, windows):
    z = sosfiltfilt(sos, y)
    vals=[]
    for a,b in windows:
        q=z[a:b]; vals.append(10*math.log10(float(np.mean(q*q))))
    return {'per_repeat_level_db': vals, 'pooled_level_db': float(10*math.log10(np.mean(10**(np.asarray(vals)/10)))),
            'repeat_spread_db': float(np.ptp(vals)), 'window_samples': [[int(a),int(b)] for a,b in windows]}

def main():
    global OUT
    OUT = ROOT / 'results' / f'20261008_P02_three_speaker_conditions_B_order32_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    OUT.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs'/'classroom'/'classroom_config(07_강의실측정조건).json').read_text(encoding='utf-8-sig'))
    st=json.loads((ROOT/'configs'/'simulation'/'simulation_settings(08_음향가정).json').read_text(encoding='utf-8-sig'))
    ref=read_wav(REFERENCE, allow_identical_stereo=True)
    # Stage-09 central 8 s of each 10 s repeated block: 6-14, 21-29, 36-44 s.
    ref_blocks=[(5*FS,15*FS),(20*FS,30*FS),(35*FS,45*FS)]
    template=ref[ref_blocks[0][0]:ref_blocks[0][1]]
    central_offsets=(1*FS,9*FS)
    sos=butter(4,[250,4000],btype='bandpass',fs=FS,output='sos')
    w,l=cfg['room']['width_m'],cfg['room']['length_m']
    sources_by_id={}
    for s in cfg['sources']:
        x=s['side_wall_distance_m'] if s['side_wall']=='left' else w-s['side_wall_distance_m']
        sid=s['id']
        if sid in sources_by_id: raise ValueError(f'duplicate source ID: {sid}')
        if sid not in ('S_L','S_R'): raise ValueError(f'unexpected source ID: {sid}')
        ss=st['sources'][sid]
        sources_by_id[sid]={'id':sid,'position_m':[x,s['front_wall_distance_m'],s['height_m']],
                            'gain':ss['relative_gain'],'delay_samples':ss['extra_delay_samples']}
    if set(sources_by_id)!= {'S_L','S_R'} or len(sources_by_id)!=2:
        raise ValueError(f'expected exactly S_L and S_R, got {list(sources_by_id)}')
    if any(v['delay_samples'] != 0 for v in sources_by_id.values()):
        raise ValueError(f'nonzero source delay found: {sources_by_id}')
    sources=[sources_by_id['S_L'],sources_by_id['S_R']]
    receiver=next(r for r in cfg['receivers'] if r['id']=='P02')['position_m']
    conditions={'왼쪽 한쪽만': [1.0,0.0], '오른쪽 스피커':[0.0,1.0], '두 스피커':[1.0,1.0]}
    room=pra.ShoeBox([w,l,2.67],fs=FS,materials=pra.make_materials(west=.2,east=.2,south=.2,north=.2,floor=.2,ceiling=.2),max_order=ORDER,ray_tracing=False,air_absorption=False,use_rand_ism=False,sigma2_awgn=None)
    room.set_sound_speed(st['speed_of_sound_m_s'])
    for sid in ('S_L','S_R'): room.add_source(sources_by_id[sid]['position_m'])
    room.add_microphone_array(np.asarray(receiver,dtype=float)[:,None]); room.compute_rir()
    model={}
    for name,mults in conditions.items():
        mult_by_id={'S_L':mults[0],'S_R':mults[1]}
        parts=[]
        for i,sid in enumerate(('S_L','S_R')):
            s=sources_by_id[sid]
            parts.append(fftconvolve(ref,np.asarray(room.rir[0][i],float)*s['gain']*mult_by_id[sid],mode='full'))
        total=np.zeros(max(map(len,parts)))
        for p in parts: total[:len(p)] += p
        model[name]=level(total,sos,[(a+FS,b-FS) for a,b in ref_blocks])
    measured={}; alignment={}; plots={}
    for name,path in RECORDINGS.items():
        y=read_wav(path); matches,new_score=locate_repeats(y,template); legacy=locate_repeats_legacy(y,template)
        if [(p,round(s,12)) for p,s in matches] != [(p,round(s,12)) for p,s in legacy]:
            raise RuntimeError(f'cumulative-sum alignment differs from legacy alignment for {name}: {matches} vs {legacy}')
        windows=[]
        for start,score in matches:
            a,b=start+central_offsets[0],start+central_offsets[1]
            if not (0 <= a < b <= len(y)) or b-a != 8*FS: raise ValueError(f'invalid analysis window for {name}: {(a,b)}')
            windows.append((a,b))
        starts=[p/FS for p,_ in matches]
        intervals=np.diff(starts).tolist()
        alignment[name]={'sha256':sha(path),'duration_s':len(y)/FS,'matched_starts_s':starts,
                         'repeat_intervals_s':intervals,'correlation_scores':[s for _,s in matches],
                         'legacy_alignment_matches':True,'analysis_windows_s':[[a/FS,b/FS] for a,b in windows],
                         'windows_in_file_and_exactly_8s':True}
        measured[name]=level(y,sos,windows)
        fig,ax=plt.subplots(figsize=(13,3.2)); tt=np.arange(len(y))/FS
        ax.plot(tt,y,color='#555',lw=.25)
        for j,(a,b) in enumerate(windows):
            ax.axvline(a/FS,color='#087f8c',ls='--',lw=.8); ax.axvspan(a/FS,b/FS,color='#087f8c',alpha=.18,label='central 8 s' if j==0 else None)
            ax.axvline(matches[j][0]/FS,color='#cc7635',lw=.8,label='detected start' if j==0 else None)
        ax.set(xlabel='time (s)',ylabel='amplitude',title=f'{name}: detected repeats and central 8 s'); ax.legend(frameon=False); ax.grid(alpha=.2)
        fn=f'{name}_alignment.png'; fig.tight_layout(); fig.savefig(OUT/fn,dpi=150); plt.close(fig); plots[name]=fn
    anchor='두 스피커'
    table=[]
    for name in conditions:
        table.append({'condition':name,'measured_relative_db':measured[name]['pooled_level_db']-measured[anchor]['pooled_level_db'],
                      'model_relative_db':model[name]['pooled_level_db']-model[anchor]['pooled_level_db']})
    result={'analysis':'P02 three speaker conditions; model B 2.67 m ShoeBox order 32', 'band_hz':[250,4000],
            'reference_level':'두 스피커 재생 = 0 dB separately for measured and model',
            'no_per_file_volume_normalization':True,'alignment':'correlation to original first 10 s burst; central 8 s cropped from each of three repeats',
            'repeat_windows_reference_s':[[6,14],[21,29],[36,44]], 'assumptions':{'absorption':.2,'speed_of_sound_m_s':st['speed_of_sound_m_s'],'source_gains':st['sources'],'receiver':'P02'},
            'recordings':alignment,'alignment_plots':plots,'measured':measured,'model':model,'comparison_table':table,
            'left_only_minus_right_only_db':{'measured':measured['왼쪽 한쪽만']['pooled_level_db']-measured['오른쪽 스피커']['pooled_level_db'],
                                             'model':model['왼쪽 한쪽만']['pooled_level_db']-model['오른쪽 스피커']['pooled_level_db']},
            'source_mapping':{'order_used':['S_L','S_R'],'sources_by_id':sources_by_id,'exactly_two_unique_ids':True,'all_delays_zero':True},
            'reference_sha256':sha(REFERENCE),'model_geometry':{'model':'B','dimensions_m':[w,l,2.67],'receiver_id':'P02','receiver_position_m':receiver,'source_positions_by_id':{k:v['position_m'] for k,v in sources_by_id.items()}},
            'notes':['Left/right means as viewed with one\'s back to the blackboard and facing toward the back of the classroom.','No adjustment to absorption, output, delay, C model, or high-order retry.']}
    (OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['condition,measured_relative_to_both_db,model_relative_to_both_db']
    for r in table: lines.append(','.join(map(str,[r['condition'],r['measured_relative_db'],r['model_relative_db']])))
    (OUT/'comparison.csv').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'output_dir':str(OUT),'table':table,'alignment':alignment},ensure_ascii=False,indent=2))
if __name__=='__main__': main()

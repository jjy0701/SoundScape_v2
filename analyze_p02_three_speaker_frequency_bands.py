from pathlib import Path
import json, math, hashlib, importlib.util
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FS = 48000
BANDS = [(250, 500), (500, 1000), (1000, 2000), (2000, 4000)]
BASE = ROOT / 'analyze_p02_three_speaker_conditions_order32.py'
PREVIOUS = ROOT / 'results' / '20261008_P02_three_speaker_conditions_B_order32_20261008_185648' / 'metrics.json'

spec = importlib.util.spec_from_file_location('base_p02', BASE)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

def db_level(y, sos, windows):
    z = sosfiltfilt(sos, y)
    vals = [10 * math.log10(float(np.mean(z[a:b] ** 2))) for a, b in windows]
    return {'per_repeat_level_db': vals,
            'pooled_level_db': float(10 * math.log10(np.mean(10 ** (np.asarray(vals) / 10)))),
            'repeat_spread_db': float(np.ptp(vals)),
            'window_samples': [[int(a), int(b)] for a, b in windows]}

def main():
    old = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    stamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'20261009_P02_three_speaker_frequency_bands_B_order32_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT/'configs'/'classroom'/'classroom_config(07_강의실측정조건).json').read_text(encoding='utf-8-sig'))
    st = json.loads((ROOT/'configs'/'simulation'/'simulation_settings(08_음향가정).json').read_text(encoding='utf-8-sig'))
    ref = base.read_wav(base.REFERENCE, allow_identical_stereo=True)
    w, l = cfg['room']['width_m'], cfg['room']['length_m']
    src = {}
    for s in cfg['sources']:
        x = s['side_wall_distance_m'] if s['side_wall'] == 'left' else w - s['side_wall_distance_m']
        src[s['id']] = {'position_m':[x, s['front_wall_distance_m'], s['height_m']], 'gain':st['sources'][s['id']]['relative_gain']}
    receiver = next(r for r in cfg['receivers'] if r['id'] == 'P02')['position_m']
    room = base.pra.ShoeBox([w, l, 2.67], fs=FS, materials=base.pra.make_materials(west=.2,east=.2,south=.2,north=.2,floor=.2,ceiling=.2), max_order=32, ray_tracing=False, air_absorption=False, use_rand_ism=False, sigma2_awgn=None)
    room.set_sound_speed(st['speed_of_sound_m_s'])
    for sid in ('S_L','S_R'): room.add_source(src[sid]['position_m'])
    room.add_microphone_array(np.asarray(receiver, dtype=float)[:, None]); room.compute_rir()
    conditions = {'왼쪽 한쪽만':(1,0), '오른쪽 스피커':(0,1), '두 스피커':(1,1)}
    model_waves = {}
    for name, mult in conditions.items():
        parts = [fftconvolve(ref, np.asarray(room.rir[0][i], float) * src[sid]['gain'] * mult[i], mode='full') for i, sid in enumerate(('S_L','S_R'))]
        total = np.zeros(max(map(len, parts)))
        for p in parts: total[:len(p)] += p
        model_waves[name] = total
    bands = {}
    for lo, hi in BANDS:
        key = f'{lo}-{hi} Hz'; sos = butter(4, [lo, hi], btype='bandpass', fs=FS, output='sos')
        measured, model = {}, {}
        for name, path in base.RECORDINGS.items():
            old_key = name
            wins = [tuple(x) for x in old['recordings'][old_key]['analysis_windows_s']]
            wins = [(round(a*FS), round(b*FS)) for a,b in wins]
            measured[name] = db_level(base.read_wav(path), sos, wins)
        model_windows = [(5*FS+FS, 15*FS-FS), (20*FS+FS, 30*FS-FS), (35*FS+FS, 45*FS-FS)]
        for name, y in model_waves.items(): model[name] = db_level(y, sos, model_windows)
        anchor = '두 스피커'; ml, mr = measured['왼쪽 한쪽만'], measured['오른쪽 스피커']; bl, br = model['왼쪽 한쪽만'], model['오른쪽 스피커']
        bands[key] = {'measured':measured, 'model':model,
                      'left_minus_right_db': {'measured': ml['pooled_level_db']-mr['pooled_level_db'], 'model': bl['pooled_level_db']-br['pooled_level_db']},
                      'relative_to_both_db': {'measured': {n:d['pooled_level_db']-measured[anchor]['pooled_level_db'] for n,d in measured.items()}, 'model': {n:d['pooled_level_db']-model[anchor]['pooled_level_db'] for n,d in model.items()}},
                      'measured_left_minus_right_minus_model_db': (ml['pooled_level_db']-mr['pooled_level_db'])-(bl['pooled_level_db']-br['pooled_level_db'])}
        rows=[]
        for n in conditions:
            rows.append([key,n, bands[key]['left_minus_right_db']['measured'], bands[key]['left_minus_right_db']['model'], bands[key]['relative_to_both_db']['measured'][n], bands[key]['relative_to_both_db']['model'][n]])
        with (out/'frequency_comparison.csv').open('a', encoding='utf-8-sig') as f:
            if f.tell()==0: f.write('band,condition,measured_L_minus_R_db,model_L_minus_R_db,measured_relative_to_both_db,model_relative_to_both_db\n')
            for row in rows: f.write(','.join(map(str,row))+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    for ax, (key, d) in zip(axes.flat, bands.items()):
        names=list(conditions); x=np.arange(3); width=.36
        ax.bar(x-width/2,[d['relative_to_both_db']['measured'][n] for n in names],width,label='Measured')
        ax.bar(x+width/2,[d['relative_to_both_db']['model'][n] for n in names],width,label='Model')
        ax.set_title(key); ax.set_xticks(x,names,rotation=18); ax.set_ylabel('relative level (dB)'); ax.grid(axis='y',alpha=.25)
    axes.flat[0].legend(frameon=False); fig.suptitle('P02 frequency-band comparison: relative to both speakers'); fig.tight_layout(); fig.savefig(out/'relative_levels_by_band.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,5)); x=np.arange(len(BANDS)); mw=[bands[k]['left_minus_right_db']['measured'] for k in bands]; bw=[bands[k]['left_minus_right_db']['model'] for k in bands]
    ax.bar(x-.18,mw,.36,label='Measured L−R'); ax.bar(x+.18,bw,.36,label='Model L−R'); ax.axhline(0,color='k',lw=.7); ax.set_xticks(x,list(bands)); ax.set_ylabel('dB'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(out/'left_right_difference_by_band.png',dpi=180); plt.close(fig)
    result={'analysis':'P02 three-speaker frequency-band comparison; model B order 32','bands_hz':BANDS,'alignment_source':str(PREVIOUS),'central_8s_x_3':True,'no_file_normalization':True,'no_physical_parameter_correction':True,'rir_reused':False,'rir_note':'No matching saved P02 B RIR was present; recomputed with unchanged B parameters.','bands':bands,'source_mapping':old['source_mapping'],'model_geometry':old['model_geometry']}
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output_dir':str(out),'left_minus_right_minus_model_db':{k:v['measured_left_minus_right_minus_model_db'] for k,v in bands.items()}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()

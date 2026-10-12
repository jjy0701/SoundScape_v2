"""Three-position pilot comparison of measured and simulated relative levels.

Uses the unchanged stage-08 model and already detected stage-06 recording windows.
Only P02 is the common level reference. No material/gain/delay fitting is performed.
Input recordings and the source test WAV are never modified.
"""
from pathlib import Path
import csv
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import wave
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt, fftconvolve
import pyroomacoustics as pra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IDS = ('P01', 'P02', 'P03')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_recording(path, fs, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError(f'{path.name}: input hash differs from recording QC; rerun QC for the new file.')
    with wave.open(str(path), 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (fs, 1, 2, 'NONE'):
            raise ValueError(f'{path.name}: expected mono 48 kHz 16-bit PCM.')
        n = w.getnframes()
        raw = w.readframes(n)
    if len(raw) != n*2:
        raise ValueError(f'{path.name}: incomplete PCM frames.')
    return np.frombuffer(raw, dtype='<i2').astype(np.float64)/32768.0


def segment_levels(x, fs, windows):
    energies, counts = [], []
    for start, end in windows:
        a,b = round(start*fs),round(end*fs)
        if not 0 <= a < b <= len(x):
            raise ValueError('Analysis window lies outside waveform.')
        chunk = x[a:b]
        energies.append(float(np.dot(chunk, chunk)))
        counts.append(len(chunk))
    powers = np.asarray(energies)/counts
    if not np.all(np.isfinite(powers)) or np.any(powers<=0):
        raise ValueError('Nonpositive or nonfinite analysis power.')
    return {'per_repeat_level_db': (10*np.log10(powers)).tolist(),
            'pooled_mean_square': float(sum(energies)/sum(counts)),
            'pooled_level_db': float(10*np.log10(sum(energies)/sum(counts))),
            'repeat_spread_db': float(np.ptp(10*np.log10(powers))),
            'windows_s': windows}


def relative_table(rows, band):
    anchor = next(r for r in rows if r['id']=='P02')[band]
    data=[]
    for r in rows:
        b=r[band]
        measured=b['measured']['pooled_level_db']-anchor['measured']['pooled_level_db']
        model=b['model_main']['pooled_level_db']-anchor['model_main']['pooled_level_db']
        lower=b['model_lower']['pooled_level_db']-anchor['model_lower']['pooled_level_db']
        data.append({'id':r['id'],'measured_relative_db':measured,'model_relative_db':model,
                     'model_minus_measured_db':model-measured,
                     'lower_order_relative_db':lower,'relative_order_change_db':model-lower})
    e=np.array([r['model_minus_measured_db'] for r in data if r['id']!='P02'])
    return {'anchor':'P02','points':data,'non_anchor_count':len(e),
            'non_anchor_mae_db':float(np.mean(np.abs(e))),
            'non_anchor_rmse_db':float(np.sqrt(np.mean(e*e))),
            'scope':'Two relative contrasts sharing P02. Descriptive pilot; not absolute SPL error, independent validation or AI test performance.'}


def main():
    base=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--recordings-dir',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=base/'support/classroom_config(07_강의실측정조건).json')
    ap.add_argument('--settings',type=Path,default=base/'support/simulation_settings(08_음향가정).json')
    ap.add_argument('--stage08-script',type=Path,default=base/'support/simulate_classroom(08_RIR과파형계산).py')
    ap.add_argument('--measurement-metrics',type=Path,default=base/'support/measurement_inputs(09_녹음구간과조건).json')
    ap.add_argument('--output-dir',type=Path,default=base/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    args=ap.parse_args()
    if pra.__version__!='0.10.1':
        raise ValueError('This comparison requires pyroomacoustics 0.10.1 for reproduction.')
    module=importlib.util.spec_from_file_location('stage08',args.stage08_script)
    stage08=importlib.util.module_from_spec(module)
    module.loader.exec_module(stage08)
    config,settings=read_json(args.config),read_json(args.settings)
    qc=read_json(args.measurement_metrics)
    if qc.get('measurement_conditions_confirmation',{}).get('only_position_changed_user_reported') is not True:
        raise ValueError('This pilot requires the recorded user confirmation of unchanged settings.')
    fs=settings['sample_rate_hz']
    x,reference=stage08.read_reference(args.reference,fs,np)
    if not reference['matches_original_06_reference']:
        raise ValueError('Playback reference differs from the original stage-06 WAV.')
    blocks=stage08.active_blocks(x,fs,np)
    model_windows=[[(a+fs)/fs,(b-fs)/fs] for a,b in blocks]
    if len(model_windows)!=3 or any(abs(b-a-8)>1e-9 for a,b in model_windows):
        raise ValueError('Expected three 8-second model core windows.')
    bands=[('band_250_500_hz',[250,500]),('band_500_1000_hz',[500,1000]),
           ('band_1000_2000_hz',[1000,2000]),('band_2000_4000_hz',[2000,4000]),
           ('band_250_4000_hz',[250,4000])]
    filters={name:butter(4,edges,btype='bandpass',fs=fs,output='sos') for name,edges in bands}
    qcs={r['measurement_id']:r for r in qc['recordings']}
    arrays={'sample_rate_hz':np.int64(fs),'receiver_ids':np.array(IDS),
            'filter_delay_samples':np.int64(pra.constants.get('frac_delay_length')//2)}
    for name,sos in filters.items(): arrays[f'{name}_sos']=sos
    rows=[]
    print('(09) 실측·시뮬레이션 상대 레벨 예비 비교',flush=True)
    for sid in IDS:
        params=dict(settings,receiver_id=sid)
        scene=stage08.resolve_inputs(config,params)
        entry=qcs[sid]
        recording=read_recording(args.recordings_dir/entry['file_name'],fs,entry['sha256'])
        candidates=entry['reference_comparison']['candidates']
        windows=[[c['core_start_s'],c['core_end_s']] for c in candidates]
        if len(windows)!=3 or any(abs(b-a-8)>1/fs for a,b in windows):
            raise ValueError(f'{sid}: expected three complete 8 s measured cores.')
        direct=stage08.build_room(scene,0,pra,np)
        timing=[]
        for src,h in zip(scene['sources'],direct.rir[0]):
            distance=math.dist(src['position_m'],scene['receiver']['position_m'])
            error=int(np.argmax(np.abs(h)))-int(arrays['filter_delay_samples'])-distance/scene['speed']*fs
            timing.append({'source':src['id'],'distance_m':distance,'error_samples':error})
            if abs(error)>1:
                raise RuntimeError('Direct arrival verification failed.')
        print(f'{sid}: order {scene["order"]}/{scene["comparison_order"]}',flush=True)
        room=stage08.build_room(scene,scene['order'],pra,np,x)
        lower_room=stage08.build_room(scene,scene['comparison_order'],pra,np)
        h=stage08.effective_rirs(room,scene,np)
        hl=stage08.effective_rirs(lower_room,scene,np)
        individual,total=stage08.synthesize(x,h,fftconvolve,np)
        _,lower=stage08.synthesize(x,hl,fftconvolve,np)
        via_h=fftconvolve(x,h.sum(axis=0))
        sum_error=float(np.max(np.abs(total-via_h)))
        if not np.allclose(total,via_h,atol=1e-10,rtol=1e-9):
            raise RuntimeError('Waveform summation check failed.')
        premix=room.simulate(return_premix=True)
        premix_errors=[]
        for i,src in enumerate(scene['sources']):
            expected=np.pad(premix[i,0],(src['extra_delay_samples'],0))
            a,b=stage08.padded_rows([individual[i],expected],np)
            premix_errors.append(float(np.max(np.abs(a-b))))
            if not np.allclose(a,b,atol=1e-10,rtol=1e-9):
                raise RuntimeError('PRA simulate vs explicit convolution failed.')
        del premix
        row={'id':sid,'position_m':scene['receiver']['position_m'],
             'position_basis':'user-reported maintained center/height, front-wall position; not a new precision survey',
             'recording_filename':entry['file_name'],'recording_sha256':entry['sha256'],
             'resolved_scene':scene,'timing_checks':timing,
             'sum_linearity_max_abs_error':sum_error,'premix_max_abs_errors':premix_errors}
        for band,ym,ys,yl in [('unfiltered',recording,total,lower)]+[
            (name,sosfiltfilt(sos,recording),sosfiltfilt(sos,total),sosfiltfilt(sos,lower))
            for name,sos in filters.items()]:
            row[band]={'measured':segment_levels(ym,fs,windows),
                       'model_main':segment_levels(ys,fs,model_windows),
                       'model_lower':segment_levels(yl,fs,model_windows)}
            row[band]['model_order_change_db']=row[band]['model_main']['pooled_level_db']-row[band]['model_lower']['pooled_level_db']
        arrays[f'{sid}_rir_main']=h
        arrays[f'{sid}_rir_lower']=hl
        rows.append(row)
        print(f'  measured band level={row["band_250_4000_hz"]["measured"]["pooled_level_db"]:.4f} dBFS',flush=True)
    result={'version':'1.0','created_at':datetime.now(timezone.utc).isoformat(),
            'purpose':'Unfitted stage-08 model vs three recordings; relative level pilot',
            'analysis_bands_hz':{name:edges for name,edges in bands},
            'filter_method':'scipy butter(4, [low, high], bandpass, output=sos), sosfiltfilt on whole waveform, crop afterward; identical operation for measured/model; repeated windows pooled by mean square before 10log10',
            'level_reference':'P02 independently subtracted once from each 3-position level vector; no per-position fitting',
            'model_level_unit':'10log10(mean-square) in arbitrary model amplitude units; not dBFS/SPL',
            'measured_level_unit':'dBFS of digital PCM; includes background, uncalibrated',
            'reference':reference,'assumptions_snapshot':settings,'config_snapshot':config,
            'measurement_conditions_confirmation':qc['measurement_conditions_confirmation'],
            'agc_status':'unknown; unchanged app setting is not proof of fixed gain',
            'room_geometry_status':'provisional; source acoustic centers/directivity and furniture not modeled',
            'fitted_acoustic_parameters':False,'calibrated_absolute_spl':False,'ai_evaluated':False,
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pyroomacoustics':pra.__version__},
            'input_hashes':{k:digest(getattr(args,k)) for k in ['config','settings','stage08_script','measurement_metrics']},
            'script_sha256':digest(__file__),'recordings':rows,
            'primary_comparison':relative_table(rows,'band_250_4000_hz'),
            'band_comparisons':{name:relative_table(rows,name) for name,_ in bands},
            'unfiltered_secondary_comparison':relative_table(rows,'unfiltered'),
            'convergence_claim':False,
            'notes':['No sample-by-sample waveform error is reported; recording onset and clocks have not been synchronized.',
                     'P02 is the common reference, so its zero error is structural and is excluded from error aggregation.',
                     'Repeated bursts within one recording do not provide independent re-placement or session uncertainty.',
                     'Keep all parameters fixed for this baseline. If tuned later using these measurements, obtain fresh evaluation measurements.']}
    # The unfiltered P02 order change should reproduce the stage-08 log.
    p02=next(r for r in rows if r['id']=='P02')
    if abs(p02['unfiltered']['model_order_change_db']-.160963887)>1e-3:
        raise RuntimeError('P02 does not reproduce stage-08 order comparison; check inputs before using results.')
    result['stage08_P02_reproduction_passed']=True
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'metrics(01_실측비교수치).json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'rirs(03_세지점원시RIR).npz',**arrays)
    with (out/'band_comparison(02_주파수대별수치).csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['analysis_band','point','measured_relative_db','simulation_relative_db','simulation_minus_measurement_db'])
        for name,_ in bands:
            for p in result['band_comparisons'][name]['points']:
                w.writerow([name,p['id'],p['measured_relative_db'],p['model_relative_db'],p['model_minus_measured_db']])
    with (out/'summary(03_주파수대별해석).md').open('w',encoding='utf-8') as f:
        f.write('# 주파수 대역별 비교 요약\n\n각 대역에서 P02를 0 dB로 둔 P01·P03 상대 레벨 비교입니다. 오차 부호는 시뮬레이션 − 실측이며, MAE/RMSE는 P02를 제외했습니다. 절대 SPL 정확도나 특정 원인 확정으로 해석하지 않습니다.\n\n')
        for name,edges in bands:
            c=result['band_comparisons'][name]; vals=[p for p in c['points'] if p['id']!='P02']
            f.write(f"- **{edges[0]:,}–{edges[1]:,} Hz**: P01 {vals[0]['model_minus_measured_db']:+.4f} dB, P03 {vals[1]['model_minus_measured_db']:+.4f} dB; MAE {c['non_anchor_mae_db']:.4f} dB, RMSE {c['non_anchor_rmse_db']:.4f} dB.\n")
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),gridspec_kw={'width_ratios':[1.55,1]})
    fig.subplots_adjust(top=.76,bottom=.20,left=.085,right=.98,wspace=.34)
    points=result['primary_comparison']['points']
    measured=np.array([p['measured_relative_db'] for p in points])
    model=np.array([p['model_relative_db'] for p in points])
    t=np.arange(3)
    for bars in [axs[0].bar(t-.18,measured,.35,label='Measured',color='#00857d'),axs[0].bar(t+.18,model,.35,label='Simulation',color='#cc7635')]:
        axs[0].bar_label(bars,fmt='%+.2f',padding=4,fontsize=10)
    axs[0].set(xticks=t,xticklabels=['P01\n2.5 m','P02\n5.0 m','P03\n7.5 m'],ylabel='Level relative to P02 (dB)',title='Spatial level differences')
    axs[0].legend(frameon=False,loc='upper right')
    axs[0].set_ylim(min(measured.min(),model.min())-1,max(measured.max(),model.max())+1.3)
    errors=(model-measured)[[0,2]]
    bars=axs[1].bar([0,1],errors,.5,color='#586d83')
    axs[1].bar_label(bars,fmt='%+.2f dB',padding=4,fontsize=10)
    axs[1].set(xticks=[0,1],xticklabels=['P01','P03'],ylabel='Simulation minus measurement (dB)',title='Two non-reference points')
    axs[1].set_ylim(min(0,errors.min())-1,max(0,errors.max())+1)
    for ax in axs:
        ax.axhline(0,color='#222222',lw=.8)
        ax.grid(axis='y',alpha=.16)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Classroom pilot | measurement vs fixed 3D model',fontsize=16,fontweight='bold',y=.96)
    fig.text(.5,.87,'250–4,000 Hz | P02 reference | two coherent sources | absorption 0.2 | order 32',ha='center',fontsize=10)
    fig.text(.5,.05,'Settings held constant (user report). Uncalibrated phone; no material fitting; not an AI accuracy test.',ha='center',fontsize=9,color='#555555')
    fig.savefig(out/'comparison(02_실측과시뮬레이션).png',dpi=170)
    fig2,ax2=plt.subplots(figsize=(10.5,5.5)); width=.18; t=np.arange(2)
    for j,(name,edges) in enumerate(bands[:-1]):
        e=np.array([p['model_minus_measured_db'] for p in result['band_comparisons'][name]['points'] if p['id']!='P02'])
        bars=ax2.bar(t+(j-1.5)*width,e,width,label=f'{edges[0]:,}–{edges[1]:,} Hz'); ax2.bar_label(bars,fmt='%+.2f',fontsize=8,padding=2)
    ax2.axhline(0,color='k',lw=.8); ax2.set_xticks(t,['P01','P03']); ax2.set_ylabel('Simulation − measurement (dB)'); ax2.set_title('Frequency-band relative differences (P02 excluded)'); ax2.legend(frameon=False,ncol=2)
    fig2.tight_layout(); fig2.savefig(out/'comparison(01_주파수대별그래프).png',dpi=170); plt.close(fig2)
    plt.close(fig)
    print(json.dumps(result['primary_comparison'],ensure_ascii=False,indent=2),flush=True)
    print(f'Results: {out}',flush=True)


if __name__=='__main__':
    main()

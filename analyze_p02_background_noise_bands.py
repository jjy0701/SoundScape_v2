from pathlib import Path
import json, math, importlib.util
import numpy as np
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; FS=48000
BANDS=[(250,500),(500,1000),(1000,2000),(2000,4000)]
PREV=ROOT/'results'/'20261009_P02_three_speaker_frequency_bands_B_order32_20261009_144154'/'metrics.json'
spec=importlib.util.spec_from_file_location('base',ROOT/'analyze_p02_three_speaker_conditions_order32.py'); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

def level(y,sos,a,b):
 z=sosfiltfilt(sos,y[a:b]); return 10*math.log10(float(np.mean(z*z)))

def main():
 old=json.loads(PREV.read_text(encoding='utf-8')); stamp=__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S'); out=ROOT/'results'/f'20261009_P02_background_noise_bands_{stamp}'; out.mkdir()
 # Selected by visual inspection of the first 16 s waveform and short-term level:
 # exclude initial handling/talk, isolated transient events, and the playback onset.
 bg={'왼쪽 한쪽만':(10.0,12.0),'오른쪽 스피커':(8.0,10.0),'두 스피커':(8.0,10.0)}
 plot=plt.subplots(3,1,figsize=(14,9)); fig,axes=plot
 all_data={}
 for ax,(name,path) in zip(axes,base.RECORDINGS.items()):
  y=base.read_wav(path); t=np.arange(min(len(y),16*FS))/FS; ax.plot(t,y[:len(t)],lw=.25); ax.axvspan(*bg[name],color='#087f8c',alpha=.25); ax.set_xlim(0,16); ax.set_title(name); ax.grid(alpha=.2)
  windows=[tuple(x) for x in old['bands']['250-500 Hz']['measured'][name]['window_samples']]
  data={}
  for lo,hi in BANDS:
   sos=butter(4,[lo,hi],btype='bandpass',fs=FS,output='sos'); key=f'{lo}-{hi} Hz'
   b0,b1=round(bg[name][0]*FS),round(bg[name][1]*FS)
   bglev=level(y,sos,b0,b1); rep=[level(y,sos,a,b) for a,b in windows]; play=float(10*math.log10(np.mean(10**(np.asarray(rep)/10))))
   data[key]={'background_noise_level_db':bglev,'playback_central_level_db':play,'difference_playback_minus_background_db':play-bglev,'background_window_s':list(bg[name]),'playback_windows_s':[[a/FS,b/FS] for a,b in windows],'playback_per_repeat_level_db':rep,'playback_repeat_spread_db':float(np.ptp(rep))}
  all_data[name]={'background_window_s':list(bg[name]),'bands':data}
 fig.tight_layout(); fig.savefig(out/'background_waveforms_and_selected_windows.png',dpi=180); plt.close(fig)
 rows=[]
 for name,d in all_data.items():
  for band,v in d['bands'].items(): rows.append([name,band,*[v[k] for k in ('background_noise_level_db','playback_central_level_db','difference_playback_minus_background_db','playback_repeat_spread_db')]])
 with (out/'background_vs_playback.csv').open('w',encoding='utf-8-sig') as f:
  f.write('condition,band,background_noise_db,playback_central_db,playback_minus_background_db,playback_repeat_spread_db\n'); [f.write(','.join(map(str,r))+'\n') for r in rows]
 result={'analysis':'P02 pre-playback background noise vs central playback bands','bands_hz':BANDS,'filter':'4th-order Butterworth bandpass + sosfiltfilt','alignment_source':str(PREV),'no_noise_removal':True,'no_volume_normalization':True,'no_model_correction':True,'background_selection_method':'Waveform inspection; excluded handling/talk transients and playback onset; selected quiet pre-playback intervals.','background_windows_s':bg,'playback_windows_preserved_from_existing_analysis':True,'data':all_data}
 (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(out)
if __name__=='__main__': main()

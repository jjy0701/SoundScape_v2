from pathlib import Path
import csv, json, math
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'results' / 'today_B_comparison_20261012_20261012_185129'
BANDS = ['250-500', '500-1000', '1000-2000', '2000-4000']


def read_rows():
    with (INPUT / 'comparison.csv').open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def main():
    rows = read_rows()
    lookup = {(r['point'], r['band_hz'], r['source']): r for r in rows}
    # P02-only calibration: only the four requested bands and single-speaker rows.
    corrections = []
    gain_by_band = {}
    for band in BANDS:
        meas_l = f(lookup['P02', band, 'measured'], 'left_minus_right_db')
        model_d = f(lookup['P02', band, 'model'], 'left_minus_right_db')
        delta = meas_l - model_d
        gain = 10 ** (-delta / 20.0)
        gain_by_band[band] = gain
        corrections.append({
            'band_hz': band,
            'P02_D_meas_db': meas_l,
            'P02_D_model_db': model_d,
            'delta_db': delta,
            'left_amplitude_gain': 1.0,
            'right_amplitude_gain': gain,
            'right_gain_db': 20 * math.log10(gain),
            'calibration_role': 'P02 empirical bandwise left/right correction; not speaker or amplifier gain estimate',
        })

    out = ROOT / 'results' / f'p02_band_correction_p01_p03_20261012_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'corrections.csv').open('w', newline='', encoding='utf-8-sig') as f0:
        w = csv.DictWriter(f0, fieldnames=corrections[0].keys()); w.writeheader(); w.writerows(corrections)

    comparison = []
    for point in ('P01', 'P02', 'P03'):
        for band in BANDS:
            measured = f(lookup[point, band, 'measured'], 'left_minus_right_db')
            model_before = f(lookup[point, band, 'model'], 'left_minus_right_db')
            gain = gain_by_band[band]
            model_after = model_before - 20 * math.log10(gain)
            err_before = model_before - measured
            err_after = model_after - measured
            comparison.append({
                'point': point, 'band_hz': band, 'measured_L_minus_R_db': measured,
                'model_before_L_minus_R_db': model_before,
                'model_after_L_minus_R_db': model_after,
                'signed_error_before_db': err_before,
                'signed_error_after_db': err_after,
                'absolute_error_before_db': abs(err_before),
                'absolute_error_after_db': abs(err_after),
                'absolute_error_change_db': abs(err_after) - abs(err_before),
                'status': 'improved' if abs(err_after) < abs(err_before) else ('worsened' if abs(err_after) > abs(err_before) else 'unchanged'),
                'evaluation_role': 'P02 calibration check' if point == 'P02' else 'other-location application evaluation',
            })
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f0:
        w = csv.DictWriter(f0, fieldnames=comparison[0].keys()); w.writeheader(); w.writerows(comparison)

    def metrics_for(sub):
        b = [x['absolute_error_before_db'] for x in sub]
        a = [x['absolute_error_after_db'] for x in sub]
        sb = [x['signed_error_before_db'] for x in sub]
        sa = [x['signed_error_after_db'] for x in sub]
        return {
            'n_bands': len(sub),
            'before': {'MAE_db': sum(b)/len(b), 'RMSE_db': math.sqrt(sum(x*x for x in sb)/len(sb))},
            'after': {'MAE_db': sum(a)/len(a), 'RMSE_db': math.sqrt(sum(x*x for x in sa)/len(sa))},
            'MAE_change_db': sum(a)/len(a) - sum(b)/len(b),
            'RMSE_change_db': math.sqrt(sum(x*x for x in sa)/len(sa)) - math.sqrt(sum(x*x for x in sb)/len(sb)),
            'improved_items': [x['band_hz'] for x in sub if x['status'] == 'improved'],
            'worsened_items': [x['band_hz'] for x in sub if x['status'] == 'worsened'],
            'unchanged_items': [x['band_hz'] for x in sub if x['status'] == 'unchanged'],
        }

    eval_rows = [x for x in comparison if x['point'] in ('P01', 'P03')]
    metrics = {
        'input_directory': str(INPUT.relative_to(ROOT)),
        'calibration_point': 'P02',
        'calibration_uses': 'P02 left/right single-speaker recordings only',
        'excluded_from_calibration': ['both-sided playback', 'previous dates', 'P01/P03 data'],
        'formula': {'D_meas': 'measured left - measured right', 'D_model': 'uncorrected B model left - right', 'delta': 'D_meas - D_model', 'right_amplitude_gain': '10**(-delta/20)', 'corrected_model_difference': 'D_model - 20*log10(gain)'},
        'corrections': corrections,
        'P02_calibration_check': metrics_for([x for x in comparison if x['point'] == 'P02']),
        'other_location_application_evaluation': {p: metrics_for([x for x in eval_rows if x['point'] == p]) for p in ('P01', 'P03')},
        'all_8_items_P01_P03': metrics_for(eval_rows),
        'limitations': ['P01/P03 are previously observed points; this is other-location application evaluation, not a fully independent blind validation.', 'Separate single-speaker recordings may have condition changes.', 'No polarity, delay, absorption, or coordinates were changed.', 'The gains are empirical P02-based left/right corrections, not speaker-specific characteristics or amplifier gain estimates.', 'P02 agreement is calibration confirmation and excluded from validation performance.'],
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    # Plot signed error and absolute error before/after for all requested evaluation rows.
    for point in ('P01', 'P03'):
        sub = [x for x in comparison if x['point'] == point]
        x = range(len(BANDS)); width = .36
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i-width/2 for i in x], [z['signed_error_before_db'] for z in sub], width, label='model before')
        ax.bar([i+width/2 for i in x], [z['signed_error_after_db'] for z in sub], width, label='model after')
        ax.axhline(0, color='black', lw=.8); ax.set_xticks(list(x), BANDS); ax.set_ylabel('signed error (model - measured), dB'); ax.set_title(f'{point}: signed error before/after P02 correction'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / f'{point}_signed_error.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [f"{x['point']}\n{x['band_hz']}" for x in eval_rows]; x = range(len(labels)); width = .36
    ax.bar([i-width/2 for i in x], [z['absolute_error_before_db'] for z in eval_rows], width, label='before')
    ax.bar([i+width/2 for i in x], [z['absolute_error_after_db'] for z in eval_rows], width, label='after')
    ax.set_xticks(list(x), labels); ax.set_ylabel('absolute error (dB)'); ax.set_title('P01/P03 other-location application: absolute error'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / 'absolute_error_all_8.png', dpi=160); plt.close(fig)

    readme = f'''# P02 기반 대역별 좌우 보정의 P01·P03 적용 효과

- 입력: `{INPUT.relative_to(ROOT)}` (2026-10-12 녹음 결과)
- P02의 왼쪽·오른쪽 단독 녹음만으로 250–500, 500–1000, 1000–2000, 2000–4000 Hz 보정값을 계산했다.
- `D_meas = 실측 L−R`, `D_model = 보정 전 B 모델 L−R`, `delta = D_meas−D_model`, `right_gain = 10**(-delta/20)`.
- 왼쪽 gain은 1로 고정했고, P01·P03에는 위 gain을 그대로 적용했다. RIR은 재계산하지 않았다.
- P02는 보정 계산 확인이며 검증 성능에서 제외했다. 양쪽 재생, 이전 날짜 자료, P01/P03 재맞춤은 사용하지 않았다.

## 결과

`comparison.csv`에 지점·대역별 실측 L−R, 모델 보정 전/후 L−R, signed error, 절대오차와 개선/악화를 저장했다. `metrics.json`에 P01·P03별 네 대역 MAE/RMSE 및 총 8개 항목의 MAE/RMSE를 저장했다.

그래프: `P01_signed_error.png`, `P03_signed_error.png`, `absolute_error_all_8.png`.

## 해석 범위와 한계

이 값은 **P02 기반 경험적 좌우 보정값**이며 스피커 고유 특성이나 실제 앰프 gain 추정치가 아니다. P01·P03은 이전에 관찰한 지점이므로 **다른 위치 적용 평가**로 표현하며 완전히 독립적인 블라인드 검증이라고 부르지 않는다. 별도 단독 녹음 사이의 조건 변화 가능성이 남아 있다. 극성·지연·흡음률·좌표는 변경하지 않았다.
'''
    (out / 'README.md').write_text(readme, encoding='utf-8')
    print(out)
    print(json.dumps(metrics['corrections'], ensure_ascii=False, indent=2))
    print(json.dumps({'P01': metrics['other_location_application_evaluation']['P01'], 'P03': metrics['other_location_application_evaluation']['P03'], 'all_8': metrics['all_8_items_P01_P03']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

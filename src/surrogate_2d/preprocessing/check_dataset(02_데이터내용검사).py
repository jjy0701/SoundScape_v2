"""(02) 기존 dataset/metadata.csv + npy 내용 검사. 원본은 수정하지 않습니다.
실행: python "src/surrogate_2d/preprocessing/check_dataset(02_데이터내용검사).py"
선택: --dataset "다른/dataset" --near-distance 0.1
필수 외부 패키지: numpy. pyroomacoustics는 필요하지 않습니다.
격자 규약: meshgrid(x,y), 배열[y,x], 각 축 벽에서 0.5m, 20개 점.
PASS는 저장 데이터의 일관성 검사이며 음향 정확도 검증이 아닙니다.
"""
import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

FIELDS = ['room_x', 'room_y', 'speaker_x', 'speaker_y', 'obstacle_x',
          'obstacle_y', 'obstacle_w', 'obstacle_l', 'wall_alpha', 'obstacle_alpha']
TOL = 1e-5  # float32 정규화 오차 허용량(dB)


def canonical_id(value):
    value = str(value).strip()
    match = re.fullmatch(r'(?:sample_)?(\d+)', value)
    if not match:
        raise ValueError(f'지원하지 않는 sample_id: {value!r}')
    return f'sample_{int(match.group(1)):04d}'


def inspect_sample(row, dataset, near_distance):
    errors, warnings = [], []
    result = dict(sample_id=str(row.get('sample_id', '')), status='', errors='', warnings='')
    try:
        sid = canonical_id(row['sample_id'])
        result['sample_id'] = sid
        v = {key: float(row[key]) for key in FIELDS}
        if not all(np.isfinite(x) for x in v.values()):
            raise ValueError('metadata에 NaN/무한대가 있습니다')
        rx, ry = v['room_x'], v['room_y']
        sx, sy = v['speaker_x'], v['speaker_y']
        ox, oy, w, h = [v[k] for k in ['obstacle_x','obstacle_y','obstacle_w','obstacle_l']]
        if rx <= 1 or ry <= 1 or w <= 0 or h <= 0:
            raise ValueError('방/장애물 크기가 현재 격자 규약에 맞지 않습니다')
        if not (0 <= ox < ox+w <= rx and 0 <= oy < oy+h <= ry):
            errors.append('장애물이 방 밖으로 나갑니다')
        if not (0 < sx < rx and 0 < sy < ry):
            errors.append('음원이 방 내부에 있지 않습니다')
        if ox <= sx <= ox+w and oy <= sy <= oy+h:
            errors.append('음원이 장애물 내부/경계에 있습니다')
        if not all(0 <= v[k] <= 1 for k in ['wall_alpha','obstacle_alpha']):
            errors.append('흡음계수가 [0,1] 범위를 벗어납니다')
        X, Y = np.meshgrid(np.linspace(.5, rx-.5, 20), np.linspace(.5, ry-.5, 20))
        mask = (X >= ox) & (X <= ox+w) & (Y >= oy) & (Y <= oy+h)
        a = np.load(dataset / 'npy' / f'{sid}.npy', allow_pickle=False)
        if a.shape != (20,20) or not np.issubdtype(a.dtype, np.floating):
            raise ValueError(f'20x20 실수 부동소수 배열 필요: shape={a.shape}, dtype={a.dtype}')
        mismatch = int(np.count_nonzero(np.isnan(a) != mask))
        outside_bad = int(np.count_nonzero(~np.isfinite(a[~mask])))
        result.update(shape=str(a.shape), dtype=str(a.dtype),
                      expected_nan_count=int(mask.sum()), actual_nan_count=int(np.isnan(a).sum()),
                      nan_mask_mismatch=mismatch, outside_nonfinite=outside_bad,
                      inf_count=int(np.isinf(a).sum()))
        if mismatch:
            errors.append(f'기하 mask와 NaN 위치 불일치 {mismatch}개')
        if outside_bad:
            errors.append(f'장애물 밖 NaN/무한대 {outside_bad}개')
        if np.isinf(a).any():
            errors.append('무한대가 있습니다')
        valid = a[~mask]
        finite = valid[np.isfinite(valid)]
        if not finite.size:
            errors.append('유효한 유한 레벨이 없습니다')
        else:
            result.update(min_db=float(finite.min()), max_db=float(finite.max()),
                          median_db=float(np.median(finite)), std_db=float(finite.std()))
            if abs(float(finite.max())) > TOL:
                errors.append('유효 영역 최대값이 0 dB가 아닙니다')
            if np.any(finite > TOL):
                errors.append('양의 상대 레벨이 있습니다')
            if np.all(np.abs(finite) <= TOL):
                warnings.append('유효 레벨이 모두 약 0 dB: 원래 에너지/RIR 확인 필요')
        if (~mask).any():
            distance = np.hypot(X[~mask]-sx, Y[~mask]-sy)
            result['min_source_receiver_distance_m'] = float(distance.min())
            result['receivers_below_near_threshold'] = int((distance < near_distance).sum())
            if distance.min() == 0:
                errors.append('음원과 receiver 좌표가 일치합니다')
            elif distance.min() < near_distance:
                warnings.append(f'음원-receiver 거리 {distance.min():.4f}m: 근접 배치 검토')
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    result['status'] = 'FAIL' if errors else ('REVIEW' if warnings else 'PASS')
    result['errors'], result['warnings'] = ' | '.join(errors), ' | '.join(warnings)
    return result


def main():
    project_root = next(parent for parent in Path(__file__).resolve().parents
                        if (parent / "configs" / "classroom").is_dir()
                        and (parent / "data").is_dir()
                        and (parent / "results").is_dir())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=project_root/'data'/'surrogate_2d'/'dataset')
    parser.add_argument('--near-distance', type=float, default=.1,
                        help='검토용 거리(m). 물리적으로 검증된 허용 기준이 아님')
    parser.add_argument('--output-root', type=Path, default=project_root/'results'/'surrogate_2d'/'checks')
    args = parser.parse_args()
    if not np.isfinite(args.near_distance) or args.near_distance <= 0:
        parser.error('--near-distance는 유한한 양수여야 합니다')
    dataset = args.dataset.resolve()
    print('(02) 데이터내용검사 v1', flush=True)
    print(f'데이터 폴더: {dataset}', flush=True)
    try:
        with (dataset/'metadata.csv').open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            missing = set(['sample_id']+FIELDS)-set(reader.fieldnames or [])
            if missing:
                raise ValueError(f'누락된 metadata 컬럼: {sorted(missing)}')
            rows = list(reader)
    except Exception as exc:
        print(f'FAIL: metadata 읽기 실패: {exc}')
        return 1
    global_errors, global_notes = [], []
    if not rows:
        global_errors.append('metadata가 비어 있습니다')
    if len(rows) != 100:
        global_notes.append(f'현재 기대한 100개와 행 수가 다릅니다: {len(rows)}개')
    ids = []
    for row in rows:
        try:
            ids.append(canonical_id(row.get('sample_id','')))
        except ValueError:
            pass  # 개별 검사에서 FAIL로 표시
    duplicates = sorted(k for k,n in Counter(ids).items() if n > 1)
    if duplicates:
        global_errors.append(f'중복 ID: {duplicates}')
    npy_ids = {p.stem for p in (dataset/'npy').glob('*.npy')}
    if npy_ids != set(ids):
        global_errors.append(f'NPY 누락: {sorted(set(ids)-npy_ids)}; metadata 없는 NPY: {sorted(npy_ids-set(ids))}')
    results = [inspect_sample(row, dataset, args.near_distance) for row in rows]
    for r in results:
        print(f"[{r['status']}] {r['sample_id']} {r['errors']} {r['warnings']}", flush=True)
    counts = Counter(r['status'] for r in results)
    distances = [r['min_source_receiver_distance_m'] for r in results if 'min_source_receiver_distance_m' in r]
    ranges = {}
    for key in FIELDS:
        values = []
        for row in rows:
            try:
                value = float(row[key])
                if np.isfinite(value):
                    values.append(value)
            except (TypeError, ValueError):
                pass
        if values:
            ranges[key] = dict(min=min(values), max=max(values), unique_count=len(set(values)))
    outcome = 'FAIL' if global_errors or counts['FAIL'] else ('REVIEW' if counts['REVIEW'] or global_notes else 'PASS')
    summary = dict(outcome=outcome, metadata_rows=len(rows), npy_files=len(npy_ids),
                   sample_counts={key:counts[key] for key in ['PASS','REVIEW','FAIL']},
                   global_errors=global_errors, notes=global_notes, metadata_ranges=ranges,
                   min_source_receiver_distance_m=min(distances) if distances else None,
                   near_distance_review_threshold_m=args.near_distance, max_zero_tolerance_db=TOL,
                   grid=dict(shape=[20,20], wall_offset_m=.5, array_order='y,x'),
                   limitations=['정규화 전 에너지의 하한 적용/빈 RIR 여부는 상대 NPY만으로 판정 불가',
                                '물리적 정확성 및 반사 차수 수렴은 별도 검증 필요',
                                'PNG 내용과 좌표 방향의 시각 검증은 포함하지 않음',
                                '근접 거리 0.1m 기본값은 검토용이며 자동 삭제 기준이 아님'])
    out = args.output_root.resolve()/('dataset_check(02_데이터내용검사)_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    keys = list(dict.fromkeys(k for r in results for k in r)) or ['sample_id','status','errors','warnings']
    with (out/'samples(01_샘플별검사).csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    (out/'summary(02_검사요약).json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines = [f'전체 결과: {outcome}', f"샘플: PASS {counts['PASS']} / REVIEW {counts['REVIEW']} / FAIL {counts['FAIL']}"]
    lines += global_errors+global_notes
    if distances:
        lines.append(f'가장 가까운 음원-receiver 거리: {min(distances):.6f} m')
    lines += ['고정 metadata 변수: '+', '.join(k for k,v in ranges.items() if v['unique_count']==1),
              'REVIEW는 원인 확인이 필요하다는 뜻이며 데이터 삭제 지시가 아닙니다.',
              '원본 dataset은 수정하지 않았습니다.']
    (out/'report(03_읽어볼결과).txt').write_text('\n'.join(lines),encoding='utf-8-sig')
    print('\n'+'\n'.join(lines))
    print(f'결과 폴더: {out}')
    return 1 if outcome == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())

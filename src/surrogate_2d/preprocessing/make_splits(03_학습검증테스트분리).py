"""(03) 현재 100개 장면을 고정 분할. 근접 18개를 13/3/2로 배분합니다.
실행: python "src/surrogate_2d/preprocessing/make_splits(03_학습검증테스트분리).py"
data/surrogate_2d/dataset과 같은 preprocessing 폴더의 02번 검사 파일이 필요합니다.
원본을 이동/수정하지 않습니다. dataset/splits에 ID와 근접 표시를 저장합니다.
기존 분할과 입력이 달라지면 덮어쓰지 않고 중단합니다.
"""
import argparse
import csv
import hashlib
import io
import json
import runpy
import shutil
import tempfile
from pathlib import Path

SEED = 42
SIZES = {'train': 70, 'validation': 15, 'test': 15}
NEAR_SIZES = {'train': 13, 'validation': 3, 'test': 2}
NEAR_THRESHOLD = .1
CHECKER = 'check_dataset(02_데이터내용검사).py'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def split_records(records):
    """고정 seed와 ID의 해시로 정렬하여 CSV 행 순서에 의존하지 않습니다."""
    if len(records) != 100 or len({r['sample_id'] for r in records}) != 100:
        raise ValueError('현재 분할은 중복 없는 100개 ID를 대상으로 합니다')
    near = [r for r in records if r['near_source']]
    other = [r for r in records if not r['near_source']]
    if len(near) != 18:
        raise ValueError(f'근접 샘플이 이전 검사(18개)와 다릅니다: {len(near)}개. 분할 전에 데이터 확인 필요')
    def rank(r):
        return hashlib.sha256(f"{SEED}:{r['sample_id']}".encode()).hexdigest()
    near.sort(key=rank)
    other.sort(key=rank)
    splits, nptr, optr = {}, 0, 0
    for name, count in SIZES.items():
        n = NEAR_SIZES[name]
        selected = near[nptr:nptr+n]+other[optr:optr+count-n]
        nptr += n
        optr += count-n
        splits[name] = sorted(r['sample_id'] for r in selected)
    flat = [sid for ids in splits.values() for sid in ids]
    if len(flat) != 100 or len(set(flat)) != 100 or set(flat) != {r['sample_id'] for r in records}:
        raise RuntimeError('분할 중복/누락 발생')
    return splits


def main():
    project_root = next(parent for parent in Path(__file__).resolve().parents
                        if (parent / "configs" / "classroom").is_dir()
                        and (parent / "data").is_dir()
                        and (parent / "results").is_dir())
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=project_root/'data'/'surrogate_2d'/'dataset')
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    print('(03) 학습검증테스트분리 v1', flush=True)
    checker_path = base/CHECKER
    if not checker_path.is_file():
        raise ValueError(f'같은 폴더에 02번 검사 파일이 필요합니다: {CHECKER}')
    checker = runpy.run_path(str(checker_path))
    with (dataset/'metadata.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = set(['sample_id']+checker['FIELDS'])
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('metadata 필수 컬럼이 누락됐습니다')
        rows = list(reader)
    records, geometries, hashes = [], set(), {}
    for row in rows:
        result = checker['inspect_sample'](row, dataset, NEAR_THRESHOLD)
        if result['status'] == 'FAIL':
            raise ValueError(f"{result['sample_id']}: {result['errors']}")
        if '유효 레벨이 모두' in result['warnings']:
            raise ValueError(f"{result['sample_id']}: 전체 0 dB 샘플은 먼저 원인을 확인하세요")
        sid = result['sample_id']
        # 완전히 같은 입력 장면이 다른 ID로 중복된 경우 그룹 정책을 먼저 정합니다.
        geometry = tuple(float(row[k]) for k in checker['FIELDS'])
        if geometry in geometries:
            raise ValueError('동일한 입력 장면이 중복됐습니다. 장면 그룹 분할이 필요합니다')
        geometries.add(geometry)
        distance = result['min_source_receiver_distance_m']
        records.append(dict(sample_id=sid, near_source=distance < NEAR_THRESHOLD,
                            min_source_receiver_distance_m=distance))
        hashes[sid] = digest(dataset/'npy'/f'{sid}.npy')
    expected_ids = {r['sample_id'] for r in records}
    actual_ids = {p.stem for p in (dataset/'npy').glob('*.npy')}
    if expected_ids != actual_ids:
        raise ValueError('metadata와 NPY 파일 ID 집합이 다릅니다')
    splits = split_records(records)
    records.sort(key=lambda r:r['sample_id'])
    for record in records:
        record['split'] = next(name for name, ids in splits.items() if record['sample_id'] in ids)
    manifest = dict(version=1, seed=SEED, method='sha256(seed:sample_id) ranked within near/non-near strata',
                    near_threshold_m=NEAR_THRESHOLD, sizes=SIZES, near_sizes=NEAR_SIZES,
                    grid=dict(shape=[20,20], offset_m=.5, array_order='y,x'), splits=splits,
                    samples=records, metadata_sha256=digest(dataset/'metadata.csv'),
                    npy_sha256=hashes, checker_sha256=digest(checker_path),
                    script_sha256=digest(Path(__file__)),
                    notes=['장면 단위 분할; 같은 장면의 receiver는 분리하지 않음',
                           '정규화/표준화 통계와 학습은 train만 사용',
                           'validation은 모델 선택, test는 선택 완료 후 최종 평가',
                           '근접 test 2개로 모집단 성능을 단정하지 않음',
                           '근접 0.1m는 검토용 기준; 8차 반사의 수렴 검증은 별도 필요'])
    artifacts = {'split_manifest.json':json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n'}
    for name, ids in splits.items():
        artifacts[f'{name}_ids.txt'] = '\n'.join(ids)+'\n'
    buffer = io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=['sample_id','split','near_source','min_source_receiver_distance_m'])
    writer.writeheader(); writer.writerows(records)
    artifacts['assignments(01_분할목록).csv'] = '\ufeff'+buffer.getvalue()
    out=dataset/'splits'
    if out.exists():
        for name, content in artifacts.items():
            if not (out/name).is_file() or (out/name).read_bytes() != content.encode('utf-8'):
                raise ValueError('기존 splits의 입력/설정/파일이 다릅니다. 기존 분할을 보존하고 중단합니다')
        print('기존 분할과 일치: 동일한 분할을 유지합니다.')
    else:
        temp=Path(tempfile.mkdtemp(prefix='splits_pending_',dir=dataset))
        try:
            for name, content in artifacts.items():
                (temp/name).write_bytes(content.encode('utf-8'))
            temp.rename(out)
        finally:
            if temp.exists(): shutil.rmtree(temp)
        print('분할 저장 완료')
    for name in SIZES:
        print(f'{name}: 전체 {SIZES[name]} / 근접 {NEAR_SIZES[name]} / 비근접 {SIZES[name]-NEAR_SIZES[name]}')
    print('PASS: 그룹 간 ID 중복 없음 / 100개 모두 배정')
    print(f'저장 위치: {out}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'FAIL: {exc}')
        raise SystemExit(1)

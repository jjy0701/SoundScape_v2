import pandas as pd
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "configs" / "classroom").is_dir()
                    and (parent / "data").is_dir()
                    and (parent / "results").is_dir())

PNG_DIR = PROJECT_ROOT / "data" / "surrogate_2d" / "dataset" / "png"
NPY_DIR = PROJECT_ROOT / "data" / "surrogate_2d" / "dataset" / "npy"
METADATA = PROJECT_ROOT / "data" / "surrogate_2d" / "dataset" / "metadata.csv"


# =========================================================
# 1. 파일 목록
# =========================================================

png_files = sorted(PNG_DIR.glob("*.png"))
npy_files = sorted(NPY_DIR.glob("*.npy"))

print("=" * 60)
print("파일 개수 검사")
print("=" * 60)

print("PNG :", len(png_files))
print("NPY :", len(npy_files))


# =========================================================
# 2. PNG / NPY 이름 비교
# =========================================================

png_ids = {f.stem for f in png_files}
npy_ids = {f.stem for f in npy_files}

print("\nPNG에만 존재:")
print(sorted(png_ids - npy_ids))

print("\nNPY에만 존재:")
print(sorted(npy_ids - png_ids))


if png_ids == npy_ids:
    print("\n✅ PNG ↔ NPY 완전히 일치")
else:
    print("\n❌ PNG ↔ NPY 불일치")


# =========================================================
# 3. Metadata
# =========================================================

df = pd.read_csv(METADATA)

print("\n" + "=" * 60)
print("Metadata 검사")
print("=" * 60)

print("Metadata 행 수 :", len(df))

print("\n컬럼:")
print(df.columns.tolist())


# =========================================================
# 4. Metadata ID 확인
# =========================================================

print("\n앞쪽 데이터:")
print(df.head())


# =========================================================
# 5. Metadata 결측값
# =========================================================

print("\n결측값:")
print(df.isnull().sum())


# =========================================================
# 6. Metadata sample ID 확인
# =========================================================

if "sample_id" in df.columns:

    metadata_ids = set(
        df["sample_id"].astype(str)
    )

    print("\nMetadata에만 존재:")
    print(sorted(metadata_ids - png_ids))

    print("\n파일에는 있지만 Metadata에 없는 것:")
    print(sorted(png_ids - metadata_ids))

    if metadata_ids == png_ids:
        print("\n✅ PNG / NPY / Metadata ID 모두 일치")
    else:
        print("\n❌ ID가 일치하지 않음")

else:

    print("\n⚠️ sample_id 컬럼이 없습니다.")


print("\n검사 완료")
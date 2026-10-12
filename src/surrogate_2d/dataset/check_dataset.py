import numpy as np
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "configs" / "classroom").is_dir()
                    and (parent / "data").is_dir()
                    and (parent / "results").is_dir())
NPY_DIR = PROJECT_ROOT / "data" / "surrogate_2d" / "dataset" / "npy"

files = sorted(NPY_DIR.glob("*.npy"))

print("=" * 60)
print(f"NPY 파일 개수: {len(files)}")
print("=" * 60)

for file in files[:5]:
    data = np.load(file)

    print(f"\n파일: {file.name}")
    print(f"shape: {data.shape}")
    print(f"dtype: {data.dtype}")
    print(f"min: {np.nanmin(data):.3f}")
    print(f"max: {np.nanmax(data):.3f}")
    print(f"mean: {np.nanmean(data):.3f}")
    print(f"NaN 개수: {np.isnan(data).sum()}")

print("\n검사 완료")
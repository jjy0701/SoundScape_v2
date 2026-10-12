import csv
from pathlib import Path


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "configs" / "classroom").is_dir()
                    and (parent / "data").is_dir()
                    and (parent / "results").is_dir())
METADATA = PROJECT_ROOT / "data" / "surrogate_2d" / "dataset" / "metadata.csv"


# =========================================================
# CSV 읽기
# =========================================================

with open(
    METADATA,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)
    columns = reader.fieldnames


# =========================================================
# sample_id 수정
# =========================================================

for row in rows:

    old_id = int(row["sample_id"])

    row["sample_id"] = f"sample_{old_id:04d}"


# =========================================================
# CSV 다시 저장
# =========================================================

with open(
    METADATA,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columns
    )

    writer.writeheader()
    writer.writerows(rows)


print("==============================================")
print("Metadata ID 수정 완료")
print("==============================================")

print("처음 :", rows[0]["sample_id"])
print("마지막 :", rows[-1]["sample_id"])
print("총 데이터 :", len(rows))
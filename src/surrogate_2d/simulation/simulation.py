import pyroomacoustics as pra
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "configs" / "classroom").is_dir()
                    and (parent / "data").is_dir()
                    and (parent / "results").is_dir())
OUTPUT_PATH = PROJECT_ROOT / "results" / "surrogate_2d" / "generated" / "rir_10x12.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1. 10m x 12m 크기의 2D 방 생성
room = pra.ShoeBox(
    [10, 12],
    fs=16000,
    max_order=10
)

# 2. 스피커 위치
room.add_source([5, 6])

# 3. 마이크 위치
room.add_microphone([7, 6])

# 4. 방의 음향 반응 계산
room.compute_rir()

# 5. 마이크에서 받은 RIR 가져오기
rir = room.rir[0][0]

# 6. RIR 그래프
plt.figure(figsize=(10, 4))
plt.plot(rir)

plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("Room Impulse Response - 10m x 12m")

plt.grid()
plt.savefig(OUTPUT_PATH)
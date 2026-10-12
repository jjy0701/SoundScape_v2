import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "configs" / "classroom").is_dir()
                    and (parent / "data").is_dir()
                    and (parent / "results").is_dir())
OUTPUT_DIR = PROJECT_ROOT / "results" / "surrogate_2d" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 방 하나의 SPL Heatmap 생성
# ==========================================
def generate_heatmap(Lx, Ly, filename):
    
    # 1. 방 생성
    room = pra.ShoeBox(
        [Lx, Ly],
        fs=16000,
        max_order=10
    )

    # 2. 스피커 위치
    # 벽에서 약 1m 떨어진 위치
    source_x = Lx / 2
    source_y = Ly - 1.0

    room.add_source([source_x, source_y])

    # 3. 측정 지점(Grid)
    grid_size = 30

    x = np.linspace(0.5, Lx - 0.5, grid_size)
    y = np.linspace(0.5, Ly - 0.5, grid_size)

    positions = []

    for yy in y:
        for xx in x:
            positions.append([xx, yy])

    # 4. 모든 위치에 마이크 배치
    for pos in positions:
        room.add_microphone(pos)

    # 5. RIR 계산
    room.compute_rir()

    # 6. 각 위치의 RIR에서 에너지 계산
    levels = []

    for i in range(len(positions)):
        rir = room.rir[i][0]

        # RIR 에너지
        energy = np.sum(rir ** 2)

        # log 변환
        if energy > 0:
            level = 10 * np.log10(energy)
        else:
            level = -100

        levels.append(level)

    levels = np.array(levels)

    # 7. 2차원 Heatmap 형태로 변환
    heatmap = levels.reshape(grid_size, grid_size)

    # ==========================================
    # 비교를 위해 각 방 내부에서 최대값을 0 dB로 정규화
    # ==========================================
    heatmap = heatmap - np.max(heatmap)

    # ==========================================
    # 8. Heatmap 출력
    # ==========================================
    plt.figure(figsize=(8, 6))

    plt.imshow(
        heatmap,
        extent=[0, Lx, 0, Ly],
        origin="lower",
        aspect="equal",
        cmap="jet"
    )

    plt.colorbar(label="Relative SPL (dB)")

    # 스피커 위치 표시
    plt.scatter(
        source_x,
        source_y,
        marker="*",
        s=200,
        color="white",
        edgecolors="black",
        label="Speaker"
    )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"SPL Heatmap - {Lx}m × {Ly}m")

    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"{Lx} × {Ly} Heatmap 생성 완료")
    print(f"최대값: {np.max(heatmap):.2f} dB")
    print(f"최소값: {np.min(heatmap):.2f} dB")


# ==========================================
# 3개 방 실행
# ==========================================

generate_heatmap(6, 6, str(OUTPUT_DIR / "heatmap_6x6.png"))

generate_heatmap(8, 10, str(OUTPUT_DIR / "heatmap_8x10.png"))

generate_heatmap(10, 12, str(OUTPUT_DIR / "heatmap_10x12.png"))
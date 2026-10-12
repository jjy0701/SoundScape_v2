import os
import csv
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# 1. 기본 설정
# =========================================================

FS = 16000

ROOM_X = 8.0
ROOM_Y = 10.0

# 이미지 소스 반사 차수
MAX_ORDER = 8

# 20 x 20 = 400개 수음점
GRID_SIZE = 20

# 생성할 데이터 개수
NUM_SAMPLES = 100

# 랜덤 시드
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# 2. Dataset 폴더 생성
# =========================================================

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "configs" / "classroom").is_dir()
                    and (parent / "data").is_dir()
                    and (parent / "results").is_dir())
DATASET_DIR = str(PROJECT_ROOT / "data" / "surrogate_2d" / "dataset")

PNG_DIR = os.path.join(
    DATASET_DIR,
    "png"
)

NPY_DIR = os.path.join(
    DATASET_DIR,
    "npy"
)

os.makedirs(
    PNG_DIR,
    exist_ok=True
)

os.makedirs(
    NPY_DIR,
    exist_ok=True
)


# =========================================================
# 3. 기본 음향 설정
# =========================================================

# 벽 흡음률
WALL_ALPHA = 0.2

# 장애물 흡음률
OBSTACLE_ALPHA = 0.1


# =========================================================
# 4. Wall2D 생성 함수
# =========================================================

def make_wall(p1, p2, absorption, name):

    corners = np.array(
        [
            p1,
            p2
        ],
        dtype=np.float32
    ).T

    return pra.wall_factory(
        corners,
        np.array(
            [absorption],
            dtype=np.float32
        ),
        np.array(
            [0.0],
            dtype=np.float32
        ),
        name
    )


# =========================================================
# 5. Room 외벽 생성
# =========================================================

def create_outer_walls():

    outer = [
        [0.0, 0.0],
        [ROOM_X, 0.0],
        [ROOM_X, ROOM_Y],
        [0.0, ROOM_Y]
    ]

    walls = []

    for i in range(4):

        p1 = outer[i]

        p2 = outer[
            (i + 1) % 4
        ]

        walls.append(
            make_wall(
                p1,
                p2,
                WALL_ALPHA,
                f"outer_{i}"
            )
        )

    return walls


# =========================================================
# 6. 장애물 생성
# =========================================================

def create_obstacle_walls(
    obstacle_x,
    obstacle_y,
    obstacle_w,
    obstacle_l
):

    x1 = obstacle_x
    y1 = obstacle_y

    x2 = obstacle_x + obstacle_w
    y2 = obstacle_y + obstacle_l

    # CW 방향
    obstacle = [
        [x1, y1],
        [x1, y2],
        [x2, y2],
        [x2, y1]
    ]

    walls = []

    for i in range(4):

        p1 = obstacle[i]

        p2 = obstacle[
            (i + 1) % 4
        ]

        walls.append(
            make_wall(
                p1,
                p2,
                OBSTACLE_ALPHA,
                f"obstacle_{i}"
            )
        )

    return walls


# =========================================================
# 7. 랜덤 장애물 크기
# =========================================================

def random_obstacle_size():

    # 장애물 폭
    obstacle_w = random.uniform(
        1.5,
        2.5
    )

    # 장애물 깊이
    obstacle_l = random.uniform(
        0.8,
        1.5
    )

    return obstacle_w, obstacle_l


# =========================================================
# 8. 랜덤 장애물 위치
# =========================================================

def random_obstacle_position(
    obstacle_w,
    obstacle_l
):

    # 방 가장자리에서 어느 정도 떨어뜨림
    margin = 0.8

    obstacle_x = random.uniform(
        margin,
        ROOM_X - obstacle_w - margin
    )

    obstacle_y = random.uniform(
        2.5,
        ROOM_Y - obstacle_l - margin
    )

    return obstacle_x, obstacle_y


# =========================================================
# 9. 랜덤 스피커 위치
#
# 장애물 내부에 들어가지 않도록 처리
# =========================================================

def random_speaker_position(
    obstacle_x,
    obstacle_y,
    obstacle_w,
    obstacle_l
):

    margin = 0.5

    while True:

        speaker_x = random.uniform(
            0.7,
            ROOM_X - 0.7
        )

        speaker_y = random.uniform(
            0.7,
            ROOM_Y - 0.7
        )

        # 장애물 영역
        inside_obstacle = (
            obstacle_x - margin
            <= speaker_x
            <= obstacle_x + obstacle_w + margin
            and
            obstacle_y - margin
            <= speaker_y
            <= obstacle_y + obstacle_l + margin
        )

        if not inside_obstacle:

            return (
                speaker_x,
                speaker_y
            )


# =========================================================
# 10. Grid 생성
# =========================================================

def create_grid():

    x = np.linspace(
        0.5,
        ROOM_X - 0.5,
        GRID_SIZE
    )

    y = np.linspace(
        0.5,
        ROOM_Y - 0.5,
        GRID_SIZE
    )

    X, Y = np.meshgrid(
        x,
        y
    )

    return X, Y


# =========================================================
# 11. 수음점 생성
# =========================================================

def create_receivers(
    X,
    Y,
    obstacle_x,
    obstacle_y,
    obstacle_w,
    obstacle_l
):

    obstacle_mask = (
        (X >= obstacle_x)
        &
        (X <= obstacle_x + obstacle_w)
        &
        (Y >= obstacle_y)
        &
        (Y <= obstacle_y + obstacle_l)
    )

    receiver_positions = []

    receiver_indices = []

    for i in range(GRID_SIZE):

        for j in range(GRID_SIZE):

            # 장애물 내부는 수음점에서 제외
            if obstacle_mask[i, j]:
                continue

            receiver_positions.append(
                [
                    X[i, j],
                    Y[i, j]
                ]
            )

            receiver_indices.append(
                (i, j)
            )

    receiver_positions = np.array(
        receiver_positions,
        dtype=np.float32
    )

    return (
        receiver_positions,
        receiver_indices
    )


# =========================================================
# 12. SPL 계산
# =========================================================

def calculate_spl(
    room,
    X,
    receiver_indices
):

    spl = np.full(
        X.shape,
        np.nan,
        dtype=np.float32
    )

    for mic_index, (i, j) in enumerate(
        receiver_indices
    ):

        # 첫 번째 source의 RIR
        rir = room.rir[mic_index][0]

        # RIR energy
        energy = np.sum(
            rir ** 2
        )

        energy = max(
            energy,
            1e-12
        )

        # Energy -> dB
        spl_value = (
            10.0 *
            np.log10(energy)
        )

        spl[i, j] = spl_value

    # 최대값을 0 dB로 정규화
    max_spl = np.nanmax(spl)

    spl = spl - max_spl

    return spl


# =========================================================
# 13. Heatmap 저장
# =========================================================

def save_heatmap(
    spl,
    X,
    Y,
    speaker_x,
    speaker_y,
    obstacle_x,
    obstacle_y,
    obstacle_w,
    obstacle_l,
    save_path,
    sample_id
):

    plt.figure(
        figsize=(8, 10)
    )

    plt.pcolormesh(
        X,
        Y,
        spl,
        shading="auto",
        cmap="viridis"
    )

    plt.colorbar(
        label="Relative SPL (dB)"
    )

    # -----------------------------------------------------
    # Speaker
    # -----------------------------------------------------

    plt.scatter(
        speaker_x,
        speaker_y,
        marker="*",
        s=300,
        edgecolors="black",
        label="Speaker"
    )

    # -----------------------------------------------------
    # Obstacle
    # -----------------------------------------------------

    plt.gca().add_patch(
        plt.Rectangle(
            (
                obstacle_x,
                obstacle_y
            ),
            obstacle_w,
            obstacle_l,
            facecolor="white",
            edgecolor="black",
            linewidth=3,
            label="Obstacle"
        )
    )

    # -----------------------------------------------------
    # 그래프 설정
    # -----------------------------------------------------

    plt.xlabel(
        "X (m)"
    )

    plt.ylabel(
        "Y (m)"
    )

    plt.title(
        f"SPL Heatmap "
        f"Sample {sample_id:04d}\n"
        f"Room = {ROOM_X} × {ROOM_Y} m"
    )

    plt.legend()

    plt.axis(
        "equal"
    )

    plt.xlim(
        0,
        ROOM_X
    )

    plt.ylim(
        0,
        ROOM_Y
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150
    )

    plt.close()


# =========================================================
# 14. Metadata CSV 생성
# =========================================================

metadata_path = os.path.join(
    DATASET_DIR,
    "metadata.csv"
)

metadata_file = open(
    metadata_path,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(
    metadata_file
)

writer.writerow(
    [
        "sample_id",

        "room_x",
        "room_y",

        "speaker_x",
        "speaker_y",

        "obstacle_x",
        "obstacle_y",

        "obstacle_w",
        "obstacle_l",

        "wall_alpha",
        "obstacle_alpha"
    ]
)


# =========================================================
# 15. Grid
# =========================================================

X, Y = create_grid()


# =========================================================
# 16. 데이터 생성 시작
# =========================================================

print()
print("=" * 60)
print("SPL Dataset Generation")
print("=" * 60)
print()

for sample_id in range(
    NUM_SAMPLES
):

    print(
        f"[{sample_id + 1:03d}/{NUM_SAMPLES}] "
        f"데이터 생성 중..."
    )

    # =====================================================
    # 랜덤 장애물 크기
    # =====================================================

    obstacle_w, obstacle_l = (
        random_obstacle_size()
    )

    # =====================================================
    # 랜덤 장애물 위치
    # =====================================================

    obstacle_x, obstacle_y = (
        random_obstacle_position(
            obstacle_w,
            obstacle_l
        )
    )

    # =====================================================
    # 랜덤 스피커 위치
    # =====================================================

    speaker_x, speaker_y = (
        random_speaker_position(
            obstacle_x,
            obstacle_y,
            obstacle_w,
            obstacle_l
        )
    )

    speaker = np.array(
        [
            speaker_x,
            speaker_y
        ],
        dtype=np.float32
    )

    # =====================================================
    # Room 생성
    # =====================================================

    walls = create_outer_walls()

    obstacle_walls = (
        create_obstacle_walls(
            obstacle_x,
            obstacle_y,
            obstacle_w,
            obstacle_l
        )
    )

    walls.extend(
        obstacle_walls
    )

    room = pra.Room(
        walls,
        fs=FS,
        max_order=MAX_ORDER
    )

    # =====================================================
    # Speaker 추가
    # =====================================================

    room.add_source(
        speaker
    )

    # =====================================================
    # Receiver 생성
    # =====================================================

    (
        receiver_positions,
        receiver_indices
    ) = create_receivers(
        X,
        Y,
        obstacle_x,
        obstacle_y,
        obstacle_w,
        obstacle_l
    )

    # =====================================================
    # Microphone Array
    # =====================================================

    room.add_microphone_array(
        pra.MicrophoneArray(
            receiver_positions.T,
            FS
        )
    )

    # =====================================================
    # RIR 계산
    # =====================================================

    room.compute_rir()

    # =====================================================
    # SPL 계산
    # =====================================================

    spl = calculate_spl(
        room,
        X,
        receiver_indices
    )

    # =====================================================
    # NPY 저장
    # =====================================================

    npy_path = os.path.join(
        NPY_DIR,
        f"sample_{sample_id:04d}.npy"
    )

    np.save(
        npy_path,
        spl.astype(np.float32)
    )

    # =====================================================
    # PNG 저장
    # =====================================================

    png_path = os.path.join(
        PNG_DIR,
        f"sample_{sample_id:04d}.png"
    )

    save_heatmap(
        spl,
        X,
        Y,

        speaker_x,
        speaker_y,

        obstacle_x,
        obstacle_y,

        obstacle_w,
        obstacle_l,

        png_path,

        sample_id
    )

    # =====================================================
    # Metadata 저장
    # =====================================================

    writer.writerow(
        [
            sample_id,

            ROOM_X,
            ROOM_Y,

            speaker_x,
            speaker_y,

            obstacle_x,
            obstacle_y,

            obstacle_w,
            obstacle_l,

            WALL_ALPHA,
            OBSTACLE_ALPHA
        ]
    )

    metadata_file.flush()

    print(
        f"    Speaker = "
        f"({speaker_x:.2f}, {speaker_y:.2f})"
    )

    print(
        f"    Obstacle = "
        f"({obstacle_x:.2f}, {obstacle_y:.2f})"
        f"  size=({obstacle_w:.2f}, {obstacle_l:.2f})"
    )

    print(
        f"    PNG -> {png_path}"
    )

    print(
        f"    NPY -> {npy_path}"
    )

    print()


# =========================================================
# 17. 종료
# =========================================================

metadata_file.close()

print("=" * 60)
print("데이터 생성 완료!")
print("=" * 60)

print()
print(
    f"PNG 폴더 : {PNG_DIR}"
)

print(
    f"NPY 폴더 : {NPY_DIR}"
)

print(
    f"Metadata  : {metadata_path}"
)

print()
print(
    f"총 {NUM_SAMPLES}개 데이터 생성 완료"
)
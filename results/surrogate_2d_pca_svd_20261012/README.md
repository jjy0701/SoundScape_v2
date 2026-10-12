# 2D acoustic-map PCA/SVD reconstruction

실행: `.venv\Scripts\python.exe pca_svd_reconstruction_2d.py`

기존 `split_manifest.json`의 train 70 / validation 15 / test 15를 그대로 사용했습니다. NPY는 100개이며 20×20, y,x 순서, 0.5 m offset입니다. 기존 생성 코드는 유효 영역 최대값을 0 dB로 정규화하고 장애물 내부를 NaN으로 저장합니다.

Train에서 칸별 평균으로 NaN을 채운 뒤 중심화 SVD를 수행했습니다. train에서 한 번도 유효하지 않은 칸은 PCA에서 제외했고, 평가에서는 원래 NaN과 제외 칸을 모두 제외했습니다. 장애물 내부를 0 dB로 취급하지 않았습니다. 표준화, 복원 후 재정규화·클리핑은 없습니다. 이 결측 대체는 편향 가능한 기준 실험입니다.

선택된 그림 k: 50. val 평균 장면 RMSE 0.5 dB 이하 후보를 우선 사용하고, 없으면 최대 유효 k를 사용했습니다. 설명분산은 결측을 채운 train 행렬 기준이며 실측 또는 AI 정확도가 아닙니다. test는 평가·성분 선택에 사용하지 않았습니다.

# 2D surrogate train-3 recomputation pilot

기존 manifest의 정렬된 train 첫 3개만 metadata 형상으로 재계산했습니다. 새 장면 생성, validation/test 사용, 주파수 필터, AI 학습은 하지 않았습니다. 원본 NPY는 변경하지 않았습니다. 각 NPZ에는 raw RIR, 수음점 좌표·인덱스, 마스크, floor 전 에너지, floor 여부, 정규화 전 dB, 장면 최대값, 정규화 지도, 원본 지도가 포함됩니다. 비교 오차는 유효 격자에서만 계산했습니다.

실행: `.venv\Scripts\python.exe pilot_recompute_surrogate2d_train3.py`

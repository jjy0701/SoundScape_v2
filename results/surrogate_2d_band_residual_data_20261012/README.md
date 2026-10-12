# Surrogate 2D band direct-vs-residual data preparation

기존 train 70개와 validation 15개만 사용해 대역별 직접 예측(A)과 광대역+대역 잔여 예측(B)을 비교하기 위한 입력 자료를 준비했습니다. test 15개는 사용하지 않았고 AI 학습은 실행하지 않았습니다. pilot 3개는 기존 raw RIR 캐시를 재사용하고, 나머지는 metadata 형상으로 재계산했습니다. 장면별 NPZ에는 RIR, 실제 길이, 수음점 대응, 마스크, 광대역·대역별 에너지와 정규화 기준, 기존 NPY 비교값이 있습니다.

RIR 길이가 n_fft=16384를 넘으면 저장은 중단해야 하는 조건으로 검사하며, 실제 이번 처리에서 초과 장면이 있으면 metrics에 표시합니다. 이전 FFT 2N→4N 최대 지도 변화 0.0599 dB와 0.05 dB 잠정 기준 미달 사실을 설계 기록에 보존했습니다.

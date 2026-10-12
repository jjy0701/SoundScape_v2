# Corrected FFT energy error report

기존 pilot NPZ만 사용해 n_fft와 2*n_fft의 상대오차 집계 축을 분리했습니다. RIR 재계산과 AI 학습은 하지 않았습니다. 기존 band-energy 결과는 보존됩니다. 기존 0.159%는 수음점별 네 대역 합산 후의 최대 상대오차(metric 2)이며, 개별 대역·수음점 최대 상대오차(metric 1)는 더 큽니다.

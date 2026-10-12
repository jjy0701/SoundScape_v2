# Pilot train-3 band energy maps

저장된 pilot NPZ의 raw RIR만 사용했습니다. 실제 RIR 길이로 자른 뒤 공통 zero-padding과 rFFT를 적용했고, 단측 스펙트럼 Parseval 가중치를 사용했습니다. 대역은 [하한, 상한)입니다. 각 대역별 0 dB 지도와 광대역 최대 에너지를 공통 기준으로 한 지도를 저장했습니다. FFT 적분 결과는 기존 Butterworth 필터 분석과 동일한 수치로 간주하지 않습니다. raw dB는 절대 SPL이나 실측 음압이 아닙니다. 장애물 내부는 NaN입니다.

# 동시 재생 에너지/I 계산 대조

입력 `results\today_B_comparison_20261012_20261012_185129`의 기존 저장 pooled 레벨과 동일한 B 모델 설정으로 재계산한 값을 대조했다. 물리적 해석과 새 보정값 추정은 수행하지 않았다.

- `b0_energy_comparison.csv`: 지점·대역별 기존/새 B0 E_left, E_right, E_both, I와 차이
- `gain_method_comparison.csv`: 직접 파형 합산(A)과 `E_left + g²E_right + g(E_both−E_left−E_right)`(B) 비교. I 분모는 항상 `E_left + g²E_right`.
- `measured_pooled_recheck.csv`: pooled 레벨에서 실측 start/end I 재계산
- `summary.json`: 최대 불일치 요약

모든 에너지는 같은 대역 필터와 같은 모델 분석창을 사용했다.

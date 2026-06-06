
## CEP50 Comparison (m)

| Dataset | Standard-LS | WLS-MoG | FG-MoG | FG+TCN | Adaptive-M3 | Best |
|---------|:----------:|:------:|:------:|:------:|:----------:|:----:|
| berlin1 | 904.5 | 936.7 | 936.7 | 936.7 | 899.7 | **Adaptive-M3** |
| berlin2 | 610.8 | 587.6 | 587.6 | 587.6 | 592.8 | FG+TCN |
| frankfurt1 | 525.2 | 596.9 | 596.9 | 596.9 | 520.2 | **Adaptive-M3** |
| frankfurt2 | 382.6 | 550.4 | 550.4 | 550.4 | 367.0 | **Adaptive-M3** |

## Method Selection Distribution

- **berlin1**: Standard-LS: 83%, Standard-LS(fallback): 11%, FG-MoG+2A: 7%
- **berlin2**: Standard-LS: 78%, FG-MoG+2A: 18%, Standard-LS(fallback): 4%
- **frankfurt1**: Standard-LS: 93%, Standard-LS(fallback): 4%, FG-MoG+2A: 3%
- **frankfurt2**: Standard-LS: 73%, Standard-LS(fallback): 16%, FG-MoG+2A: 11%

## Online Learning Effect

- **berlin1**: +63.2% (586m → 216m)
- **berlin2**: -21.0% (679m → 821m)
- **frankfurt1**: +44.4% (268m → 149m)
- **frankfurt2**: -490.7% (132m → 781m)

## Success Criteria

- **C1_Adaptive_not_worse_than_LS**: PASS
- **C2_Adaptive_beats_best_static_3of4**: PASS (3/4)
- **C3_Learning_effect_2of4**: PASS (2/4)
- **C4_Frankfurt1_adaptive_under_490m**: FAIL (CEP50=520.2m)
- **C5_CUSUM_detection**: PASS
- **BONUS_TCN_improves_3of4**: FAIL (1/4)

**Total pipeline time**: 0.7 min
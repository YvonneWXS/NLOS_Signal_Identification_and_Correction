
## CEP50 Comparison (m)

| Dataset | Standard-LS | WLS-MoG | FG-MoG | FG+TCN | Adaptive-M3 | Best |
|---------|:----------:|:------:|:------:|:------:|:----------:|:----:|
| berlin1 | 904.5 | 968.6 | 968.6 | 968.6 | 872.8 | **Adaptive-M3** |
| berlin2 | 610.8 | 750.2 | 750.2 | 750.2 | 598.5 | **Adaptive-M3** |
| frankfurt1 | 525.2 | 472.6 | 472.6 | 472.6 | 467.4 | **Adaptive-M3** |
| frankfurt2 | 382.6 | 562.8 | 562.8 | 562.8 | 368.0 | **Adaptive-M3** |

## Method Selection Distribution

- **berlin1**: Standard-LS: 53%, Standard-LS(fallback): 36%, FG-MoG+2A: 11%, WLS-MoG: 1%
- **berlin2**: Standard-LS(fallback): 49%, FG-MoG+2A: 39%, Standard-LS: 11%, WLS-MoG: 1%
- **frankfurt1**: FG-MoG+2A: 46%, Standard-LS(fallback): 29%, Standard-LS: 26%
- **frankfurt2**: Standard-LS: 44%, Standard-LS(fallback): 37%, FG-MoG+2A: 20%

## Online Learning Effect

- **berlin1**: +64.9% (587m → 206m)
- **berlin2**: -9.7% (740m → 812m)
- **frankfurt1**: +51.1% (269m → 131m)
- **frankfurt2**: -490.7% (132m → 781m)

## Success Criteria

- **C1_Adaptive_not_worse_than_LS**: PASS
- **C2_Adaptive_beats_best_static_3of4**: PASS (4/4)
- **C3_Learning_effect_2of4**: PASS (2/4)
- **C4_Frankfurt1_adaptive_under_490m**: PASS (CEP50=467.4m)
- **C5_CUSUM_detection**: PASS
- **BONUS_TCN_improves_3of4**: FAIL (0/4)

**Total pipeline time**: 0.6 min
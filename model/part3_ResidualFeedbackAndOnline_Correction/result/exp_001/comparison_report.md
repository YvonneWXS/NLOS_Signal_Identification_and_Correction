
## CEP50 Comparison (m)

| Dataset | Standard-LS | WLS-MoG | FG-MoG | Adaptive-M3 | Best |
|---------|:----------:|:------:|:------:|:----------:|:----:|
| berlin1 | 1016.2 | 949.7 | 949.7 | 930.0 | **Adaptive-M3** |
| berlin2 | 721.7 | 659.3 | 659.3 | 649.2 | **Adaptive-M3** |
| frankfurt1 | 513.7 | 596.9 | 596.9 | 496.7 | **Adaptive-M3** |
| frankfurt2 | 545.3 | 639.9 | 639.9 | 524.1 | **Adaptive-M3** |

## Method Selection Distribution

- **berlin1**: Standard-LS(fallback): 36%, WLS-MoG: 29%, FG-MoG+2A: 28%, Standard-LS: 7%
- **berlin2**: FG-MoG+2A: 64%, Standard-LS(fallback): 19%, WLS-MoG: 10%, Standard-LS: 7%
- **frankfurt1**: Standard-LS(fallback): 63%, Standard-LS: 21%, WLS-MoG: 15%, FG-MoG+2A: 2%
- **frankfurt2**: Standard-LS(fallback): 61%, Standard-LS: 19%, WLS-MoG: 12%, FG-MoG+2A: 8%

## Success Criteria

- **C1_Adaptive_not_worse_than_LS**: PASS
- **C2_Adaptive_beats_best_static_3of4**: PASS (4/4)
- **C3_Learning_effect_2of4**: FAIL (1/4)
- **C4_Frankfurt1_adaptive_under_490m**: FAIL (CEP50=496.7m)
- **C5_CUSUM_detection**: PASS

**Total pipeline time**: 1.0 min
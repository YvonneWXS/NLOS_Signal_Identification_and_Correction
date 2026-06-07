# Module 3 v2 Results: frankfurt2_westendtower

## Primary Positioning Table

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 382.6 | 1295.9 | 507.4 |  | 1.5% |
| WLS-MoG | 550.4 | 1313.2 | 603.4 | -43.9% | 0.7% |
| FG-MoG | 550.4 | 1313.2 | 603.4 | -43.9% | 0.7% |
| FG-MoG+TCN | 550.4 | 1313.2 | 603.4 | -43.9% | 0.7% |
| Adaptive-M3 | 373.8 | 1294.3 | 499.0 | +2.3% | 1.5% |

## Method Selection Distribution

- **Standard-LS**: 79.6%
- **Standard-LS(fallback)**: 12.4%
- **FG-MoG+TCN**: 8.0%

**Adaptive vs Best Static**: +2.3% (adaptive=373.8m, best_static=382.6m)

**Online Learning**: -490.7% (first 100→132.2m, last 100→781.1m)

**CUSUM**: 0 positive shifts, 100 negative shifts (100 total)
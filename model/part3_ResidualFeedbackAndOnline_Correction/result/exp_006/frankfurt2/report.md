# Module 3 v2 Results: frankfurt2_westendtower

## Primary Positioning Table

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 382.6 | 1295.9 | 507.4 |  | 1.5% |
| WLS-MoG | 562.8 | 1325.4 | 615.9 | -47.1% | 0.5% |
| FG-MoG | 562.8 | 1325.4 | 615.9 | -47.1% | 0.5% |
| FG-MoG+TCN | 562.8 | 1325.4 | 615.9 | -47.1% | 0.5% |
| Adaptive-M3 | 368.0 | 1281.8 | 485.0 | +3.8% | 1.6% |

## Method Selection Distribution

- **Standard-LS**: 43.7%
- **Standard-LS(fallback)**: 36.7%
- **FG-MoG+2A**: 19.6%

**Adaptive vs Best Static**: +3.8% (adaptive=368.0m, best_static=382.6m)

**Online Learning**: -490.7% (first 100→132.2m, last 100→781.1m)

**CUSUM**: 0 positive shifts, 241 negative shifts (241 total)
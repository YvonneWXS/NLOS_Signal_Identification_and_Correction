# Module 3 v2 Results: berlin2_gendarmenmarkt

## Primary Positioning Table

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 610.8 | 1329.7 | 655.3 |  | 2.4% |
| WLS-MoG | 750.2 | 1301.2 | 715.3 | -22.8% | 1.1% |
| FG-MoG | 750.2 | 1301.2 | 715.3 | -22.8% | 1.1% |
| FG-MoG+TCN | 750.2 | 1301.2 | 715.3 | -22.8% | 1.1% |
| Adaptive-M3 | 598.5 | 1253.4 | 626.5 | +2.0% | 2.5% |

## Method Selection Distribution

- **Standard-LS(fallback)**: 49.2%
- **FG-MoG+2A**: 39.1%
- **Standard-LS**: 10.9%
- **WLS-MoG**: 0.8%

**Adaptive vs Best Static**: +2.0% (adaptive=598.5m, best_static=610.8m)

**Online Learning**: -9.7% (first 100→740.4m, last 100→811.9m)

**CUSUM**: 0 positive shifts, 705 negative shifts (705 total)
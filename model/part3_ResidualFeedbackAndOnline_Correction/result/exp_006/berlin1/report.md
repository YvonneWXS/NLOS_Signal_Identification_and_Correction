# Module 3 v2 Results: berlin1_potsdamer_platz

## Primary Positioning Table

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 904.5 | 1289.1 | 779.7 |  | 1.7% |
| WLS-MoG | 968.6 | 1666.3 | 939.1 | -7.1% | 1.6% |
| FG-MoG | 968.6 | 1666.3 | 939.1 | -7.1% | 1.6% |
| FG-MoG+TCN | 968.6 | 1666.3 | 939.1 | -7.1% | 1.6% |
| Adaptive-M3 | 872.8 | 1278.1 | 772.3 | +3.5% | 1.7% |

## Method Selection Distribution

- **Standard-LS**: 52.9%
- **Standard-LS(fallback)**: 35.7%
- **FG-MoG+2A**: 10.7%
- **WLS-MoG**: 0.7%

**Adaptive vs Best Static**: +3.5% (adaptive=872.8m, best_static=904.5m)

**Online Learning**: +64.9% (first 100→587.2m, last 100→206.0m)

**CUSUM**: 0 positive shifts, 27 negative shifts (27 total)
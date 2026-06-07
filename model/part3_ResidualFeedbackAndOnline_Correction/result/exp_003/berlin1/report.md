# Module 3 v2 Results: berlin1_potsdamer_platz

## Primary Positioning Table

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 904.5 | 1289.1 | 779.7 |  | 1.7% |
| WLS-MoG | 936.7 | 1391.8 | 848.2 | -3.6% | 1.7% |
| FG-MoG | 936.7 | 1391.8 | 848.2 | -3.6% | 1.7% |
| FG-MoG+TCN | 936.7 | 1391.8 | 848.2 | -3.6% | 1.7% |
| Adaptive-M3 | 899.7 | 1292.1 | 778.2 | +0.5% | 1.7% |

## Method Selection Distribution

- **Standard-LS**: 83.4%
- **Standard-LS(fallback)**: 10.2%
- **FG-MoG+2A**: 6.3%
- **WLS-MoG**: 0.1%

**Adaptive vs Best Static**: +0.5% (adaptive=899.7m, best_static=904.5m)

**Online Learning**: +63.2% (first 100→586.5m, last 100→215.8m)

**CUSUM**: 2 positive shifts, 6 negative shifts (18 total)
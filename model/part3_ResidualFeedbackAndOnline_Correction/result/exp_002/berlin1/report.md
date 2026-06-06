# Module 3 v2 Results: berlin1_potsdamer_platz

## Primary Positioning Table

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 904.5 | 1289.1 | 779.7 |  | 1.7% |
| WLS-MoG | 936.7 | 1391.8 | 848.2 | -3.6% | 1.7% |
| FG-MoG | 936.7 | 1391.8 | 848.2 | -3.6% | 1.7% |
| FG-MoG+TCN | 936.7 | 1391.8 | 848.2 | -3.6% | 1.7% |
| Adaptive-M3 | 899.7 | 1292.1 | 778.3 | +0.5% | 1.7% |

## Method Selection Distribution

- **Standard-LS**: 82.6%
- **Standard-LS(fallback)**: 10.7%
- **FG-MoG+2A**: 6.7%

**Adaptive vs Best Static**: +0.5% (adaptive=899.7m, best_static=904.5m)

**Online Learning**: +63.2% (first 100→586.5m, last 100→215.8m)

**CUSUM**: 0 positive shifts, 5 negative shifts (5 total)
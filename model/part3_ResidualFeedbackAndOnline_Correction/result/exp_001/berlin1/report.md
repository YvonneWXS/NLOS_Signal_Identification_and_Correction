# Module 3 Results: berlin1_potsdamer_platz

## CEP50 Comparison (m)

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 1016.2 | 1315.6 | 876.2 |  | 0.0% |
| WLS-MoG | 949.7 | 1373.9 | 856.6 | +6.5% | 0.0% |
| FG-MoG | 949.7 | 1373.9 | 856.6 | +6.5% | 0.0% |
| Adaptive-M3 | 930.0 | 1285.2 | 833.3 | +8.5% | 0.0% |

## Method Selection Distribution

- **Standard-LS(fallback)**: 36.2%
- **WLS-MoG**: 29.1%
- **FG-MoG+2A**: 27.7%
- **Standard-LS**: 7.0%

**Adaptive vs Best Static**: +2.1% (adaptive=930.0m, best_static=949.7m)

**Online Learning Effect**: +63.3% (first 100=624.9m → last 100=229.4m)
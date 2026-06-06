# Module 3 Results: frankfurt2_westendtower

## CEP50 Comparison (m)

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 545.3 | 1062.7 | 553.4 |  | 0.4% |
| WLS-MoG | 639.9 | 1259.8 | 640.4 | -17.4% | 0.2% |
| FG-MoG | 639.9 | 1259.8 | 640.4 | -17.4% | 0.2% |
| Adaptive-M3 | 524.1 | 1054.4 | 531.1 | +3.9% | 0.4% |

## Method Selection Distribution

- **Standard-LS(fallback)**: 61.1%
- **Standard-LS**: 19.1%
- **WLS-MoG**: 11.9%
- **FG-MoG+2A**: 7.8%

**Adaptive vs Best Static**: +3.9% (adaptive=524.1m, best_static=545.3m)

**Online Learning Effect**: -509.6% (first 100=121.4m → last 100=740.0m)
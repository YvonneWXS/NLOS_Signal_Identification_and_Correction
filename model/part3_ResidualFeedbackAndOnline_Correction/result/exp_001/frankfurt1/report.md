# Module 3 Results: frankfurt1_maintower

## CEP50 Comparison (m)

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 513.7 | 1056.7 | 533.1 |  | 1.9% |
| WLS-MoG | 596.9 | 1235.8 | 623.0 | -16.2% | 1.3% |
| FG-MoG | 596.9 | 1235.8 | 623.0 | -16.2% | 1.3% |
| Adaptive-M3 | 496.7 | 1051.6 | 520.9 | +3.3% | 2.2% |

## Method Selection Distribution

- **Standard-LS(fallback)**: 62.9%
- **Standard-LS**: 20.6%
- **WLS-MoG**: 14.7%
- **FG-MoG+2A**: 1.8%

**Adaptive vs Best Static**: +3.3% (adaptive=496.7m, best_static=513.7m)

**Online Learning Effect**: -20.3% (first 100=213.3m → last 100=256.7m)
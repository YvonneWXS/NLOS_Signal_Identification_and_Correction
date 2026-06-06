# Module 3 Results: berlin2_gendarmenmarkt

## CEP50 Comparison (m)

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 721.7 | 1471.2 | 741.3 |  | 1.1% |
| WLS-MoG | 659.3 | 1313.2 | 675.7 | +8.6% | 1.3% |
| FG-MoG | 659.3 | 1313.2 | 675.7 | +8.6% | 1.3% |
| Adaptive-M3 | 649.2 | 1318.1 | 669.5 | +10.0% | 1.7% |

## Method Selection Distribution

- **FG-MoG+2A**: 64.1%
- **Standard-LS(fallback)**: 19.0%
- **WLS-MoG**: 9.9%
- **Standard-LS**: 7.0%

**Adaptive vs Best Static**: +1.5% (adaptive=649.2m, best_static=659.3m)

**Online Learning Effect**: -24.1% (first 100=711.7m → last 100=883.1m)
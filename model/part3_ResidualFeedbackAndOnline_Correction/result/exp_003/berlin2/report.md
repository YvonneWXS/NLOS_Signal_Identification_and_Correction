# Module 3 v2 Results: berlin2_gendarmenmarkt

## Primary Positioning Table

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 610.8 | 1329.7 | 655.3 |  | 2.4% |
| WLS-MoG | 587.6 | 1232.8 | 611.8 | +3.8% | 3.0% |
| FG-MoG | 587.6 | 1232.8 | 611.8 | +3.8% | 3.0% |
| FG-MoG+TCN | 587.6 | 1232.8 | 611.8 | +3.8% | 3.0% |
| Adaptive-M3 | 592.8 | 1309.1 | 634.9 | +3.0% | 2.7% |

## Method Selection Distribution

- **Standard-LS**: 78.2%
- **FG-MoG+2A**: 16.3%
- **Standard-LS(fallback)**: 4.3%
- **WLS-MoG**: 1.2%

**Adaptive vs Best Static**: -0.9% (adaptive=592.8m, best_static=587.6m)

**Online Learning**: -21.0% (first 100→678.8m, last 100→821.4m)

**CUSUM**: 0 positive shifts, 444 negative shifts (774 total)
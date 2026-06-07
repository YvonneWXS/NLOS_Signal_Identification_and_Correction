# Module 3 v2 Results: frankfurt1_maintower

## Primary Positioning Table

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 525.2 | 1505.2 | 610.9 |  | 3.4% |
| WLS-MoG | 596.9 | 1513.3 | 682.2 | -13.7% | 2.2% |
| FG-MoG | 596.9 | 1513.3 | 682.2 | -13.7% | 2.2% |
| FG-MoG+TCN | 596.9 | 1513.3 | 682.2 | -13.7% | 2.2% |
| Adaptive-M3 | 521.9 | 1498.0 | 608.3 | +0.6% | 3.4% |

## Method Selection Distribution

- **Standard-LS**: 95.3%
- **Standard-LS(fallback)**: 2.8%
- **FG-MoG+TCN**: 1.9%

**Adaptive vs Best Static**: +0.6% (adaptive=521.9m, best_static=525.2m)

**Online Learning**: +44.4% (first 100→268.0m, last 100→149.0m)

**CUSUM**: 0 positive shifts, 40 negative shifts (40 total)
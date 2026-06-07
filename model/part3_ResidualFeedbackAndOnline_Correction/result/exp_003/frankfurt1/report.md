# Module 3 v2 Results: frankfurt1_maintower

## Primary Positioning Table

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 525.2 | 1505.2 | 610.9 |  | 3.4% |
| WLS-MoG | 596.9 | 1513.3 | 682.2 | -13.7% | 2.2% |
| FG-MoG | 596.9 | 1513.3 | 682.2 | -13.7% | 2.2% |
| FG-MoG+TCN | 596.9 | 1513.3 | 682.2 | -13.7% | 2.2% |
| Adaptive-M3 | 520.2 | 1497.5 | 606.8 | +1.0% | 3.4% |

## Method Selection Distribution

- **Standard-LS**: 93.5%
- **Standard-LS(fallback)**: 4.0%
- **FG-MoG+2A**: 1.9%
- **WLS-MoG**: 0.7%

**Adaptive vs Best Static**: +1.0% (adaptive=520.2m, best_static=525.2m)

**Online Learning**: +44.4% (first 100→268.0m, last 100→149.0m)

**CUSUM**: 0 positive shifts, 66 negative shifts (123 total)
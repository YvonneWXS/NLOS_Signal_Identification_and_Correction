# Module 3 v2 Results: frankfurt1_maintower

## Primary Positioning Table

| Method | CEP50 | CEP95 | Mean 2D | vs Std LS | %<50m |
|--------|:-----:|:-----:|:-------:|:---------:|:-----:|
| Standard-LS | 525.2 | 1505.2 | 610.9 |  | 3.4% |
| WLS-MoG | 472.6 | 1559.4 | 594.3 | +10.0% | 2.2% |
| FG-MoG | 472.6 | 1559.4 | 594.3 | +10.0% | 2.2% |
| FG-MoG+TCN | 472.6 | 1559.4 | 594.3 | +10.0% | 2.2% |
| Adaptive-M3 | 467.4 | 1488.9 | 567.7 | +11.0% | 4.6% |

## Method Selection Distribution

- **FG-MoG+2A**: 45.7%
- **Standard-LS(fallback)**: 28.7%
- **Standard-LS**: 25.6%

**Adaptive vs Best Static**: +1.1% (adaptive=467.4m, best_static=472.6m)

**Online Learning**: +51.1% (first 100→268.6m, last 100→131.3m)

**CUSUM**: 4 positive shifts, 1048 negative shifts (1052 total)
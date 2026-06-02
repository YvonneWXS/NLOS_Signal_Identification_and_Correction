# Module 2 v2 Positioning Results

## CEP50 (m) — Median 2D Error

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-elevation | 1095.0 | 877.6 | 839.6 | 451.7 |
| WLS-MoG | 962.8 | 715.8 | 454.4 | 436.9 |
| Hard-threshold | 1393.0 | 1138.8 | 1286.5 | 695.0 |
| FactorGraph-MoG | 945.2 | 735.5 | 455.7 | 436.9 |
| FactorGraph-MoG+2A | 945.2 | 735.5 | 455.7 | 436.9 |

## CEP95 (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 1289.1 | 1329.7 | 1505.2 | 1295.9 |
| WLS-elevation | 1442.9 | 1572.7 | 1587.7 | 794.5 |
| WLS-MoG | 1635.8 | 1338.1 | 1567.3 | 1273.5 |
| Hard-threshold | 9687.8 | 2884.6 | 5391.3 | 2398.6 |
| FactorGraph-MoG | 1564.8 | 1232.0 | 1594.5 | 1273.5 |
| FactorGraph-MoG+2A | 1564.8 | 1232.0 | 1594.5 | 1273.5 |

## Mean 2D (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 779.7 | 655.3 | 610.9 | 507.4 |
| WLS-elevation | 950.1 | 881.0 | 858.6 | 441.0 |
| WLS-MoG | 930.6 | 708.7 | 591.1 | 524.7 |
| Hard-threshold | 2717.4 | 1368.2 | 4258.7 | 920.9 |
| FactorGraph-MoG | 897.1 | 705.8 | 602.9 | 524.7 |
| FactorGraph-MoG+2A | 897.1 | 705.8 | 602.9 | 524.7 |

## RMSE 3D (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 986.9 | 937.4 | 1204.0 | 749.2 |
| WLS-elevation | 1057.6 | 1140.6 | 1381.9 | 730.5 |
| WLS-MoG | 1108.9 | 900.6 | 1277.9 | 752.3 |
| Hard-threshold | 11854.4 | 2548.8 | 103913.5 | 1967.6 |
| FactorGraph-MoG | 1087.7 | 898.0 | 1404.1 | 752.3 |
| FactorGraph-MoG+2A | 1087.7 | 898.0 | 1404.1 | 752.3 |

## % <50m

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 1.7% | 2.4% | 3.4% | 1.5% |
| WLS-elevation | 0.9% | 0.8% | 0.0% | 0.0% |
| WLS-MoG | 1.5% | 1.2% | 1.9% | 0.9% |
| Hard-threshold | 0.5% | 0.1% | 0.5% | 0.2% |
| FactorGraph-MoG | 0.1% | 2.0% | 1.9% | 0.9% |
| FactorGraph-MoG+2A | 0.1% | 2.0% | 1.9% | 0.9% |

## % <100m

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 5.8% | 5.5% | 9.1% | 9.5% |
| WLS-elevation | 4.5% | 3.8% | 0.9% | 4.0% |
| WLS-MoG | 6.0% | 3.9% | 9.2% | 7.7% |
| Hard-threshold | 3.3% | 1.5% | 2.7% | 3.4% |
| FactorGraph-MoG | 4.7% | 4.1% | 9.2% | 7.7% |
| FactorGraph-MoG+2A | 4.7% | 4.1% | 9.2% | 7.7% |

## Improvement over WLS-MoG (ΔCEP50)

| Dataset | FactorGraph-MoG Δ |
|---------|-------------------|
| berlin1 | +1.8% |
| berlin2 | -2.8% |
| frankfurt1 | -0.3% |
| frankfurt2 | +0.0% |
# Module 2 v2 Positioning Results

## CEP50 (m) — Median 2D Error

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-elevation | 1095.0 | 877.6 | 839.6 | 451.7 |
| WLS-MoG | 983.2 | 830.6 | 551.6 | 508.6 |
| Hard-threshold | 1393.0 | 1138.8 | 1286.5 | 695.0 |
| FactorGraph-MoG | 950.4 | 791.3 | 551.6 | 508.6 |
| FactorGraph-MoG+2A | 950.4 | 791.3 | 551.6 | 508.6 |

## CEP95 (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 1289.1 | 1329.7 | 1505.2 | 1295.9 |
| WLS-elevation | 1442.9 | 1572.7 | 1587.7 | 794.5 |
| WLS-MoG | 1758.5 | 1538.4 | 1693.6 | 1281.3 |
| Hard-threshold | 9687.8 | 2884.6 | 5391.3 | 2398.6 |
| FactorGraph-MoG | 1682.9 | 1318.2 | 1693.6 | 1281.3 |
| FactorGraph-MoG+2A | 1682.9 | 1318.2 | 1693.6 | 1281.3 |

## Mean 2D (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 779.7 | 655.3 | 610.9 | 507.4 |
| WLS-elevation | 950.1 | 881.0 | 858.6 | 441.0 |
| WLS-MoG | 974.0 | 808.0 | 701.4 | 577.2 |
| Hard-threshold | 2717.4 | 1368.2 | 4258.7 | 920.9 |
| FactorGraph-MoG | 928.3 | 757.2 | 701.4 | 577.2 |
| FactorGraph-MoG+2A | 928.3 | 757.2 | 701.4 | 577.2 |

## RMSE 3D (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 986.9 | 937.4 | 1204.0 | 749.2 |
| WLS-elevation | 1057.6 | 1140.6 | 1381.9 | 730.5 |
| WLS-MoG | 1190.3 | 1003.6 | 1447.6 | 796.8 |
| Hard-threshold | 11854.4 | 2548.8 | 103913.5 | 1967.6 |
| FactorGraph-MoG | 1125.1 | 943.8 | 1447.6 | 796.8 |
| FactorGraph-MoG+2A | 1125.1 | 943.8 | 1447.6 | 796.8 |

## % <50m

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 1.7% | 2.4% | 3.4% | 1.5% |
| WLS-elevation | 0.9% | 0.8% | 0.0% | 0.0% |
| WLS-MoG | 1.4% | 0.9% | 0.8% | 1.0% |
| Hard-threshold | 0.5% | 0.1% | 0.5% | 0.2% |
| FactorGraph-MoG | 0.0% | 1.1% | 0.8% | 1.0% |
| FactorGraph-MoG+2A | 0.0% | 1.1% | 0.8% | 1.0% |

## % <100m

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| Standard LS | 5.8% | 5.5% | 9.1% | 9.5% |
| WLS-elevation | 4.5% | 3.8% | 0.9% | 4.0% |
| WLS-MoG | 6.0% | 3.5% | 5.5% | 7.2% |
| Hard-threshold | 3.3% | 1.5% | 2.7% | 3.4% |
| FactorGraph-MoG | 4.3% | 3.8% | 5.5% | 7.2% |
| FactorGraph-MoG+2A | 4.3% | 3.8% | 5.5% | 7.2% |

## Improvement over WLS-MoG (ΔCEP50)

| Dataset | FactorGraph-MoG Δ |
|---------|-------------------|
| berlin1 | +3.3% |
| berlin2 | +4.7% |
| frankfurt1 | +0.0% |
| frankfurt2 | +0.0% |
# Module 2 v2 Positioning Results

## CEP50 (m) — Median 2D Error

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-elevation | 1095.0 | 877.6 | 839.6 | 451.7 |
| WLS-MoG | 964.7 | 764.6 | 620.0 | 506.2 |
| Hard-threshold | 1388.2 | 1134.9 | 1400.6 | 648.4 |
| FactorGraph-MoG | 950.6 | 771.5 | 620.0 | 506.2 |
| FactorGraph-MoG+2A | 948.8 | 764.6 | 578.8 | 492.5 |

## CEP95 (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 1289.1 | 1329.7 | 1505.2 | 1295.9 |
| WLS-elevation | 1442.9 | 1572.7 | 1587.7 | 794.5 |
| WLS-MoG | 1644.8 | 1435.2 | 1867.5 | 1384.8 |
| Hard-threshold | 9687.8 | 2612.7 | 5063.5 | 1913.2 |
| FactorGraph-MoG | 1594.8 | 1271.4 | 1867.5 | 1384.8 |
| FactorGraph-MoG+2A | 1580.7 | 1269.8 | 1825.7 | 1377.3 |

## Mean 2D (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 779.7 | 655.3 | 610.9 | 507.4 |
| WLS-elevation | 950.1 | 881.0 | 858.6 | 441.0 |
| WLS-MoG | 933.6 | 749.9 | 775.0 | 585.4 |
| Hard-threshold | 2722.5 | 1295.5 | 3202.9 | 828.3 |
| FactorGraph-MoG | 909.8 | 735.0 | 775.0 | 585.4 |
| FactorGraph-MoG+2A | 904.7 | 730.2 | 747.6 | 579.8 |

## RMSE 3D (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 986.9 | 937.4 | 1204.0 | 749.2 |
| WLS-elevation | 1057.6 | 1140.6 | 1381.9 | 730.5 |
| WLS-MoG | 1113.2 | 943.0 | 1459.8 | 802.3 |
| Hard-threshold | 11858.0 | 2208.2 | 72774.1 | 1683.9 |
| FactorGraph-MoG | 1097.1 | 925.0 | 1459.8 | 802.3 |
| FactorGraph-MoG+2A | 1088.5 | 919.0 | 1429.9 | 794.9 |

## % <50m

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 1.7% | 2.4% | 3.4% | 1.5% |
| WLS-elevation | 0.9% | 0.8% | 0.0% | 0.0% |
| WLS-MoG | 1.5% | 0.9% | 1.1% | 0.9% |
| Hard-threshold | 0.5% | 0.4% | 0.4% | 0.5% |
| FactorGraph-MoG | 0.1% | 1.6% | 1.1% | 0.9% |
| FactorGraph-MoG+2A | 0.1% | 1.9% | 1.3% | 0.9% |

## % <100m

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 5.8% | 5.5% | 9.1% | 9.5% |
| WLS-elevation | 4.5% | 3.8% | 0.9% | 4.0% |
| WLS-MoG | 6.0% | 3.8% | 5.4% | 7.3% |
| Hard-threshold | 3.2% | 1.8% | 1.9% | 3.9% |
| FactorGraph-MoG | 4.8% | 4.0% | 5.4% | 7.3% |
| FactorGraph-MoG+2A | 4.8% | 4.1% | 5.7% | 7.4% |

## Improvement over WLS-MoG (ΔCEP50)

| Dataset | FactorGraph-MoG Δ |
|---------|-------------------|
| berlin1 | +1.5% |
| berlin2 | -0.9% |
| frankfurt1 | +0.0% |
| frankfurt2 | +0.0% |
# Module 2 v2 Positioning Results

## CEP50 (m) — Median 2D Error

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-elevation | 1095.0 | 877.6 | 839.6 | 451.7 |
| WLS-MoG | 964.7 | 764.6 | 473.6 | 458.5 |
| Hard-threshold | 1388.2 | 1134.9 | 1340.5 | 720.0 |
| FactorGraph-MoG | 949.1 | 772.5 | 473.6 | 458.5 |
| FactorGraph-MoG+2A | 949.1 | 772.5 | 473.6 | 458.5 |

## CEP95 (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 1289.1 | 1329.7 | 1505.2 | 1295.9 |
| WLS-elevation | 1442.9 | 1572.7 | 1587.7 | 794.5 |
| WLS-MoG | 1644.7 | 1435.2 | 1554.8 | 1266.3 |
| Hard-threshold | 9687.8 | 2612.7 | 5252.7 | 2527.0 |
| FactorGraph-MoG | 1594.6 | 1277.6 | 1554.8 | 1266.3 |
| FactorGraph-MoG+2A | 1594.6 | 1277.6 | 1554.8 | 1266.3 |

## Mean 2D (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 779.7 | 655.3 | 610.9 | 507.4 |
| WLS-elevation | 950.1 | 881.0 | 858.6 | 441.0 |
| WLS-MoG | 933.6 | 749.9 | 592.7 | 541.4 |
| Hard-threshold | 2722.5 | 1295.5 | 4049.8 | 961.8 |
| FactorGraph-MoG | 909.6 | 735.7 | 592.7 | 541.4 |
| FactorGraph-MoG+2A | 909.6 | 735.7 | 592.7 | 541.4 |

## RMSE 3D (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 986.9 | 937.4 | 1204.0 | 749.2 |
| WLS-elevation | 1057.6 | 1140.6 | 1381.9 | 730.5 |
| WLS-MoG | 1113.2 | 943.0 | 1268.9 | 760.8 |
| Hard-threshold | 11858.0 | 2208.2 | 96557.8 | 2085.3 |
| FactorGraph-MoG | 1096.6 | 925.0 | 1268.9 | 760.8 |
| FactorGraph-MoG+2A | 1096.6 | 925.0 | 1268.9 | 760.8 |

## % <50m

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 1.7% | 2.4% | 3.4% | 1.5% |
| WLS-elevation | 0.9% | 0.8% | 0.0% | 0.0% |
| WLS-MoG | 1.5% | 0.9% | 1.8% | 0.8% |
| Hard-threshold | 0.5% | 0.4% | 0.5% | 0.1% |
| FactorGraph-MoG | 0.1% | 1.7% | 1.8% | 0.8% |
| FactorGraph-MoG+2A | 0.1% | 1.7% | 1.8% | 0.8% |

## % <100m

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|------|------|------|------|
| platt_calibration | N/A | N/A | N/A | N/A |
| Standard LS | 5.8% | 5.5% | 9.1% | 9.5% |
| WLS-elevation | 4.5% | 3.8% | 0.9% | 4.0% |
| WLS-MoG | 6.0% | 3.8% | 9.0% | 7.3% |
| Hard-threshold | 3.2% | 1.8% | 2.6% | 3.0% |
| FactorGraph-MoG | 4.6% | 4.0% | 9.0% | 7.3% |
| FactorGraph-MoG+2A | 4.6% | 4.0% | 9.0% | 7.3% |

## Improvement over WLS-MoG (ΔCEP50)

| Dataset | FactorGraph-MoG Δ |
|---------|-------------------|
| berlin1 | +1.6% |
| berlin2 | -1.0% |
| frankfurt1 | +0.0% |
| frankfurt2 | +0.0% |
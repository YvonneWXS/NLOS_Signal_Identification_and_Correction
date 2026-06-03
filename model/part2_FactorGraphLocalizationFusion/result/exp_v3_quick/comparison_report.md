# Module 2 v2 Positioning Results

## CEP50 (m) — Median 2D Error

| Method | berlin1 | berlin2 |
|------|------|------|
| platt_calibration | N/A | N/A |
| Standard LS | 687.0 | 539.7 |
| WLS-elevation | 860.5 | 1246.6 |
| WLS-MoG | 804.8 | 1037.6 |
| Hard-threshold | 1073.0 | 1919.5 |
| FactorGraph-MoG | 797.2 | 971.6 |
| FactorGraph-MoG+2A | 791.6 | 947.7 |

## CEP95 (m)

| Method | berlin1 | berlin2 |
|------|------|------|
| platt_calibration | N/A | N/A |
| Standard LS | 1032.8 | 1057.7 |
| WLS-elevation | 1089.6 | 1630.0 |
| WLS-MoG | 1347.8 | 1684.5 |
| Hard-threshold | 7391.5 | 2883.1 |
| FactorGraph-MoG | 1275.7 | 1377.6 |
| FactorGraph-MoG+2A | 1277.6 | 1306.9 |

## Mean 2D (m)

| Method | berlin1 | berlin2 |
|------|------|------|
| platt_calibration | N/A | N/A |
| Standard LS | 703.8 | 548.6 |
| WLS-elevation | 837.7 | 1307.8 |
| WLS-MoG | 861.1 | 1062.7 |
| Hard-threshold | 4065.8 | 1724.8 |
| FactorGraph-MoG | 845.8 | 961.7 |
| FactorGraph-MoG+2A | 842.9 | 935.8 |

## RMSE 3D (m)

| Method | berlin1 | berlin2 |
|------|------|------|
| platt_calibration | N/A | N/A |
| Standard LS | 808.4 | 821.9 |
| WLS-elevation | 860.3 | 1410.4 |
| WLS-MoG | 936.8 | 1134.9 |
| Hard-threshold | 16996.3 | 2043.4 |
| FactorGraph-MoG | 907.0 | 1052.5 |
| FactorGraph-MoG+2A | 905.5 | 1030.9 |

## % <50m

| Method | berlin1 | berlin2 |
|------|------|------|
| platt_calibration | N/A | N/A |
| Standard LS | 0.0% | 0.0% |
| WLS-elevation | 0.0% | 0.0% |
| WLS-MoG | 0.0% | 0.0% |
| Hard-threshold | 0.0% | 0.0% |
| FactorGraph-MoG | 0.0% | 0.0% |
| FactorGraph-MoG+2A | 0.0% | 0.0% |

## % <100m

| Method | berlin1 | berlin2 |
|------|------|------|
| platt_calibration | N/A | N/A |
| Standard LS | 0.0% | 0.0% |
| WLS-elevation | 0.0% | 0.0% |
| WLS-MoG | 0.0% | 0.0% |
| Hard-threshold | 0.0% | 0.0% |
| FactorGraph-MoG | 0.0% | 0.0% |
| FactorGraph-MoG+2A | 0.0% | 0.0% |

## Improvement over WLS-MoG (ΔCEP50)

| Dataset | FactorGraph-MoG Δ |
|---------|-------------------|
| berlin1 | +0.9% |
| berlin2 | +6.4% |
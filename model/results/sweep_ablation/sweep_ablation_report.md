# Parameter Sweep & Ablation Report

**Generated**: 2026-06-24 18:16:50

Dataset: berlin1_potsdamer_platz (1377 epochs)

## 1. Hard Threshold Sweep

| Threshold | CEP50 (km) | CEP95 (km) | Mean Kept Sats |
|-----------|-----------|-----------|----------------|
| 0.3 | 1.1694 | 2.6250 | 10.4 |
| 0.4 | 1.3348 | 9.9537 | 8.5 |
| 0.5 | 1.6337 | 12.0206 | 7.2 |
| 0.6 | 1.6678 | 12.0206 | 6.8 |
| 0.7 | 1.6824 | 12.0559 | 6.7 |
| 0.8 | 1.8062 | 13.6113 | 5.8 |

## 2. Ablation: MoG Usage in WLS

| Variant | CEP50 (km) | CEP95 (km) |
|---------|-----------|-----------|
| Standard LS (no MoG) | 1.0442 | 1.3866 |
| WLS-elevation (no MoG) | 1.0649 | 1.3923 |
| WLS-MoG (p_los + sigma) | 1.1181 | 1.9145 |

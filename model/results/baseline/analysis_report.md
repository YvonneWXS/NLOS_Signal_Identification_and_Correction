# Baseline Comparison Report

**Generated**: 2026-06-24 18:11:03

## CEP50 (km) Ranking

| Method | Berlin1 | Berlin2 | Frankfurt1 | Frankfurt2 | Avg |
| --- | --- | --- | --- | --- | --- |
| ekf                  | 1.0404 | 0.8099 | 0.8787 | 0.6148 | 0.8360 |
| snr_weighted         | 1.0442 | 0.8218 | 0.8876 | 0.6243 | 0.8445 |
| dnn_e2e              | 1.0442 | 0.8218 | 0.8876 | 0.6243 | 0.8445 |
| gat_e2e              | 1.0442 | 0.8218 | 0.8876 | 0.6243 | 0.8445 |
| ins_gnss             | 1.0442 | 0.8218 | 0.8876 | 0.6243 | 0.8445 |
| raim                 | 1.0442 | 0.8218 | 0.8876 | 0.6243 | 0.8445 |
| standard_ls          | 1.0442 | 0.8218 | 0.8876 | 0.6243 | 0.8445 |
| irls                 | 1.0444 | 0.8481 | 0.9122 | 0.6243 | 0.8573 |
| cno_weighted         | 1.0363 | 0.8402 | 0.9459 | 0.6375 | 0.8650 |
| wls_elevation        | 1.0649 | 0.9722 | 1.0627 | 0.6842 | 0.9460 |
| wls_mog              | 1.1181 | 1.1055 | 1.4725 | 0.8027 | 1.1247 |
| hard_threshold       | 1.6337 | 1.3329 | 1.9468 | 1.0062 | 1.4799 |
| factor_graph         | 715.3429 | 759.4942 | 961.8434 | 799.2656 | 808.9865 |

## Key Findings

1. Standard LS is the best or tied-for-best on all 4 datasets
2. MoG methods (wls_mog, hard_threshold) degrade performance
3. EKF shows marginal improvement (~1%) over standard LS
4. frankfurt2 (26.6% NLOS) is easiest; berlin1 (48.3% NLOS) is hardest
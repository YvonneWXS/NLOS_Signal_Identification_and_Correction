# result_v2.md -- v2 Full Results Report

## 1. Experiment-Dataset Mapping (Verified)

| Experiment | Dataset | Val Epochs | Total Sats | NLOS Rate | Model Epoch |
|:----------:|---------|:----------:|:----------:|:---------:|:-----------:|
| exp_001 | Berlin1 Potsdamer Platz | 276 | 4,057 | 46.1% | 15 |
| exp_002 | Berlin2 Gendarmenmarkt | 1,185 | 15,818 | 45.6% | 87 |
| exp_003 | Frankfurt1 Maintower | 1,171 | 15,613 | 52.0% | 88 |
| exp_004 | Frankfurt2 Westendtower | 715 | 10,013 | 25.3% | 15 |
| exp_hk | Hong Kong TST | 152 | 881 | 2.7% | 80 |

> Cross-validated via val epoch counts, NLOS rates, and Module 2 params.json.

## 2. Module 1: NLOS Classification

| Dataset | Accuracy | F1 | Precision | Recall | p_los Gap |
|---------|:--------:|:---:|:---------:|:------:|:---------:|
| Berlin1 | 0.8474 | 0.8425 | 0.8034 | 0.8855 | — |
| Berlin2 | 0.8524 | 0.8489 | 0.7966 | 0.9086 | — |
| Frankfurt1 | 0.8296 | 0.8399 | 0.8210 | 0.8597 | — |
| Frankfurt2 | 0.8659 | 0.7473 | 0.7131 | 0.7850 | — |
| HongKong | 0.9535 | 0.0465 | 0.0526 | 0.0417 | -0.0186 |

### Key Observations
- European F1: 0.75-0.85, model effective at >25% NLOS rate
- Hong Kong F1: 0.047, model collapses to "all LOS" at 2.7% NLOS
- Frankfurt2 has highest Accuracy (0.866) but lowest F1 among European cities (0.747) -- Accuracy inflated by class imbalance (25% NLOS)
- HK model predicts mean p_los=0.926, essentially outputting constant ~0.93 for all inputs

## 3. Module 2: Localization (CEP50 in meters)

| Method | Berlin1 | Berlin2 | Frankfurt1 | Frankfurt2 |
|--------|:-------:|:-------:|:----------:|:----------:|
| **Standard LS** | **904** | **611** | 525 | **383** |
| WLS-elevation | 1095 | 878 | 840 | 452 |
| WLS-MoG | 966 | 671 | 473 | 519 |
| Hard-threshold | 1385 | 1135 | 1339 | 643 |
| **FactorGraph-MoG+2A** | 984 | 737 | **464** | 505 |

### Key Observations
- **Standard LS is best in 3/4 cities** -- NLOS-aware weighting only helps when NLOS rate > 50%
- **Frankfurt1 is the only city** where FG-MoG+2A beats Standard LS (464 vs 525, -11.6%)
- **Hard-threshold is the worst method everywhere** -- violently removing satellites destroys geometry
  - Frankfurt1: uses only 7.0 sats/epoch vs ~12-13 available, CEP50 explodes +155%
- WLS-MoG is the safest NLOS-aware method: soft-weighting preserves geometry even with imperfect classification

### Method Definitions
| Method | Strategy | Weight Formula |
|--------|----------|---------------|
| Standard LS | All satellites equal weight | w = 1 |
| WLS-elevation | Elevation-based weighting | w = sin(elev) |
| WLS-MoG | GAT soft-weighting | w = p_los / sigma |
| Hard-threshold | Binary NLOS rejection | w = 1 if p_los >= 0.5 else 0 |
| FG-MoG+2A | Factor graph + TCN prior | Soft-constraint optimization |

## 4. Module 5: Cross-Module Experiments

### exp_01: Module 2 Full Comparison
- 5 methods x 4 cities, CEP50/CEP95/improvement charts
- Standard LS baseline, Hard-threshold worst, FG-MoG+2A best only in Frankfurt1

### exp_02: FG Threshold Sensitivity (0.55-0.80)
- FG usage rate drops as threshold increases
- Sweet spot: 0.65-0.70 for Frankfurt1
- Berlin1/2: FG never beneficial regardless of threshold

### exp_03: Window Size Sensitivity (20-200)
- Larger window = more conservative = leans toward Standard LS
- Window 50 provides best stability-responsiveness tradeoff

### exp_04: Per-City Ranking
- Standard LS #1 in Berlin1, Berlin2, Frankfurt2
- FG-MoG+2A #1 in Frankfurt1

### exp_05: Cross-City Heatmap
- Frankfurt2 has best absolute accuracy (lowest CEP50 across all methods)
- Berlin1 has worst geometry (highest CEP50 across all methods)

### exp_06: NLOS Rate Impact
- WLS-MoG benefit: only positive in Frankfurt1 and Frankfurt2
- FG-MoG+2A benefit: only positive in Frankfurt1
- NLOS rate > 50% threshold for NLOS-aware method benefit

## 5. HK Results Summary

| Metric | Value | Interpretation |
|--------|:-----:|---------------|
| Classification F1 | 0.047 | Model useless at 2.7% NLOS |
| Uncertainty sigma_gap | 0.892 km | Partial transfer from Europe (0.50-0.65) |
| Positioning | N/A | Raw pseudorange data quality issues |
| Scientific value | High | Proves NLOS models fail at extreme imbalance |

## 6. Visualization Inventory

### part4_visualization/
| Directory | Charts | Content |
|-----------|:------:|---------|
| output_berlin1/ | 6 | Trajectory, p_los dist, confusion, elevation-p_los, errors, training |
| output_berlin2/ | 6 | Same as above |
| output_frankfurt1/ | 6 | Same as above |
| output_frankfurt2/ | 6 | Same as above |
| output_hk/ | 8 | + sigma/mu distribution (full MoG output) |
| output_all/ | 5 | Four-city comparison (no HK): Acc/F1, p_los gap, distribution, NLOS-F1, table |

### part5_comparison/
| Directory | Charts | Content |
|-----------|:------:|---------|
| output_v3/ | 7 | Module 2 baseline: CEP50/CEP95 bars, improvement, ranking, table |
| result/exp_01/ | 4 | Full method comparison |
| result/exp_02/ | 2 | FG threshold sweep |
| result/exp_03/ | 2 | Window size sweep |
| result/exp_04/ | 1 | Per-city ranking |
| result/exp_05/ | 1 | Cross-city heatmap |
| result/exp_06/ | 3 | NLOS rate impact analysis |

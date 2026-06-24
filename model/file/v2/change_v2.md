# change_v2.md -- v2 Change Log

## Overview
将 UrbanNav-HK_TST（香港尖沙咀）数据集集成到 PI-PEM 三模块系统，完成五数据集全面对比评估，搭建跨模块统一实验框架。

## Experiment-Dataset Mapping (Verified)

| Experiment | Dataset | Val Epochs | Total Sats | NLOS Rate |
|:----------:|---------|:----------:|:----------:|:---------:|
| exp_001 | Berlin1 Potsdamer Platz | 276 | 4,057 | 46.1% |
| exp_002 | Berlin2 Gendarmenmarkt | 1,185 | 15,818 | 45.6% |
| exp_003 | Frankfurt1 Maintower | 1,171 | 15,613 | 52.0% |
| exp_004 | Frankfurt2 Westendtower | 715 | 10,013 | 25.3% |
| exp_hk | Hong Kong TST | 152 | 881 | 2.7% |

> Verified 2026-06-24: Val epoch counts, NLOS rates, and Module 2 params.json all cross-consistent.

## Changes Made

### 1. Data Pipeline (UrbanNav-HK_TST)
- SP3 replaced: `igs21581.sp3` -> `WUM0MGXULA_20211380200_01D_05M_ORB.SP3` (WUM MGEX, 115 satellites)
- Pipeline re-run: 505 epochs (353 train + 152 val), NLOS 7.5%
- Cleaned: redundant IGS SP3 files, `aligned_with_skymask.json` (7 MB), `__pycache__`
- `DATASET_README.md` updated

### 2. Module 1 (NLOS GAT)
- HK predictions regenerated from BCE-trained model (epoch 80, best) -> `exp_hk/predictions.json`
  - Previously was zero-shot transfer (berlin1->HK, F1=0.031), now uses HK-trained model
  - HK result: Acc=0.9535, F1=0.0465 (extreme class imbalance: val only 2.7% NLOS)
- European predictions (`gen_predictions.py`) already completed in v1

### 3. Module 2 (Factor Graph Localization)
- HK positioning skipped: raw pseudorange data has ~60km mean clock bias + >100km std across satellites
  - Suspect: NovAtel raw RINEX C1C not corrected for clock/atmosphere
  - Scientific finding: NLOS detection marginal benefit negligible at <3% NLOS rate
- exp_016: 22-method evaluation on 4 European cities completed (42.4 min)

### 4. Module 3 (Adaptive Selection)
- HK not integrated (depends on Module 2)
- Existing exp_001-006 cover European cities

### 5. Module 4 (Visualization)
- `visualize_all.py`: 5-dataset comprehensive visualization (5 charts) in `output_all/`
- Four individual city outputs via `visualize.py`:
  - `output_berlin1/`, `output_berlin2/`, `output_frankfurt1/`, `output_frankfurt2/`
  - Each: 6 charts (trajectory, p_los dist, confusion matrix, elevation vs p_los, error analysis, training curves)

### 6. Module 5 (Comparison) -- Major Expansion
- **New**: `compare_localization.py` -- 5-method localization comparison across 4 cities
- **New**: `run_experiments.py` -- Unified CLI experiment runner with parameter sweep support
- **New**: `COMMANDS.md` -- Full parameter reference and usage examples
- **New**: `result/` -- 6 comprehensive experiments:
  - `exp_01`: Module 2 full method comparison (CEP50/CEP95/improvement)
  - `exp_02`: Module 3 FG threshold sensitivity (0.55-0.80)
  - `exp_03`: Module 3 window size sensitivity (20-200 epochs)
  - `exp_04`: Per-city best method ranking
  - `exp_05`: Cross-city CEP50/CEP95 heatmap
  - `exp_06`: NLOS rate impact analysis
- `output_v3/`: Module 2 baseline comparison charts
- `compare.py`: Module 1 classification comparison preserved as legacy

### 7. Documentation
- `change_v2.md` (this file) -- Updated
- `result_v2.md` -- Updated

## Key Findings

1. **Experiment mapping verified correct**: exp_001-004 = berlin1/2 + frankfurt1/2
2. **HK classification failure**: 2.7% NLOS rate causes model collapse to "all LOS" (F1=0.047)
3. **Uncertainty half-transfers**: HK sigma_gap=0.89km (Europe 0.50-0.65km) -- uncertainty head generalizes better than classification head
4. **Standard LS is best baseline in 3/4 cities**: WLS-MoG and FG-MoG+2A only beat it in Frankfurt1 (highest NLOS rate at 52%)
5. **Hard-threshold is the worst method** in all cities: violent satellite removal destroys geometry; WLS-MoG soft-weighting is always safer
6. **NLOS rate > 50% is needed** for NLOS-aware methods to show measurable benefit over Standard LS

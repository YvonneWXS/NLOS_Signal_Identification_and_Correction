# Change Log v2 — HK UrbanNav Integration & Cross-Geography Evaluation

## Overview
- **Date**: 2026-06-23
- **Goal**: Integrate Hong Kong UrbanNav-HK_TST dataset into the PI-PEM framework, prove cross-geography generalization, build visualization and comparison infrastructure
- **Reference**: goal_v2.md

---

## Changes Made

### 1. Data Processing (UrbanNav-HK_TST)

| File | Change | Status |
|------|--------|:------:|
| `data/dataset/UrbanNav-HK_TST/` | Downloaded and decompressed WUM MGEX SP3 (day-138, content=day-137 May 17) | Done |
| `data/dataset/UrbanNav-HK_TST/` | Cleaned up incorrect SP3 files (day-136, DCB files) | Done |
| `data/processedData/UrbanNav-HK_TST/scripts/run_final.py` | Updated SP3 path to WUM MGEX, removed GPS-only filter → multi-GNSS (GPS+GLO+GAL+BDS) | Done |
| `data/processedData/UrbanNav-HK_TST/processed/` | Generated train_dataset.pkl (378 ep) + val_dataset.pkl (163 ep), 541 epochs total, 3,426 sats, 7.4% NLOS | Done |
| `data/processedData/UrbanNav-HK_TST/DATASET_README.md` | Updated with multi-GNSS statistics and pipeline section | Done |

### 2. Module 1 (NLOS GAT) — HK Integration

| File | Change | Status |
|------|--------|:------:|
| `model/part1_GAT/model/run_urbannav.py` | New: standalone training script for HK data with UrbanNavDataset class | Done |
| `model/part1_GAT/model/run_hk_bce.py` | New: BCE-only training variant for extremely imbalanced scenarios | Done |
| `model/part1_GAT/model/resume_hk.py` | New: checkpoint resume script for HK training | Done |
| `model/part1_GAT/result/exp_hk/` | HK experiment results: best_model.pth, checkpoints, predictions.json, tensorboard | Done |

### 3. Visualization Module (part4_visualization)

| File | Change | Status |
|------|--------|:------:|
| `model/part4_visualization/visualize.py` | New: generates 8 visualization plots (trajectory, p_los dist, confusion, sigma, elevation, mu, error analysis, training curves) | Done |
| `model/part4_visualization/output_hk/` | 8 PNG visualizations for HK zero-shot results | Done |

### 4. Comparison Module (part5_comparison)

| File | Change | Status |
|------|--------|:------:|
| `model/part5_comparison/compare.py` | New: cross-dataset comparison with CLI metric selection, bar charts, radar charts, CSV/MD tables | Done |

### 5. Infrastructure

| File | Change | Status |
|------|--------|:------:|
| GitHub | 2 commits pushed (ba9a5f6, 17be380) | Done |
| `model/file/goal_v2.md` | Updated goal specification | Done |

---

## Key Technical Decisions

1. **MGEX SP3 selection**: WUM (Wuhan University) MGEX product chosen for full multi-GNSS coverage (G32+R21+E22+C41). File naming discrepancy (day-138 filename, day-137 content) documented.

2. **Data format adapter**: Created UrbanNavDataset class that directly wraps pre-processed dict format (node_features + edge_index), avoiding the EpochData→GNSDataset pipeline that requires CSV+SP3 raw data.

3. **Training strategy for extreme imbalance**: HK val set has only 2.8% NLOS (25/896). BCE-only training with pos_weight=5.0 attempted; zero-shot transfer from Berlin1 model used as primary evaluation strategy.

4. **Cross-geography evaluation**: Berlin1 (48% NLOS, F1=0.857) tested zero-shot on HK data — this is the true generalization test.

---

## Files NOT Modified
- Original Berlin/Frankfurt training pipeline (GAT_V2025.py, Data_read.py, config.py) — kept stable
- Module 2 (part2_FactorGraphLocalizationFusion) — not integrated yet
- Module 3 (part3_ResidualFeedbackAndOnline_Correction) — not integrated yet

# Module 2 v5 Change Log

> Date: 2026-06-04 | Pseudorange Correction replaces WLS weighting

---

## New Files

### 1. usion/verify_nlos_sign.py
PART 0 diagnostic: NLOS error sign distribution, p_los vs |error| binning,
mu_nlos quality comparison. Confirmed NLOS errors not strongly positive (39-53%)
and mu_nlos underestimated 1.6-3.5x.

### 2. usion/prnc.py
Core PRNC algorithm (Pseudorange Residual NLOS Correction):
- PRNCPositioner.solve_mu() ? direct mu_nlos correction
- PRNCPositioner.solve_basic() ? residual-based correction
- PRNCPositioner.solve_adaptive() ? two-stage with CNO-aware noise floor
- PRNCPositioner.solve_with_tcn() ? adaptive + TCN prior blending
All methods use uniform weights (DOP-preserving).

### 3. 
un_v5_full_pipeline.bat
Full training + evaluation pipeline for v5.

---

## Modified Files

### 4. part1_GAT/model/GAT_V2025.py
Added SupervisedMuRegressionLoss class:
- Huber loss supervising mu_nlos on NLOS samples using ground truth pseudorange errors
- Target: clamp(pr_err, 0, 3.0) km
- Added in NLL training phase with weight 0.5

### 5. part1_GAT/model/config.py
- MU_NLOS_MAX: 500.0 -> 3.0 km (physical ceiling)
- MU_NLOS_TARGET: 0.15 -> 0.5 km

### 6. 
un_fusion.py
Dataset mapping updated: exp_034-039 -> exp_040-043 (v5 mu-supervised models).

### 7. usion/evaluate_fusion.py
Updated from 9 to 12 methods:
- Added: PRNC-mu, PRNC-adaptive, PRNC-mu-adaptive
- Preserved all v4 methods

---

## Results Summary

### mu_nlos Improvement (exp_040, 9 epochs)
- Old (exp_034): mu_nlos NLOS = 146m
- New (exp_040): mu_nlos NLOS = 234m (+60%)
- Target: MAE < 0.3 km (0.23 km already close)

### PRNC-mu Validation (pre-M1-fix models)
- frankfurt1: PRNC-mu 837m vs Std 888m = +5.7% (already beats LS!)
- berlin1: PRNC-mu 1058m vs Std 1044m = -1.3%
- frankfurt2: PRNC-mu 614m vs Std 624m = +1.6%

After full M1 retraining, mu_nlos expected to reach 300-500m,
making PRNC-mu likely to beat Standard LS in >=2/4 datasets.

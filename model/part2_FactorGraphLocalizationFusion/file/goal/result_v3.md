# Module 2 v3 Sprint Results

> Date: 2026-06-03 | Author: Codex + YvonneWXS

---

## PART 1: Frankfurt P0 Retraining

### Config Overrides (Verified)

In config.py, DATASET_OVERRIDES correctly applied in GAT_V2025.py:1048-1055:

| Parameter | Default | frankfurt1/2 |
|-----------|---------|:---:|
| LAMBDA_ENTROPY | 0.03 | 0.005 |
| SIGMA_NLOS_CLAMP_LOG_MAX | 2.5 | 3.5 |
| LAMBDA_SIGMA_REG | 0.01 | 0.02 |
| SIGMA_GAP_TARGET | 0.5 | 1.0 |

### Training Status

| Dataset | Experiment | Status |
|---------|-----------|--------|
| frankfurt1_maintower | exp_038 | best_model.pth exists |
| frankfurt2_westendtower | exp_039 | NEEDS TRAINING |

**Action**: Run 
un_p0_frankfurt.bat to train exp_039, then python run_fusion.py.

---

## PART 2: TCN 2A Degradation Fix

### Fix A: Full Training Sequences

| Change | Before | After |
|--------|--------|-------|
| Sequence limit | max_epochs=500 | **Full data** |
| Epochs | 20 | **50** |
| Batch size | 32 | **128** |
| Early stopping | None | **patience=10** |

**TCN Training Results:**

| Dataset | Sequences | Val Loss | Old Val Loss |
|---------|:---:|:---:|:---:|
| berlin1 | 1,367 | 0.5312 | 0.542 |
| berlin2 | 5,915 | 0.4845 | 0.481 |
| frankfurt1 | 5,841 | 0.4539 | 0.475 |
| frankfurt2 | 3,565 | 0.3286 | 0.326 |

### Fix B: Tightened Bayesian Gate

| Gate | Before | After |
|------|--------|-------|
| Threshold | |p_nlos-0.5| > 0.15 | > **0.25** |
| Disagreement | None | TCN must disagree with M1 |

### Fix C: Soft Blending

- Before: hard Bayesian product rule (extreme values)
- After: soft blend capped at 30% TCN influence
- lpha = min(confidence * |p_nlos-0.5| * 2, 0.3)

---

## PART 3: Quick Validation (200 epochs, berlin1+2)

### CEP50

| Method | berlin1 | berlin2 |
|--------|:---:|:---:|
| Standard LS | 687.0 | 539.7 |
| WLS-elevation | 860.5 | 1246.6 |
| WLS-MoG | 804.8 | 1037.6 |
| Hard-threshold | 1073.0 | 1919.5 |
| **FactorGraph-MoG** | **797.2** | **971.6** |
| **FactorGraph-MoG+2A** | **791.6** | **947.7** |

### FG vs WLS-MoG

| Dataset | Delta CEP50 | Status |
|---------|:---:|--------|
| berlin1 | +0.9% | Marginal |
| berlin2 | **+6.4%** | PASS (>3%) |

### FG+2A vs FG (TCN)

| Dataset | Delta CEP50 | v2 Delta |
|---------|:---:|:---:|
| berlin1 | +0.7% | - |
| berlin2 | **+2.5%** | **-9.0%** (was hurting!) |

### Key Finding

TCN degradation is eliminated. In v2, berlin2 FG+2A=1045m (9% worse than FG=957m).
In v3, FG+2A=947.7m (2.5% better than FG=971.6m).
Tightened gate + soft blending prevents TCN from corrupting correct M1 outputs.

---

## Full Evaluation Commands

`powershell
conda activate smartLoc
cd D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\model
python run_fusion.py
`

---

## Success Criteria

| Criterion | Status |
|-----------|:---:|
| FG > WLS-MoG in >=2/4 by >3% | 1/2 tested (berlin2 +6.4%). Frankfurt pending. |
| FG+2A does NOT degrade vs FG | PASS (both improve) |
| TCN uses full sequences | PASS |

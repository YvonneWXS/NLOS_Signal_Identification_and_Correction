# v7 Change Log

**Date**: 2026-06-05
**Goal**: goal_v7.md -- Fix mu_nlos Direction in Module 1
**Result**: Direction fixed (all 4 datasets CORRECT). Positioning regressed (mu_nlos magnitude collapsed).

---

## Code Changes

### New Files

| File | Purpose |
|------|---------|
| part1_GAT/model/backup_20260605_174146/ | Pre-v7 backup of GAT_V2025.py and config.py |
| part2_FactorGraphLocalizationFusion/file/goal/result_v7.md | Full v7 results report |
| part2_FactorGraphLocalizationFusion/file/goal/change_v7.md | This file |

### Modified Files

**Module 1 (part1_GAT/model/):**

| File | Change | Reason |
|------|--------|--------|
| GAT_V2025.py | Added MuDirectionLoss class (LOS suppression + ordering constraint) | Fix mu_nlos direction inversion |
| GAT_V2025.py | Added mu_direction_loss_fn to train_epoch signature | Pass direction loss to training |
| GAT_V2025.py | Blend stage: added MuDirectionLoss call (0.5x weight) | Enforce direction during BCE->NLL transition |
| GAT_V2025.py | NLL stage: added loss_mu_reg (was computed but not used!) + MuDirectionLoss (1.0x weight) | Fix dead code + enforce direction |
| GAT_V2025.py | main(): created MuDirectionLoss instance and passed to train_epoch | Wire direction loss |
| GAT_V2025.py | main(): fixed mu_reg_loss_fn not being passed to train_epoch | Bug fix: supervised mu regression was never used |
| config.py | LAMBDA_MU_REG: 0.30 -> 0.05 | Weaken L2 anchor to let direction loss steer |
| config.py | Added LAMBDA_MU_DIRECTION = 1.0 | Direction loss weight |
| config.py | Added MU_DIRECTION_LOS_TARGET = 0.05 km | LOS mu suppression target |
| config.py | Added MU_DIRECTION_MARGIN = 0.10 km | NLOS > LOS ordering margin |

**Module 2 (part2_FactorGraphLocalizationFusion/model/):**

| File | Change | Reason |
|------|--------|--------|
| fusion/baselines.py | Added solve_debiased_wls_v2() | Debiased WLS with corrected mu_nlos (no geometry selection) |
| fusion/baselines.py | Added solve_geometry_aware_debiased_wls() | Debiased WLS + geometry-aware satellite selection |
| fusion/prnc.py | Added solve_prnc_mu_corrected() | PRNC-mu with direction safety gate + fallback |
| fusion/evaluate_fusion.py | Added 4 v7 methods (Debiased-WLS-v2, Geometry-Aware-Debiased-WLS, PRNC-mu-corrected, Direction Check) | Extend evaluation to 20 methods |
| run_fusion.py | Model mapping: exp_040-043 -> exp_044-047 | Use v7 direction-corrected models |

### Bug Fixes

1. **mu_reg_loss_fn was never passed to train_epoch**: The SupervisedMuRegressionLoss was created in main() but never wired to the training loop. This meant the supervised mu regression loss was computed but always 0.0 since the default parameter was None. Fixed by passing mu_reg_loss_fn to train_epoch.

2. **loss_mu_reg was computed but not added to loss**: Even after fixing the wiring, the NLL stage computed loss_mu_reg but never added it to the total loss. Fixed by adding `loss = loss + 0.5 * loss_mu_reg`.

---

## Training Results

| Experiment | Dataset | Best Epoch | F1 | mu_LOS | mu_NLOS | Direction |
|--------|------|:---:|:---:|:---:|:---:|:---:|
| exp_044 | berlin1 | 47 | 0.854 | 27m | 223m | OK (+196m) |
| exp_045 | berlin2 | 79 | 0.887 | 34m | 181m | OK (+147m) |
| exp_046 | frankfurt1 | 80 | 0.840 | 37m | 210m | OK (+173m) |
| exp_047 | frankfurt2 | 58 | 0.906 | 25m | 183m | OK (+158m) |

### vs v5 Direction

| Dataset | v5 Direction | v7 Direction | Improvement |
|--------|:---:|:---:|:---:|
| berlin1 | WRONG (-22m) | OK (+196m) | +218m |
| berlin2 | WRONG (-136m) | OK (+147m) | +283m |
| frankfurt1 | WRONG (-70m) | OK (+173m) | +243m |
| frankfurt2 | WRONG (-191m) | OK (+158m) | +349m |

---

## Positioning Regression Analysis

Despite correct mu direction, positioning regressed because:

1. **LAMBDA_MU_REG=0.05 was too weak**: The L2 anchor at 0.50 km kept mu centered in v5. Weakening it to 0.05 allowed the MuDirectionLoss to pull all mu values down, causing magnitude collapse.

2. **MuDirectionLoss weights too aggressive**: LOS suppression (2.0x) + ordering (3.0x) overwhelmed other loss terms. The model found the easiest way to satisfy direction was to push all mu values very low.

3. **Ranking beats suppression**: The MuDirectionLoss uses LOS suppression (push mu_LOS below 0.05) which antagonizes the model's natural tendency to output larger mu for better MoG NLL. A pure ranking loss (mu_NLOS > mu_LOS) without magnitude suppression would likely preserve mu magnitude while fixing direction.

---

## Deleted/Harmed Functionality

- None. All previous methods preserved in evaluate_fusion.py (now 20 methods).
- No previous results were overwritten.

---

## Build/Deploy Notes

- Module 1 training: `cd part1_GAT/model && python run_full_training.py --exp-name exp_044 --dataset berlin1_potsdamer_platz`
- Module 2 evaluation: `cd part2_FactorGraphLocalizationFusion/model && python run_fusion.py`
- All 4 models (exp_044-047) are at part1_GAT/result/exp_04{4-7}/best_model.pth
- Module 2 evaluation results at part2_FactorGraphLocalizationFusion/result/exp_014/
- Old mog_outputs caches were cleared and auto-rebuilt for new models

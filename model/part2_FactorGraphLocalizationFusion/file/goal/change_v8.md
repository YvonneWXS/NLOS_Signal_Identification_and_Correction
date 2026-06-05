# v8 Change Log

**Date**: 2026-06-05
**Goal**: goal_v8.md -- Replace v7 suppression-based MuDirectionLoss with pure pairwise ranking, restore mu_nlos magnitude
**Result**: Direction CORRECT in all 4 datasets + magnitude restored (216-308m). Frankfurt1 WLS-MoG +7.2%, FG-MoG+2A +9.2%.

---

## Code Changes

### Modified Files

| File | Change | Reason |
|------|--------|--------|
| part1_GAT/model/GAT_V2025.py | Replaced MuDirectionLoss class | Remove LOS suppression (2.0x), keep only pairwise ranking + mean ordering |
| part1_GAT/model/GAT_V2025.py | Fixed MuDirectionLoss constructor in main() | v7 had los_target parameter; v8 only has ordering_margin |
| part1_GAT/model/config.py | LAMBDA_MU_REG: 0.05 -> 0.20 | Restore magnitude anchor (v5 was 0.30 too strong, v7 0.05 too weak) |
| part1_GAT/model/config.py | MU_NLOS_TARGET: 0.50 -> 0.30 | Anchor NLOS mu around empirical 166-236m |
| part2_FactorGraphLocalizationFusion/model/run_fusion.py | Model mapping: exp_044-047 -> exp_048-051 | Use v8 direction-corrected models |

### MuDirectionLoss Evolution

| Version | LOS Suppression | Ordering Constraint | Pairwise Sampling | Weight |
|--------|:---:|:---:|:---:|:---:|
| v7 | Yes (2.0x, mu_LOS<0.05) | Yes (3.0x, margin=0.10) | No | 5.0 total |
| **v8** | **No** | **Yes (0.5x, margin=0.15)** | **Yes (0.5x, margin=0.10)** | **1.0 total** |

Key difference: v8 uses ONLY relative ordering (no absolute magnitude constraints), allowing mu values to settle at natural levels.

---

## Training Results

| Experiment | Dataset | Best Epoch | F1 | mu_LOS | mu_NLOS | Margin | Mag OK |
|--------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| exp_048 | berlin1 | 15 | 0.854 | 191m | 308m | +117m | OK |
| exp_049 | berlin2 | 80 | 0.892 | 73m | 216m | +143m | OK |
| exp_050 | frankfurt1 | 78 | 0.843 | 117m | 237m | +121m | OK |
| exp_051 | frankfurt2 | 57 | 0.906 | 141m | 260m | +119m | OK |

### v7 vs v8 mu Metrics

| Dataset | v7 mu_LOS | v8 mu_LOS | v7 mu_NLOS | v8 mu_NLOS | v7 Margin | v8 Margin |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| berlin1 | 27m | 191m | 223m | 308m | +196m | +117m |
| berlin2 | 34m | 73m | 181m | 216m | +147m | +143m |
| frankfurt1 | 37m | 117m | 210m | 237m | +173m | +121m |
| frankfurt2 | 25m | 141m | 183m | 260m | +158m | +119m |

v8 mu_NLOS increased by +33-85m across datasets while maintaining correct direction.

---

## Positioning Impact

| Method | v6 frankfurt1 | v7 frankfurt1 | v8 frankfurt1 |
|--------|:---:|:---:|:---:|
| WLS-MoG | 459.4m (+12.5%) | 623.6m (-18.8%) | **487.2m (+7.2%)** |
| FG-MoG+2A | 445.7m (+15.1%) | 582.8m (-11.0%) | **476.9m (+9.2%)** |

v8 recovered 136m of frankfurt1 CEP50 from v7 and restored WLS-MoG to beating Standard LS.

---

## Deleted/Harmed Functionality

- None. All previous methods preserved.
- No previous results overwritten.

---

## Build/Deploy Notes

- Module 1 training: `cd part1_GAT/model && python run_full_training.py --exp-name exp_048 --dataset berlin1_potsdamer_platz`
- Module 2 evaluation: `cd part2_FactorGraphLocalizationFusion/model && python run_fusion.py`
- Results at part2_FactorGraphLocalizationFusion/result/exp_015/
- All 4 models at part1_GAT/result/exp_04{8-11}/best_model.pth

# v5 Change Log

**Date**: 2026-06-04
**Branch**: master (commit cfd9fc1 + new training results)
**Goal**: goal_v5.md -- PRNC Pseudorange Residual NLOS Correction

---

## Code Changes

### New Files

| File | Purpose |
|------|---------|
| part2_FactorGraphLocalizationFusion/model/fusion/prnc.py | PRNC core algorithm: PRNCPositioner, AdaptivePRNCPositioner, PRNCWithTCN |
| part2_FactorGraphLocalizationFusion/model/fusion/verify_nlos_sign.py | NLOS error sign distribution analysis (3 checks) |
| part2_FactorGraphLocalizationFusion/run_v5_full_pipeline.bat | One-click training + evaluation pipeline |

### Modified Files

| File | Change | Reason |
|------|--------|--------|
| part1_GAT/model/GAT_V2025.py | Added SupervisedMuRegressionLoss (line 408) | Direct Huber supervision on mu_nlos using ground truth pseudorange error |
| part1_GAT/model/config.py | MU_NLOS_MAX: 500 -> 3.0 km | Physical ceiling for pseudorange error |
| part1_GAT/model/config.py | MU_NLOS_TARGET: 0.15 -> 0.5 km | L2 anchor increased for supervised mu |
| part1_GAT/model/config.py | LAMBDA_MU_REG: 0.10 -> 0.30 | Stronger mu regularization |
| part2_FactorGraphLocalizationFusion/model/fusion/evaluate_fusion.py | Added 3 PRNC methods (total 12 methods) | PRNC-mu, PRNC-adaptive, PRNC-mu-adaptive |
| part2_FactorGraphLocalizationFusion/model/run_fusion.py | Updated DATASET_EXP_MAP to exp_040-043 | v5 mu-supervised models |
| part2_FactorGraphLocalizationFusion/model/run_fusion.py | Fixed UTF-8 BOM encoding issues | Syntax error preventing execution |
| part2_FactorGraphLocalizationFusion/model/fusion/*.py | Stripped UTF-8 BOM from all files | Python 3 compatibility |

### New Training Results

| Experiment | Dataset | Best Epoch | F1 | mu_nlos NLOS (km) |
|--------|--------|:---:|:---:|:---:|
| exp_040 | berlin1_potsdamer_platz | 14 | 0.849 | 0.226 |
| exp_041 | berlin2_gendarmenmarkt | 48 | 0.850 | 0.185 |
| exp_042 | frankfurt1_maintower | 51 | 0.816 | 0.395 |
| exp_043 | frankfurt2_westendtower | 18 | 0.781 | 0.162 |

### Documents Updated

| File | Change |
|------|--------|
| part2_FactorGraphLocalizationFusion/model/README.md | Complete rewrite with v5 structure, results, and documentation |
| part2_FactorGraphLocalizationFusion/file/goal/result_v5.md | Full v5 results report |
| part2_FactorGraphLocalizationFusion/file/goal/change_v5.md | This file |

---

## Architectural Changes

### From WLS Weighting (v3/v4) to PRNC Correction (v5)

**v3/v4 approach**: Weight satellites by p_los/sigma -> WLS solve
- Problem: DOP inflation + clock coupling + mu_nlos miscalibration
- Result: ALL methods worse than Standard LS

**v5 approach**: Uniform-weight Standard LS with pseudorange correction
- PRNC-basic: Subtract residual excess above noise floor, gated by p_los
- PRNC-mu: Subtract p_nlos * mu_nlos directly
- PRNC-adaptive: CNO-adaptive noise floor + two-stage gating
- Result: Still worse than Standard LS in 3/4 datasets

### SupervisedMuRegressionLoss

Added to GAT_V2025.py training loop (NLL stage, epoch 34+):
`python
loss_mu_reg = SupervisedMuRegressionLoss()(mu_nlos, pseudorange_error, labels)
total_loss += LAMBDA_MU_REG * loss_mu_reg
`

Effect: mu_nlos NLOS increased from 0.05-0.15 km to 0.16-0.40 km.
Problem: mu_nlos LOS > mu_nlos NLOS in all datasets (direction reversed).

---

## Key Findings

1. **NLOS pseudorange errors are symmetric** (39-53% positive), not predominantly positive as PRNC assumes
2. **mu_nlos direction is wrong**: Model outputs higher mu for LOS than NLOS
3. **PRNC does not beat Standard LS** in any dataset
4. **WLS-MoG beats Standard LS in frankfurt1** (+11.7% CEP50), suggesting weighting can work with the right geometry
5. **Training is stable** with MoG + SupervisedMuRegressionLoss (no sigma explosion)
6. **Block-diagonal batching (bs=32)** delivers 2.7x speedup with zero quality loss

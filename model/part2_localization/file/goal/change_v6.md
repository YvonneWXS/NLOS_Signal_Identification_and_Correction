# v6 Change Log

**Date**: 2026-06-05
**Goal**: goal_v6.md -- LOS-Anchored Clock Contamination Fix
**Result**: Hypothesis rejected. Clock contamination is NOT the root cause.

---

## Code Changes

### New Files

| File | Purpose |
|------|---------|
| part2_FactorGraphLocalizationFusion/model/fusion/los_anchored_ls.py | 5 LOS-anchored positioning methods + geometry-aware satellite selection |
| part2_FactorGraphLocalizationFusion/model/fusion/verify_clock_contamination.py | Clock contamination verification script (PART 0 diagnostic) |
| part2_FactorGraphLocalizationFusion/file/goal/result_v6.md | Full v6 results report |
| part2_FactorGraphLocalizationFusion/file/goal/change_v6.md | This file |

### Modified Files

| File | Change | Reason |
|------|--------|--------|
| part2_FactorGraphLocalizationFusion/model/fusion/evaluate_fusion.py | Added 5 LOS-anchored methods + import from los_anchored_ls | Extend evaluation to 16 methods |
| part2_FactorGraphLocalizationFusion/model/fusion/evaluate_fusion.py | Added data extraction for p_los/sigma/mu arrays | Pass MoG outputs to LOS-anchored methods |
| part2_FactorGraphLocalizationFusion/cache/clock_contamination_analysis.json | Diagnostic results from verify_clock_contamination.py | Evidence for hypothesis testing |

### New Methods Added (5)

| # | Method | File | Description |
|:---:|------|------|------|
| 12 | LOS-Anchored-LS | los_anchored_ls.py | Standard LS with LOS-only clock estimate |
| 13 | LOS-Anchored-WLS-MoG | los_anchored_ls.py | WLS-MoG with LOS-only clock estimate |
| 14 | LOS-Anchored-PRNC | los_anchored_ls.py | PRNC with LOS-only clock (numerically unstable) |
| 15 | LOS-Anchored-Debiased-WLS | los_anchored_ls.py | Debiased WLS with LOS-only clock |
| 16 | LOS-Anchored-Combined | los_anchored_ls.py | Geometry-aware selection + LOS clock + debiased + WLS |

### New Evaluation Results

| Experiment | Dataset Count | Total Methods | Result |
|--------|:---:|:---:|------|
| exp_012 | 4 datasets | 16 methods | LOS-anchored methods make 0% difference |

---

## Key Technical Findings

1. **LOS-anchored clock = Standard clock**: LOS-Anchored-LS produces IDENTICAL CEP50 to Standard LS in all 4 datasets. The iterative LS algorithm self-corrects any clock bias during convergence.

2. **Clock is a free parameter in LS**: The Gauss-Newton iteration solves for clock simultaneously with position. Initial clock estimate does not affect final solution.

3. **LOS-anchored WLS is slightly WORSE than regular WLS**: The binary threshold (p_los > 0.7) is less effective than continuous weighting by p_los directly.

4. **mu_nlos direction is the real bottleneck**: All methods that use mu_nlos fail because Module 1 outputs mu_nlos_LOS > mu_nlos_NLOS (wrong direction).

5. **Frankfurt1 remains the only success case**: FactorGraph-MoG+2A achieves +15.1% vs Standard LS in frankfurt1, but this is due to favorable geometry, not clock handling.

---

## Deleted/Harmed Functionality

- None. All previous methods preserved.
- evaluate_fusion.py maintains all 12 original v5 methods + 5 new v6 methods.
- No existing results were overwritten.

---

## Build/Deploy Notes

- los_anchored_ls.py is a standalone module with no dependencies beyond numpy
- All 5 LOS-anchored methods follow the same interface: (obs_list, sv_positions, p_los, sigma_los, mu_nlos=...) -> (x_ecef, clk)
- The LOS_ANCHORED_METHODS dict enables easy registration in evaluate_fusion.py

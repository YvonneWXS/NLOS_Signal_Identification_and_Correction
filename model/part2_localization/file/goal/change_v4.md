# Module 2 v4 Change Log

> Date: 2026-06-03 | v4 Diagnostic + Remediation Sprint

---

## Files Created

### 1. usion/diagnose_weighting.py ? PART 0: Root Cause Diagnosis
4-dimensional analysis (A: weight distribution, B: DOP inflation, C: clock coupling, D: residuals).
Generates diagnosis summary and saves to cache/diagnosis_v4.json.

### 2. cache/{dataset}_mog_outputs.pkl (4 files) ? MoG Inference Cache
Precomputed Module 1 inference results for all 4 datasets to avoid redundant forward passes.

### 3. 
un_v4_eval.py ? Full 9-Method Evaluation Script

---

## Files Modified

### 4. usion/baselines.py ? PART 1: 6 New Weight Schemes

Added 6 new solver functions:
- solve_wls_aggressive_power() ? Scheme 1: p_los^3 / sigma^2
- solve_wls_log_odds() ? Scheme 2: log(p/(1-p)) / sigma^2
- solve_wls_soft_floor() ? Scheme 3: max(0.05, p_los^2) / sigma^2
- solve_wls_geometry_aware() ? Scheme 4: PDOP-aware (keep critical sats)
- solve_wls_debiased() ? Scheme 5: subtract (1-p_los)*mu_nlos
- solve_raim_mog() ? Scheme 6: iterative NLOS exclusion

### 5. usion/factor_graph_fusion.py ? PART 2: Debiased FactorGraph

Added FactorGraphPositioner.solve_epoch_debiased() method:
- Corrects pseudoranges by subtracting (1-p_los)*mu_nlos
- Uses WLS-debiased as initial solution
- Quick 2-3 iteration L-BFGS-B refinement

### 6. usion/evaluate_fusion.py ? PART 3: 9-Method Evaluation

Updated from 6 to 9 methods:
1. Standard LS (keep)
2. WLS-elevation (keep)
3. WLS-MoG-linear (keep)
4. WLS-power3 (new)
5. WLS-log-odds (new)
6. WLS-debiased (new)
7. RAIM-MoG (new)
8. FG-debiased (new)
9. FG-debiased+2A (new)

---

## Results: All Methods FAIL

No method beats Standard LS in any dataset.
RAIM-MoG is neutral (identical to Standard LS).
WLS-debiased is worse in all 4 datasets (-9% to -38%).

## Root Cause (from diagnosis)

- Weight discrimination adequate (LOS/NLOS ratio 2.5-3.9)
- DOP inflation is the primary killer (frankfurt1: 58.2% epochs degraded)
- Clock coupling strongly correlates with error increase (berlin1 corr=0.575, frankfurt1 corr=0.651)
- NLOS residuals INCREASE under MoG weighting (all datasets: -15% to -40%)

## Skipped

PART 4 (TCN 83-dim retrain): Skipped because PART 3 failed.
Condition was: WLS-debiased must beat Standard LS in >=2/4 datasets (0/4 achieved).

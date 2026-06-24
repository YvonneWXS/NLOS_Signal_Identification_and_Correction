Goal: Module 2 fundamental fix — diagnose and resolve why MoG 
weighting universally underperforms Standard LS, then achieve 
WLS-MoG CEP50 better than Standard LS in at least 2/4 datasets.

This is a multi-part diagnostic and fix task. Run ALL parts in 
sequence. Do not skip any step.

================================================================
PART 0: ROOT CAUSE DIAGNOSIS (run first, print full report)
================================================================

File: fusion/diagnose_weighting.py

The core symptom: WLS-MoG CEP50 > Standard LS in ALL 4 datasets.
Standard LS uses uniform weights w=1. WLS-MoG uses w=p_los/sigma_los².
Somewhere in this formula, useful information is being discarded 
or geometry is being degraded. Find out exactly why.

--- Diagnosis A: Weight distribution analysis ---
For each dataset, run Module 1 inference on ALL epochs.
Compute per-satellite weights for WLS-MoG: w_i = p_los_i / sigma_los_i²
Print the following statistics:
  - Mean weight for LOS satellites vs NLOS satellites (should be different)
  - Ratio: mean_weight_LOS / mean_weight_NLOS (want >> 1.0)
  - Fraction of epochs where max_weight/min_weight < 2.0 
    (near-uniform → WLS ≈ LS)
  - Histogram of w_i values (10 bins)
  
If mean_weight_LOS / mean_weight_NLOS < 2.0, weights are NOT 
discriminative enough. This is the primary failure mode.

--- Diagnosis B: Geometric dilution impact ---
For each epoch, compute:
  PDOP_standard = PDOP with uniform weights
  PDOP_mog = PDOP with WLS-MoG weights (use weighted covariance 
    formula: P = (H^T W H)^{-1}, PDOP = sqrt(P[0,0]+P[1,1]+P[2,2]))
  
Print per-dataset statistics:
  - Mean PDOP_standard vs Mean PDOP_mog
  - Fraction of epochs where PDOP_mog > PDOP_standard × 1.1
    (MoG weights actively worsening geometry)
  
If PDOP_mog > PDOP_standard in >30% of epochs, DOP inflation is
the secondary failure mode (good NLOS satellites being downweighted).

--- Diagnosis C: Clock bias coupling analysis ---
In LS, clock bias absorbs the common pseudorange offset.
When you weight satellites non-uniformly, the clock estimate
shifts, which can redistribute errors onto position.
  
For each epoch, compare:
  clk_standard = clock bias estimated by Standard LS
  clk_mog = clock bias estimated by WLS-MoG
  delta_clk = clk_mog - clk_standard  (in meters)
  
Print per-dataset statistics:
  - Mean |delta_clk| and std
  - Correlation between |delta_clk| and positioning error increase
    (delta_error = error_mog - error_standard)
    
If |mean delta_clk| > 50m and correlation > 0.3, clock coupling
is the tertiary failure mode.

--- Diagnosis D: Per-satellite residual analysis ---
After Standard LS, compute per-satellite residuals.
After WLS-MoG, compute per-satellite residuals.
For each dataset, print:
  - Mean absolute residual for LOS/NLOS satellites in Standard LS
  - Mean absolute residual for LOS/NLOS satellites in WLS-MoG
  - Whether NLOS residuals are reduced by MoG weighting

--- Print diagnosis summary ---
At the end of diagnose_weighting.py, print a clear diagnosis:
  DATASET berlin1: PRIMARY_CAUSE=X, SECONDARY=Y
  DATASET berlin2: PRIMARY_CAUSE=X, SECONDARY=Y
  ...
Where causes are: WEIGHT_NOT_DISCRIMINATIVE / DOP_INFLATION / 
CLOCK_COUPLING / UNKNOWN

================================================================
PART 1: FIX WEIGHT FORMULA (based on diagnosis results)
================================================================

File: fusion/baselines.py — add new weight functions

Implement ALL of the following weight schemes and test all of them:

--- Scheme 1: Aggressive p_los power (current is linear) ---
  w_i = p_los_i^3 / sigma_los_i²
  (cubing p_los makes the LOS/NLOS distinction more aggressive)

--- Scheme 2: Log-odds weighting ---
  odds_i = p_los_i / (1 - p_los_i + 1e-6)
  w_i = max(0.01, log(odds_i)) / sigma_los_i²
  (log-odds is more spread out than linear probability)

--- Scheme 3: Soft exclusion with floor ---
  w_i = max(0.05, p_los_i²) / sigma_los_i²
  (keep all satellites but severely downweight NLOS)
  
--- Scheme 4: Geometry-aware weighting ---
  Compute PDOP if satellite i is removed.
  If removing satellite i increases PDOP by > 0.5:
    w_i = max(0.3, p_los_i) / sigma_los_i²  # keep geometrically critical sats
  Else:
    w_i = p_los_i² / sigma_los_i²  # aggressive downweight

--- Scheme 5: Two-stage approach ---
  Stage 1: Run Standard LS to get initial position estimate
  Stage 2: Compute residuals. For each satellite i:
    expected_residual_los  = 0 (km)
    expected_residual_nlos = mu_nlos_i (km)
    predicted_residual = p_los_i * 0 + (1-p_los_i) * mu_nlos_i
    corrected_pr_i = pr_mes_i - predicted_residual  # subtract expected NLOS bias
    w_i = p_los_i / sigma_los_i²
  Run WLS with corrected pseudoranges. This is the key insight:
  WLS alone adjusts weights, but doesn't correct the NLOS bias.
  Subtracting mu_nlos*(1-p_los) directly debiases NLOS pseudoranges.

--- Scheme 6: RAIM-style iterative exclusion ---
  iter = 0, active_set = all satellites
  while iter < 5:
    Run LS on active_set → get position estimate x
    Compute residuals r_i = pr_mes_i - predicted_pr_i
    Compute normalized residual z_i = |r_i| / sigma_expected_i
      where sigma_expected_i = p_los_i*sigma_los_i + (1-p_los_i)*sigma_nlos_i
    Satellite with max z_i: if z_i > 3.0 AND p_los_i < 0.5:
      remove from active_set
    else: break
  Return LS solution on final active_set.
  (Never remove satellites if active_set would have < 5 sats)

Add all 6 new solve functions to baselines.py:
  solve_wls_aggressive_power()
  solve_wls_log_odds()
  solve_wls_soft_floor()
  solve_wls_geometry_aware()
  solve_wls_debiased()  ← most theoretically correct, test carefully
  solve_raim_mog()

================================================================
PART 2: UPDATE FACTOR GRAPH WITH DEBIASING (P0)
================================================================

File: fusion/factor_graph_fusion.py

The most important fix: the MoG observation model already models
mu_nlos, but L-BFGS-B cannot exploit it due to flat NLL surface.
Instead, bake the debiasing directly into the observation:

In MoGObservationModel, add a corrected_pseudorange property:
  corrected_pr_i = pr_mes_i - (1 - p_los_i) * mu_nlos_i
  
  This shifts the pseudorange toward what it would be if the NLOS
  bias were perfectly corrected. Use corrected_pr_i as input to
  the geometric range computation.

Implement a new FactorGraphPositioner method: solve_epoch_debiased()
  1. Apply pseudorange debiasing: pr_corrected = pr - (1-p_los)*mu_nlos
  2. Use WLS with Scheme 5 weights on debiased pseudoranges
  3. Optionally refine with 2-3 iterations of L-BFGS-B
  
This should be the primary FactorGraph method going forward.

================================================================  
PART 3: FULL EVALUATION WITH ALL NEW METHODS
================================================================

File: fusion/evaluate_fusion.py — update to include new schemes

Run all methods on all 4 datasets. The new method list:
  1. Standard LS (baseline, no Module 1)
  2. WLS-elevation (baseline, no Module 1)
  3. WLS-MoG-linear (current, p_los/sigma²)
  4. WLS-MoG-power3 (Scheme 1)
  5. WLS-log-odds (Scheme 2)
  6. WLS-debiased (Scheme 5) ← expected to be best WLS variant
  7. RAIM-MoG (Scheme 6)
  8. FactorGraph-debiased (Part 2)
  9. FactorGraph-debiased+2A (with TCN prior)

For each method, report:
  CEP50 (m), CEP95 (m), Mean2D (m), RMSE3D (m)
  % improvement over Standard LS (positive = better than LS)
  
Save to fusion/result/exp_v4/positioning_results_full.json

Print the following additional analysis:
  --- Which methods beat Standard LS? ---
  For each method: list datasets where it beats Standard LS
  Required: WLS-debiased should beat Standard LS in >=2/4 datasets
  
  --- Debiasing contribution ---
  Compare WLS-MoG-linear vs WLS-debiased:
  Shows the isolated effect of mu_nlos debiasing
  
  --- Per-epoch improvement distribution ---
  For the best-performing new method vs Standard LS:
  Plot (print as ASCII histogram) the distribution of 
  per-epoch improvement: positive = new method better
  Shows whether improvement is consistent or episodic

================================================================
PART 4: TCN RETRAIN WITH IMPROVED p_los TARGETS
================================================================

File: fusion/train_tcn.py

The current TCN uses Module 1 p_los as soft labels.
With debiased positioning, the TCN target should be updated:
  target_i = (1 - p_los_gat_i)  ← unchanged
  
BUT the TCN input features should include mu_nlos:
  Per satellite slot: [elevation/90, azimuth/360, p_los, mu_nlos/0.5]
  → 4 features per satellite × 20 slots + 3 velocity = 83-dim per timestep
  (was 63-dim: 3 features per satellite)

Update motion_geometry_predictor.py INPUT_DIM from 63 to 83.
Rebuild all 4 caches with new 83-dim features.
Retrain all 4 TCN models (50 epochs, batch=128, patience=10).
Save updated models to fusion/models/tcn_v2_{dataset}.pth

In evaluate_fusion.py, update FactorGraph-debiased+2A to:
  1. Load tcn_v2_{dataset}.pth
  2. Apply TCN prior (soft blend, alpha cap 0.3)
  3. Apply debiased pseudoranges
  4. Run WLS-debiased with updated p_los

================================================================
PART 5: FINAL SUCCESS CRITERIA REPORT
================================================================

At the end of run_fusion.py, print a structured pass/fail report:

=== Module 2 Success Criteria ===
[PASS/FAIL] WLS-debiased beats Standard LS in >=2/4 datasets by >3%
  berlin1: {delta}% | berlin2: {delta}% | frk1: {delta}% | frk2: {delta}%
  
[PASS/FAIL] Best FG method beats Standard LS in >=2/4 datasets by >3%
  berlin1: {delta}% | berlin2: {delta}% | frk1: {delta}% | frk2: {delta}%

[PASS/FAIL] FG+2A does NOT degrade vs FG in any dataset

[PASS/FAIL] Diagnosis: weight discrimination ratio >= 2.0 in >=3/4 datasets
  berlin1: {ratio} | berlin2: {ratio} | frk1: {ratio} | frk2: {ratio}

[PASS/FAIL] Diagnosis: mean |delta_clk| < 100m in all datasets

[PASS/FAIL] FG debiased beats WLS debiased in >=2/4 datasets
  (tests if factor graph optimization adds value beyond debiasing alone)

If all PASS: print "Module 2 COMPLETE — ready for Module 3"
If any FAIL: print which specific step caused the failure

================================================================
IMPLEMENTATION ORDER (strictly follow this sequence)
================================================================

Step 1: Run diagnose_weighting.py → read diagnosis output
Step 2: Implement all 6 weight schemes in baselines.py
Step 3: Implement debiased factor graph in factor_graph_fusion.py
Step 4: Run full evaluation (Part 3) → check if WLS-debiased 
        beats Standard LS in >=2/4 datasets
Step 5: If Step 4 passes → rebuild TCN with 83-dim features (Part 4)
        If Step 4 fails → print detailed failure analysis and STOP
Step 6: Run final evaluation with updated TCN → print criteria report

================================================================
CONSTRAINTS
================================================================
- Do NOT modify Module 1 files (GAT_V2025.py, config.py, etc.)
- Keep all previous methods in evaluate_fusion.py for comparison
- Frankfurt models: use exp_038/exp_039 (already trained)
- If debiasing makes things worse, print diagnosis and try without
  mu_nlos (set mu_nlos=0) to isolate the effect
- All 9 methods must run on all 4 datasets without crash
- Cache Module 1 inference results per dataset to avoid redundant
  forward passes: save to fusion/cache/{dataset}_mog_outputs.pkl
  and reload if file exists
- Print timing: report wall-clock time per method per dataset
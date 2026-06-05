Goal: Module 2 v7 — Fix mu_nlos directional inversion in Module 1,
retrain all 4 models, then verify that WLS-debiased and PRNC-mu 
methods achieve CEP50 improvement over Standard LS in >=2/4 datasets.

This is the final targeted fix. The v1-v6 diagnostic has conclusively
identified mu_nlos direction inversion as the sole remaining bottleneck.
All other hypotheses (DOP inflation, clock contamination, PRNC 
assumptions, WLS weight formulas) have been tested and ruled out.

The evidence:
  berlin1:    mu_nlos LOS=0.248 km > NLOS=0.226 km  (WRONG direction)
  berlin2:    mu_nlos LOS=0.321 km > NLOS=0.185 km  (WRONG direction)
  frankfurt1: mu_nlos LOS=0.465 km > NLOS=0.395 km  (WRONG direction)
  frankfurt2: mu_nlos LOS=0.353 km > NLOS=0.162 km  (WRONG direction)

Physical ground truth: NLOS pseudorange error = positive delay (signal
travels extra distance). So mu_nlos[NLOS] should be >> mu_nlos[LOS].

================================================================
PART 1: FIX mu_nlos DIRECTION IN MODULE 1 (CRITICAL)
================================================================

File: part1_GAT/model/GAT_V2025.py

The current training has two problems:
Problem A: SupervisedMuRegressionLoss only trained on NLOS samples.
  LOS samples have no constraint on mu_nlos → backbone learns to 
  output high mu_nlos for LOS samples to improve MoG NLL.

Problem B: No ordering constraint between LOS and NLOS mu values.
  Nothing forces mu_nlos[NLOS] > mu_nlos[LOS].

--- Fix A: Add LOS mu_nlos suppression loss ---

Add a new loss that explicitly suppresses mu_nlos for LOS samples:

class MuDirectionLoss(nn.Module):
  def forward(self, mu_nlos, labels):
    """
    labels: 1=NLOS, 0=LOS
    LOS samples: penalize mu_nlos > 0.05 km (should be near zero)
    NLOS samples: handled by SupervisedMuRegressionLoss
    """
    los_mask = (labels == 0)
    nlos_mask = (labels == 1)
    
    loss = torch.tensor(0.0, device=mu_nlos.device)
    
    # LOS constraint: mu_nlos should be small (LOS has no NLOS delay)
    if los_mask.any():
      # Penalize LOS mu above 0.05 km (5-sigma LOS noise floor)
      los_excess = torch.relu(mu_nlos[los_mask] - 0.05)
      loss_los = los_excess.mean()
      loss = loss + 2.0 * loss_los
    
    # Ordering constraint: mean(mu_NLOS) > mean(mu_LOS) + margin
    if los_mask.any() and nlos_mask.any():
      mu_los_mean = mu_nlos[los_mask].mean()
      mu_nlos_mean = mu_nlos[nlos_mask].mean()
      margin = 0.1  # km — NLOS should be at least 100m more than LOS
      ordering_violation = torch.relu(mu_los_mean - mu_nlos_mean + margin)
      loss = loss + 3.0 * ordering_violation
    
    return loss

--- Fix B: Update training loop in GAT_V2025.py ---

In the NLL training stage (epoch >= MOG_PURE_BCE_EPOCHS + MOG_BLEND_EPOCHS):

  # Existing losses
  loss_mog_nll = MoGNLLLoss(...)(p_los, mu_nlos, sigma_los, sigma_nlos, 
                                  pseudorange_error, labels)
  loss_bce = NLOSLoss(...)(p_los, labels)
  
  # Existing supervised mu (Huber on NLOS only)
  loss_mu_supervised = SupervisedMuRegressionLoss()(
      mu_nlos, pseudorange_error, labels)
  
  # NEW: Direction constraint (add this)
  loss_mu_direction = MuDirectionLoss()(mu_nlos, labels)
  
  # Combined loss
  total_loss = (dynamic_bce_weight * loss_bce + 
                0.1 * loss_mog_nll + 
                0.5 * loss_mu_supervised +   # existing
                1.0 * loss_mu_direction)     # NEW

Also apply MuDirectionLoss in the Blend stage (epoch 9-33):
  loss_comp_nll = SupervisedComponentNLLLoss(...)(...)
  loss_bce_blend = NLOSLoss(...)(...)
  loss_mu_direction = MuDirectionLoss()(mu_nlos, labels)
  total_loss = blend_loss + 0.5 * loss_mu_direction

--- Fix C: Update config.py ---

Change:
  MU_NLOS_TARGET = 0.5    # km (was 0.15, keeps supervised target reasonable)
  LAMBDA_MU_REG = 0.05    # Reduce L2 anchor (was 0.30, was pulling toward 0)
  MU_NLOS_MAX = 3.0       # km (keep as v5)
  MU_NLOS_MIN = 0.0       # km (keep)

--- Fix D: Separate LOS/NLOS mu supervision in Blend stage ---

In SupervisedComponentNLLLoss.forward(), currently only NLOS samples
are supervised. ADD explicit LOS supervision:

  # LOS samples: pseudorange error should be near 0
  if los_mask.any():
    pr_err_los = pseudorange_error[los_mask]
    # Target for LOS: small error, expected ~0
    mu_los_target = torch.clamp(torch.abs(pr_err_los), max=0.1)
    loss_mu_los = F.huber_loss(mu_nlos[los_mask], 
                                mu_los_target, delta=0.05)
    total_loss = total_loss + 0.5 * loss_mu_los

================================================================
PART 2: RETRAIN ALL 4 MODELS (exp_044-047)
================================================================

Run training for all 4 datasets:
  python run_full_training.py --dataset berlin1_potsdamer_platz  → exp_044
  python run_full_training.py --dataset berlin2_gendarmenmarkt   → exp_045
  python run_full_training.py --dataset frankfurt1_maintower     → exp_046
  python run_full_training.py --dataset frankfurt2_westendtower  → exp_047

After each training, run analyze_mog.py and verify:
  REQUIRED: mu_nlos[NLOS] > mu_nlos[LOS] for ALL 4 datasets
  TARGET: mu_nlos[NLOS] > 0.25 km (meaningful correction)
  TARGET: mu_nlos[LOS] < 0.10 km (suppressed as expected)
  CLASSIFICATION: F1 should not drop more than 0.01 from exp_040-043

If any model fails direction check (mu_LOS >= mu_NLOS):
  STOP training that dataset, print diagnosis:
    - Print mu_nlos distribution for LOS and NLOS samples
    - Print MuDirectionLoss value at best epoch
    - Check if batch contains enough LOS+NLOS pairs to compute loss
  Then increase loss_mu_direction weight from 1.0 to 2.0 and retrain.

Save analysis to result/exp_044-047/mu_direction_check.json with fields:
  {
    "mu_nlos_los_mean": float,
    "mu_nlos_nlos_mean": float,
    "direction_correct": bool,  // nlos_mean > los_mean
    "direction_margin_km": float,  // nlos_mean - los_mean
    "f1": float,
    "p_los_gap": float
  }

================================================================
PART 3: GEOMETRY-AWARE WLS WITH CORRECTED mu_nlos
================================================================

File: fusion/baselines.py — add new method
File: fusion/evaluate_fusion.py — add to method list

The v4 diagnosis showed DOP inflation as a primary failure mode in
frankfurt1 (58.2% epochs degraded). The geometry-aware satellite 
selection from v6 (los_anchored_ls.py) was never tested with 
CORRECTED mu_nlos. Now test it.

--- New Method: Geometry-Aware-Debiased-WLS ---

Add to fusion/baselines.py:

def solve_geometry_aware_debiased_wls(obs_list, sv_positions, 
                                       p_los, sigma_los, mu_nlos,
                                       max_iter=10, pdop_threshold=1.15):
  """
  Step 1: Initial Standard LS to get approximate position
  Step 2: Geometry-aware satellite selection 
    (only exclude NLOS sats if PDOP doesn't increase > pdop_threshold)
  Step 3: Debias selected sats: pr_corrected = pr - (1-p_los)*mu_nlos
  Step 4: WLS with weights p_los/sigma² on selected+debiased sats
  Step 5: Iterate steps 2-4 until convergence
  """
  from fusion.los_anchored_ls import select_satellites_geometry_aware
  
  pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
  p_nlos = 1.0 - p_los
  
  # Initial position from standard LS
  x, clk = run_standard_ls_4dof(pr_mes, sv_positions)
  
  for iteration in range(max_iter):
    # Geometry-aware satellite selection
    selected = select_satellites_geometry_aware(
        sv_positions, p_los, sigma_los, x, 
        min_sats=5, pdop_limit=pdop_threshold)
    
    sv_sel = sv_positions[selected]
    pr_sel = pr_mes[selected]
    p_los_sel = p_los[selected]
    p_nlos_sel = p_nlos[selected]
    sigma_sel = sigma_los[selected]
    mu_sel = mu_nlos[selected]
    
    # Debias selected pseudoranges using corrected mu_nlos
    pr_corrected = pr_sel - p_nlos_sel * mu_sel
    
    # WLS iteration
    dists = np.linalg.norm(sv_sel - x[np.newaxis,:], axis=1)
    residuals = pr_corrected - dists - clk
    
    weights = np.maximum(0.01, p_los_sel) / np.maximum(0.01, sigma_sel**2)
    W = np.diag(weights)
    
    los_vecs = (sv_sel - x[np.newaxis,:]) / dists[:,np.newaxis]
    H = np.hstack([-los_vecs, np.ones((len(pr_corrected),1))])
    
    try:
      WH = np.sqrt(W) @ H
      Wr = np.sqrt(W) @ residuals
      delta = np.linalg.lstsq(WH, Wr, rcond=None)[0]
    except:
      break
    
    x += delta[:3]
    clk += delta[3]
    
    if np.linalg.norm(delta[:3]) * 1000 < 0.1:
      break
  
  return x, clk

Also add a simpler variant for comparison:

def solve_debiased_wls_v2(obs_list, sv_positions, p_los, sigma_los, mu_nlos):
  """
  Pure debiased WLS without geometry selection.
  Tests whether corrected mu_nlos alone fixes the debiasing approach.
  """
  pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
  p_nlos = 1.0 - p_los
  pr_corrected = pr_mes - p_nlos * mu_nlos
  
  weights = np.maximum(0.01, p_los) / np.maximum(0.01, sigma_los**2)
  
  x = np.zeros(3); x[2] = 6371.0
  clk = np.median(pr_corrected - np.linalg.norm(sv_positions - x, axis=1))
  
  for _ in range(10):
    dists = np.linalg.norm(sv_positions - x[np.newaxis,:], axis=1)
    residuals = pr_corrected - dists - clk
    W = np.diag(weights)
    los_vecs = (sv_positions - x[np.newaxis,:]) / dists[:,np.newaxis]
    H = np.hstack([-los_vecs, np.ones((len(pr_mes),1))])
    try:
      delta = np.linalg.lstsq(np.sqrt(W)@H, np.sqrt(W)@residuals, rcond=None)[0]
    except:
      break
    x += delta[:3]; clk += delta[3]
    if np.linalg.norm(delta[:3])*1000 < 0.1: break
  
  return x, clk

================================================================
PART 4: PRNC-MU WITH CORRECTED mu_nlos
================================================================

File: fusion/prnc.py — add corrected variant
File: fusion/evaluate_fusion.py — add to method list

The v5 PRNC-mu failed because mu_nlos was directionally wrong
(mu_LOS > mu_NLOS → correction added bias instead of removing it).
With corrected mu_nlos, retest PRNC-mu:

Add to fusion/prnc.py:

class CorrectedMuPRNC:
  """PRNC using directionally-correct mu_nlos from exp_044-047."""
  
  def solve_epoch(self, obs_list, sv_positions, mog_outputs,
                   max_iters=5):
    pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
    p_los = mog_outputs['p_los']
    p_nlos = 1.0 - p_los
    mu_nlos = mog_outputs['mu_nlos']
    sigma_los = mog_outputs['sigma_los']
    
    # Verify direction before applying (safety check)
    # If mu_nlos looks wrong, fall back to Standard LS
    if len(p_los) > 0:
      los_mask = p_los > 0.7
      nlos_mask = p_los < 0.3
      if los_mask.sum() > 0 and nlos_mask.sum() > 0:
        if mu_nlos[los_mask].mean() > mu_nlos[nlos_mask].mean():
          # Direction still wrong — fall back gracefully
          return run_standard_ls_4dof(pr_mes, sv_positions)
    
    # Apply mu_nlos correction: subtract expected NLOS delay
    # Scale by p_nlos: stronger correction for more likely NLOS sats
    correction = p_nlos * mu_nlos
    pr_corrected = pr_mes - correction
    
    # Run WLS with corrected pseudoranges (uniform weights)
    x, clk = run_standard_ls_4dof(pr_corrected, sv_positions)
    
    # Iterative refinement
    for iteration in range(max_iters - 1):
      dists = np.linalg.norm(sv_positions - x[np.newaxis,:], axis=1)
      residuals = pr_corrected - dists - clk
      los_vecs = (sv_positions - x[np.newaxis,:]) / dists[:,np.newaxis]
      H = np.hstack([-los_vecs, np.ones((len(pr_mes),1))])
      try:
        delta = np.linalg.lstsq(H, residuals, rcond=None)[0]
      except:
        break
      x += delta[:3]; clk += delta[3]
      if np.linalg.norm(delta[:3])*1000 < 0.1: break
    
    return x, clk

================================================================
PART 5: FINAL 20-METHOD EVALUATION
================================================================

File: fusion/evaluate_fusion.py — update to v7, 20 methods

Update model mapping to exp_044-047 at the top of evaluate_fusion.py.
Delete old caches: fusion/cache/{dataset}_mog_outputs.pkl
(They will be rebuilt automatically for new models.)

Add 4 new methods to the 16 existing ones:
  17. WLS-debiased-v2 (Part 3: debiased WLS with corrected mu, no geometry selection)
  18. Geometry-Aware-Debiased-WLS (Part 3: full method with geometry selection)
  19. PRNC-mu-corrected (Part 4: PRNC-mu with direction check + fallback)
  20. FactorGraph-MoG+2A (keep, update to use exp_044-047)

For EACH of the 20 methods on ALL 4 datasets, report:
  CEP50 (m), CEP95 (m), Mean 2D (m)
  delta_CEP50 vs Standard LS (%)
  delta_CEP50 vs WLS-MoG-linear (%)

Additional per-method diagnostics for methods 17-20:
  - Mean mu_nlos correction applied (km)
  - Fraction of epochs where mu correction direction is correct
    (i.e., mean mu_nlos[p_los<0.4] > mean mu_nlos[p_los>0.7])
  - For Geometry-Aware: mean fraction of sats selected, mean PDOP ratio

Save to fusion/result/exp_v7/positioning_results_v7.json

================================================================
PART 6: mu_nlos DIRECTION QUALITY GATE
================================================================

File: fusion/evaluate_fusion.py — add pre-evaluation check

Before running any positioning methods, run a direction quality check:

def check_mu_direction_quality(mog_outputs_all_epochs):
  """
  Computes direction quality across ALL epochs and satellites.
  Returns dict with per-dataset direction metrics.
  """
  all_mu = []
  all_plos = []
  all_labels = []
  
  for epoch_data in mog_outputs_all_epochs:
    all_mu.extend(epoch_data['mu_nlos'])
    all_plos.extend(epoch_data['p_los'])
    all_labels.extend(epoch_data['nlos_label'])  # ground truth
  
  all_mu = np.array(all_mu)
  all_plos = np.array(all_plos)
  all_labels = np.array(all_labels)
  
  los_mask = (all_labels == 0)
  nlos_mask = (all_labels == 1)
  
  mu_los_mean = all_mu[los_mask].mean()
  mu_nlos_mean = all_mu[nlos_mask].mean()
  direction_correct = mu_nlos_mean > mu_los_mean
  margin = mu_nlos_mean - mu_los_mean
  
  print(f"mu_nlos direction check:")
  print(f"  LOS mean: {mu_los_mean*1000:.1f} m")
  print(f"  NLOS mean: {mu_nlos_mean*1000:.1f} m")
  print(f"  Direction: {'CORRECT ✓' if direction_correct else 'WRONG ✗'}")
  print(f"  Margin: {margin*1000:.1f} m")
  
  if not direction_correct:
    print("WARNING: mu_nlos direction still wrong!")
    print("Methods 17-19 will use fallback Standard LS")
  
  return {
    'mu_los_mean_km': float(mu_los_mean),
    'mu_nlos_mean_km': float(mu_nlos_mean),
    'direction_correct': bool(direction_correct),
    'margin_km': float(margin)
  }

Call this at the start of evaluate_all_methods() and store result.
Pass direction_correct flag into PRNC-mu-corrected method.

================================================================
PART 7: SUCCESS CRITERIA AND FINAL REPORT
================================================================

File: fusion/generate_final_report.py (NEW)

Generate a comprehensive final report for Module 2:

--- Section 1: mu_nlos direction fix verification ---
For each dataset (exp_044-047):
  mu_nlos LOS mean (km) | NLOS mean (km) | margin (km) | direction
  Target: NLOS > LOS by at least 0.1 km

--- Section 2: Positioning improvement table ---
Best method per dataset (compare to Standard LS):
  Dataset | Best Method | CEP50 | Improvement vs Std LS
  
  Required for Module 2 COMPLETE:
    At least 2/4 datasets where best MoG method beats Standard LS by >3%

--- Section 3: Version progression ---
Print a table showing CEP50 trajectory across v3-v7:
  Method: FactorGraph-MoG+2A (the one consistent winner)
  v3: berlin1=959m, berlin2=765m, frankfurt1=474m, frankfurt2=493m
  v4: [same model, different eval]
  ...
  v7: [new results with corrected mu]

--- Section 4: Key scientific findings ---
Print 5 key findings from v1-v7 investigation:
  1. WLS fails in 3/4 datasets due to DOP inflation
  2. Clock contamination is NOT a factor (iterative LS self-corrects)
  3. NLOS errors are symmetric (not positive-biased) after clock absorption
  4. mu_nlos direction inversion was the core Module 1 bug
  5. Factor graph + temporal prior (TCN) provides best results when geometry permits

--- Print final verdict ---
if >= 2 datasets show > 3% improvement over Standard LS:
  print "Module 2 SUCCESS CRITERIA MET"
  print "Best method: {method_name}"
  print "Ready to proceed to Module 3 (Residual Feedback + Online Correction)"
else:
  print "Module 2 PARTIAL SUCCESS"
  print "Frankfurt1: +15.1% (FactorGraph-MoG+2A consistently)"
  print "Recommendation: accept current results and proceed to Module 3"
  print "Module 3 residual feedback should generalize the frankfurt1 result"
  print "to other datasets by adaptively learning scene-specific parameters"

Save to fusion/result/exp_v7/FINAL_REPORT.md

================================================================
PART 8: PAPER TABLE GENERATION
================================================================

File: fusion/generate_paper_table.py (update from v6)

Generate clean comparison tables for the paper:

Table 1: CEP50 across datasets and key methods (LaTeX format)
  Rows: Standard LS, WLS-elevation, WLS-MoG, FG-MoG, FG-MoG+2A, 
        Geometry-Aware-Debiased-WLS (v7 best)
  Columns: berlin1, berlin2, frankfurt1, frankfurt2

Table 2: Module 1 quality metrics (LaTeX format)
  Rows: exp_034 (baseline), exp_040 (supervised mu v5), 
        exp_044 (direction fix v7)
  Columns: F1, p_los_gap, mu_nlos[LOS], mu_nlos[NLOS], direction_correct

Table 3: Version history of best CEP50 per dataset
  Shows the research progression from v3 to v7

================================================================
IMPLEMENTATION ORDER
================================================================

Step 1: Implement MuDirectionLoss and update training loop 
        in GAT_V2025.py (Part 1)
        Quick verification: run 10 epochs on berlin1 only,
        check if direction reverses (mu_NLOS > mu_LOS)
        If not reversed in 10 epochs, increase MuDirectionLoss weight to 3.0

Step 2: Train exp_044-047 (full 100 epochs each, ~3.5 hours total)
        Run analyze_mog.py after each, verify direction_correct=True

Step 3: Rebuild inference caches for exp_044-047
        Delete: fusion/cache/berlin*_mog_outputs.pkl
                fusion/cache/frankfurt*_mog_outputs.pkl
        They will auto-rebuild on next evaluate run

Step 4: Implement methods 17-19 in baselines.py and prnc.py (Parts 3-4)
        Quick test on berlin1 only to check for crashes

Step 5: Full 20-method evaluation on all 4 datasets (Part 5)

Step 6: Run generate_final_report.py (Part 7)

Step 7: Run generate_paper_table.py (Part 8)

================================================================
CONSTRAINTS
================================================================
- Do NOT change GAT architecture, 11-dim features, block-diagonal batching
- MuDirectionLoss weight must be tunable — put it in config.py as 
  LAMBDA_MU_DIRECTION = 1.0
- The direction check (Part 6) must run BEFORE any positioning method
  If direction_correct=False for a dataset, flag it prominently
- Keep all 16 existing methods in evaluate_fusion.py unchanged
- All 20 methods must run on all 4 datasets without crash
- Cache exp_040-043 results are STALE — delete and rebuild for exp_044-047
- Minimum classification quality: F1 must stay >= 0.78 for all datasets
  If F1 drops below 0.78, MuDirectionLoss weight is too high, reduce it
- Print mu_nlos direction verdict at start of evaluation for each dataset
- If Step 1 quick test shows direction does NOT reverse in 10 epochs,
  also try: set LAMBDA_MU_REG to 0.0 (remove the L2 anchor that was 
  pulling mu toward 0.5) before increasing loss weight
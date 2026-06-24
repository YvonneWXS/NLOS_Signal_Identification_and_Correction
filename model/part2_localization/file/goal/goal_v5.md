Goal: Module 2 v5 — abandon WLS weighting, implement Pseudorange 
Residual NLOS Correction (PRNC) that preserves DOP geometry while 
using Module 1 p_los to gate bias correction. Also fix Module 1 
mu_nlos regression to produce physically meaningful values.

The v4 diagnosis is definitive: downweighting NLOS satellites 
destroys DOP geometry faster than it reduces NLOS error. The correct 
approach is to CORRECT pseudoranges (not weight satellites), so all 
satellites remain in the solution at equal weight.

================================================================
PART 0: THEORETICAL MOTIVATION (implement diagnostic first)
================================================================

File: fusion/verify_nlos_sign.py

Before implementing the correction algorithm, verify the physical
assumptions it relies on.

For ALL 4 datasets, load Module 1 cache outputs and compute:

--- Check 1: NLOS error sign distribution ---
Using ground truth labels (nlos_label field in observations):
  For LOS satellites (label=0):
    Compute pseudorange_error = pr_mes - (||gt_pos - sv_pos|| + clk_est)
    where clk_est = median(pr_mes_i - dist_i) over all satellites
    Report: mean, std, P5, P25, P75, P95 of errors
    Report: fraction with error > 0 (expected ~50% for symmetric noise)
    
  For NLOS satellites (label=1):  
    Same computation.
    Report: mean, std, P5, P25, P75, P95 of errors
    Report: fraction with error > 0 (expected >>50%, NLOS = positive delay)
    Report: mean positive error for NLOS with error > 0
    Report: fraction of NLOS with |error| > 0.3 km (correctable)

--- Check 2: Does p_los discriminate residual magnitude? ---
Bin satellites by p_los into 5 buckets: [0,0.2), [0.2,0.4), 
[0.4,0.6), [0.6,0.8), [0.8,1.0)
For each bucket, report: mean |error|, fraction NLOS ground truth

--- Check 3: Residual-based mu_nlos estimation quality ---
For NLOS satellites only, compare:
  mu_nlos_module1: learned value from MoG (from cache)
  mu_nlos_empirical: actual mean positive pseudorange error
Report per dataset: mean(mu_nlos_module1), mean(mu_nlos_empirical)
This quantifies how wrong Module 1 mu_nlos is.

Print a summary table. Expected results:
  NLOS fraction with error > 0: >65% (confirms NLOS = positive delay)
  mean NLOS error: 0.3-1.5 km range
  mu_nlos_module1 vs mu_nlos_empirical: expect 10-20x underestimate

Save output to cache/nlos_sign_analysis.json

================================================================
PART 1: MODULE 1 FIX — mu_nlos SUPERVISED REGRESSION
================================================================

The root cause of mu_nlos being 0.05-0.15 km (vs actual 0.5-1.5 km)
is that the current MoGNLLLoss does not use ground truth pseudorange
errors to supervise mu_nlos. Fix this in GAT_V2025.py.

File: part1_GAT/model/GAT_V2025.py

--- Add SupervisedMuRegressionLoss ---
In the NLL training stage (epoch 34+), add a direct regression 
supervision for mu_nlos using ground truth pseudorange error:

  class SupervisedMuRegressionLoss(nn.Module):
    def forward(self, mu_nlos, pseudorange_error, labels, p_los):
      # Only supervise on NLOS samples
      nlos_mask = (labels == 1)
      if not nlos_mask.any():
        return torch.tensor(0.0)
      
      # Target: actual positive pseudorange error for NLOS
      # Clamp to physical range [0, 3.0] km
      mu_target = torch.clamp(pseudorange_error[nlos_mask], 
                              min=0.0, max=3.0)
      mu_pred = mu_nlos[nlos_mask]
      
      # Huber loss (robust to outliers)
      loss = F.huber_loss(mu_pred, mu_target, delta=0.3)
      return loss

In the training loop, when USE_MIXTURE_GAUSSIAN=True:
  In NLL stage (epoch >= MOG_PURE_BCE_EPOCHS + MOG_BLEND_EPOCHS):
    loss_mu_reg = SupervisedMuRegressionLoss()(
        mu_nlos, pseudorange_error_batch, labels_batch, p_los)
    total_loss += 0.5 * loss_mu_reg  # weight 0.5

pseudorange_error_batch is already available in the training loop 
as it's used in uncertainty loss. Pass it through to the MoG loss.

Also update MU_NLOS_MAX in config.py: 500.0 → 3.0 (physical ceiling)
And MU_NLOS_TARGET in config.py: 0.15 → 0.5

--- Retrain all 4 models with supervised mu ---
Run training for all 4 datasets with new loss:
  python run_full_training.py --dataset berlin1_potsdamer_platz  → exp_040
  python run_full_training.py --dataset berlin2_gendarmenmarkt   → exp_041
  python run_full_training.py --dataset frankfurt1_maintower     → exp_042
  python run_full_training.py --dataset frankfurt2_westendtower  → exp_043

After training, run analyze_mog.py for each. 
Verify: mean mu_nlos for NLOS samples > 0.3 km (was 0.05-0.15)
Save analysis JSONs to result/exp_040-043/

--- Update Module 2 model mapping ---
In run_fusion.py, update to use exp_040-043.
Rebuild MoG inference caches: delete cache/*_mog_outputs.pkl
so they are regenerated with new models.

================================================================
PART 2: PSEUDORANGE RESIDUAL NLOS CORRECTION (PRNC)
================================================================

File: fusion/prnc.py (NEW FILE)

Core algorithm — Pseudorange Residual NLOS Correction:

class PRNCPositioner:
  """
  Corrects NLOS pseudorange biases using residuals from an 
  initial LS solution, gated by Module 1 p_los predictions.
  Preserves uniform satellite weights (no DOP degradation).
  """
  
  def solve_epoch(self, obs_list, sv_positions, mog_outputs,
                  max_iters=5, convergence_threshold=0.001):
    """
    obs_list: list of observations with pr_mes (km)
    sv_positions: (N, 3) ECEF km
    mog_outputs: dict with p_los, mu_nlos, sigma_los, sigma_nlos
    Returns: position (3,) ECEF km, clock_bias (km), diagnostics
    """
    
    pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
    p_los  = mog_outputs['p_los']     # (N,)
    p_nlos = 1.0 - p_los
    sigma_los = mog_outputs['sigma_los']  # (N,)
    
    # Step 1: Initial Standard LS solution (uniform weights)
    x, clk = run_standard_ls(pr_mes, sv_positions)
    
    prev_x = x.copy()
    
    for iteration in range(max_iters):
      # Step 2: Compute residuals
      dists = np.linalg.norm(sv_positions - x[np.newaxis,:], axis=1)
      predicted_pr = dists + clk
      residuals = pr_mes - predicted_pr   # (N,)
      
      # Step 3: Estimate NLOS correction for each satellite
      # NLOS errors are predominantly positive (signal delay)
      # Noise floor = 2 * sigma_los to avoid correcting LOS noise
      noise_floor = 2.0 * sigma_los   # (N,) in km
      
      # Positive residual above noise floor = likely NLOS delay
      excess_positive = np.maximum(0.0, residuals - noise_floor)
      
      # Gate by p_nlos: only correct satellites Module 1 thinks are NLOS
      # Soft gate: correction scales with NLOS probability
      nlos_correction = p_nlos * excess_positive
      
      # Additional gate: only apply correction if p_los < 0.6 AND
      # residual > noise_floor (both Module 1 AND residual agree)
      gate = (p_los < 0.6).astype(float) * (residuals > noise_floor).astype(float)
      nlos_correction = nlos_correction * gate
      
      # Step 4: Apply correction to pseudoranges
      pr_corrected = pr_mes - nlos_correction
      
      # Step 5: Run Standard LS on corrected pseudoranges
      # IMPORTANT: uniform weights — preserves DOP geometry
      x_new, clk_new = run_standard_ls(pr_corrected, sv_positions)
      
      # Step 6: Check convergence
      delta = np.linalg.norm(x_new - prev_x) * 1000  # meters
      x = x_new
      clk = clk_new
      prev_x = x.copy()
      
      if delta < convergence_threshold:
        break
    
    diagnostics = {
      'iterations': iteration + 1,
      'final_residuals': residuals,
      'nlos_corrections': nlos_correction,
      'num_corrected': int(gate.sum()),
      'mean_correction_km': float(nlos_correction[gate>0].mean()) 
                            if gate.sum() > 0 else 0.0
    }
    return x, clk, diagnostics

  def solve_with_mu_nlos(self, obs_list, sv_positions, mog_outputs,
                          max_iters=5):
    """
    Variant: use Module 1 mu_nlos directly as correction (after fix)
    For comparison with residual-based approach.
    """
    pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
    p_nlos = 1.0 - mog_outputs['p_los']
    mu_nlos = mog_outputs['mu_nlos']
    
    # Direct mu_nlos correction: subtract expected NLOS bias
    # Soft correction: scale by probability
    nlos_correction = p_nlos * mu_nlos  # (N,)
    pr_corrected = pr_mes - nlos_correction
    
    # Standard LS on corrected pseudoranges (uniform weights)
    x, clk = run_standard_ls(pr_corrected, sv_positions)
    
    return x, clk, {'corrections': nlos_correction}

--- Helper: run_standard_ls ---
def run_standard_ls(pr_mes, sv_positions, max_iter=10):
  """Standard iterative LS, uniform weights, returns (pos_ecef, clk)."""
  # Initialize at centroid of sv_positions (Earth center approximately)
  x = np.array([0.0, 0.0, 6371.0])  # rough Earth center in ECEF km
  clk = np.median(pr_mes - np.linalg.norm(sv_positions - x, axis=1))
  
  for _ in range(max_iter):
    dists = np.linalg.norm(sv_positions - x[np.newaxis,:], axis=1)
    predicted = dists + clk
    residuals = pr_mes - predicted
    
    los_vecs = (sv_positions - x[np.newaxis,:]) / dists[:,np.newaxis]
    H = np.hstack([-los_vecs, np.ones((len(pr_mes),1))])
    
    try:
      delta = np.linalg.lstsq(H, residuals, rcond=None)[0]
    except:
      break
    
    x += delta[:3]
    clk += delta[3]
    
    if np.linalg.norm(delta[:3]) * 1000 < 0.1:  # 0.1m convergence
      break
  
  return x, clk

================================================================
PART 3: MULTI-ITERATION PRNC WITH ADAPTIVE GATE
================================================================

File: fusion/prnc.py (add to same file)

The basic PRNC applies a fixed gate. Implement an adaptive variant
that adjusts the gate threshold per epoch based on signal quality:

class AdaptivePRNCPositioner(PRNCPositioner):
  
  def solve_epoch_adaptive(self, obs_list, sv_positions, mog_outputs,
                            max_iters=7):
    pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
    p_los = mog_outputs['p_los']
    sigma_los = mog_outputs['sigma_los']
    cno = np.array([obs.cno for obs in obs_list])
    
    # Adaptive noise floor based on CNO
    # Low CNO = noisier = higher threshold before correction kicks in
    cno_normalized = np.clip(cno / 45.0, 0.3, 1.0)  # normalize to [0.3, 1.0]
    noise_floor = sigma_los * (1.0 + 2.0 * (1.0 - cno_normalized))
    
    # Two-stage: coarse correction first, then fine
    # Stage 1: correct obviously-NLOS satellites (p_los < 0.3, big residual)
    # Stage 2: soft-correct ambiguous satellites (0.3 < p_los < 0.6)
    
    x, clk = run_standard_ls(pr_mes, sv_positions)
    
    for iteration in range(max_iters):
      dists = np.linalg.norm(sv_positions - x[np.newaxis,:], axis=1)
      residuals = pr_mes - dists - clk
      
      # Stage 1: high-confidence NLOS correction
      gate_hard = (p_los < 0.3) & (residuals > noise_floor)
      correction_hard = np.where(gate_hard, residuals - noise_floor, 0.0)
      
      # Stage 2: soft correction for ambiguous satellites
      gate_soft = (p_los >= 0.3) & (p_los < 0.6) & (residuals > noise_floor)
      soft_weight = (0.6 - p_los) / 0.3  # 1.0 at p_los=0.3, 0.0 at p_los=0.6
      correction_soft = np.where(gate_soft, 
                                  soft_weight * (residuals - noise_floor) * 0.5, 
                                  0.0)
      
      total_correction = correction_hard + correction_soft
      pr_corrected = pr_mes - total_correction
      
      x_new, clk_new = run_standard_ls(pr_corrected, sv_positions)
      
      if np.linalg.norm(x_new - x) * 1000 < 0.5:
        break
      x, clk = x_new, clk_new
    
    return x, clk, {}

================================================================
PART 4: TCN-ENHANCED PRNC
================================================================

File: fusion/prnc.py (add method)

After PRNC correction, use TCN temporal prior to further refine
the satellite quality estimate before the next PRNC iteration:

class PRNCWithTCN(AdaptivePRNCPositioner):
  
  def solve_epoch_with_tcn(self, obs_list, sv_positions, mog_outputs,
                             tcn_prior, epoch_idx):
    """
    tcn_prior: dict with 'p_nlos_prior' (20,) and 'confidence' (20,)
    """
    if epoch_idx < 10 or tcn_prior is None:
      return self.solve_epoch_adaptive(obs_list, sv_positions, mog_outputs)
    
    # Blend Module 1 p_los with TCN prior
    p_los_gat = mog_outputs['p_los'].copy()
    p_nlos_tcn = tcn_prior['p_nlos_prior'][:len(p_los_gat)]
    conf = tcn_prior['confidence'][:len(p_los_gat)]
    
    # Soft blend: alpha = confidence * |p_nlos_tcn - 0.5| * 2, capped at 0.25
    alpha = np.clip(conf * np.abs(p_nlos_tcn - 0.5) * 2, 0, 0.25)
    p_los_blended = (1 - alpha) * p_los_gat + alpha * (1 - p_nlos_tcn)
    
    # Only blend when TCN disagrees with Module 1
    disagree = ((p_nlos_tcn > 0.6) & (p_los_gat > 0.5)) | \
               ((p_nlos_tcn < 0.4) & (p_los_gat < 0.5))
    p_los_final = np.where(disagree, p_los_blended, p_los_gat)
    
    mog_updated = dict(mog_outputs)
    mog_updated['p_los'] = p_los_final
    
    return self.solve_epoch_adaptive(obs_list, sv_positions, mog_updated)

================================================================
PART 5: FULL EVALUATION — ALL METHODS
================================================================

File: fusion/evaluate_fusion.py — add PRNC methods, update to v5

New method list (12 methods total):
  1.  Standard LS (baseline)
  2.  WLS-elevation (baseline)
  3.  WLS-MoG-linear (v3 best)
  4.  WLS-debiased (v4, kept for comparison)
  5.  RAIM-MoG (v4, kept)
  6.  PRNC-basic (Part 2, basic residual correction)
  7.  PRNC-mu (Part 2, use Module 1 mu_nlos directly, after fix)
  8.  PRNC-adaptive (Part 3, adaptive gate)
  9.  PRNC-adaptive+2A (Part 4, with TCN)
  10. PRNC-mu-adaptive (combination: mu_nlos + adaptive gate)

For each method, report:
  CEP50, CEP95, Mean2D, RMSE3D
  % improvement vs Standard LS (positive = better)
  % improvement vs WLS-MoG-linear (shows gain from PRNC vs weighting)

Also report for PRNC methods:
  Mean num satellites corrected per epoch
  Mean correction applied (km)
  Fraction of epochs where PRNC improved vs Standard LS

Save full results to fusion/result/exp_v5/positioning_results_v5.json

--- Expected behavior ---
PRNC should NOT degrade DOP (all sats kept at uniform weight).
PRNC should beat Standard LS when:
  1. NLOS ratio > 30% (berlin1 48.3%, frankfurt1 43.0%)
  2. NLOS errors are predominantly positive (verified in PART 0)
  3. p_los provides good LOS/NLOS discrimination (gap > 0.5)

If PRNC-basic fails: check if correction is being applied 
(mean_correction_km should be > 0.1 km in berlin1/frankfurt1).
If correction is near zero, the gate conditions are too strict.
Try relaxing: p_los < 0.7 (was 0.6) and residual > 0.5*sigma_los.

================================================================
PART 6: PER-EPOCH DIAGNOSTIC REPORT
================================================================

File: fusion/prnc_diagnostics.py (NEW)

For one representative epoch per dataset (epoch index 100),
generate a detailed per-satellite diagnostic:

For each satellite in the epoch:
  Print: svid, elevation, cno, nlos_label(GT), p_los, residual_before,
         nlos_correction_applied, residual_after, was_corrected

Print epoch-level summary:
  - Position error before PRNC (m)
  - Position error after PRNC (m)
  - Number of satellites corrected
  - Mean correction applied (km)
  - DOP before and after (should be identical)

Also generate the per-epoch correction statistics for full dataset:
  - Distribution of corrections applied (histogram, 10 bins)
  - Correlation between correction magnitude and actual error reduction
  - Fraction of corrections that "helped" (positive → negative residual change)
  - Fraction that "hurt" (applied to LOS satellite, making residual worse)

Save to fusion/result/exp_v5/prnc_diagnostics_{dataset}.json

================================================================
PART 7: MODULE 1 QUALITY ASSESSMENT FOR MODULE 2 READINESS
================================================================

File: fusion/module1_quality_report.py (NEW)

After retraining Module 1 with supervised mu (PART 1), assess
whether Module 2 PRNC now has sufficient input quality to work.

Check the following across all 4 datasets:

  1. mu_nlos accuracy: 
     Compare mu_nlos_module1 vs mu_nlos_empirical per satellite
     Metric: mean absolute error of mu_nlos prediction
     Target: MAE < 0.3 km (was ~0.5 km before fix)
     
  2. p_los discrimination quality for PRNC gating:
     Compute: fraction of corrections correctly applied to NLOS sats
     = (corrected sats that ARE NLOS) / (all corrected sats)
     Target: > 70% (precision of correction gating)
     
  3. Effective correction magnitude:
     Mean correction applied to NLOS sats after Module 1 fix
     Target: > 0.2 km (was near 0 before fix)
     
  4. DOP preservation check:
     Compare mean PDOP(Standard LS) vs mean PDOP(PRNC)
     They should be identical (or within 0.01)
     
Print a readiness report:
  [PASS/FAIL] mu_nlos MAE < 0.3 km: {dataset1} {dataset2} {dataset3} {dataset4}
  [PASS/FAIL] correction precision > 70%: ...
  [PASS/FAIL] DOP preserved (PRNC == Standard LS PDOP): ...
  [PASS/FAIL] PRNC-adaptive beats Standard LS in >=2/4 datasets by >3%: ...
  
  OVERALL: [READY / NOT READY] for Module 3

================================================================
IMPLEMENTATION ORDER
================================================================

Run these steps IN ORDER. Do not skip ahead.

Step 1: Run fusion/verify_nlos_sign.py
  → Confirm NLOS errors are predominantly positive
  → Get mu_nlos_empirical values for reference
  → If NLOS fraction positive < 55%, stop and report (unexpected)

Step 2: Implement fusion/prnc.py (Parts 2-4)
  → Test on berlin1 only first (1377 epochs, fastest)
  → Run PRNC-basic on berlin1, check if it beats Standard LS
  → If yes, proceed. If no, check mean_correction_km in diagnostics

Step 3: Update Module 1 (Part 1)
  → Add SupervisedMuRegressionLoss to GAT_V2025.py
  → Train exp_040-043
  → Verify mu_nlos improved before proceeding

Step 4: Rebuild inference caches with new models (exp_040-043)
  → Delete fusion/cache/*_mog_outputs.pkl
  → Run run_fusion.py which will rebuild them

Step 5: Full 12-method evaluation (Part 5)
  → Run all methods on all 4 datasets
  → Report improvement table

Step 6: Run per-epoch diagnostics (Part 6)
  → Understand where PRNC helps and where it fails

Step 7: Run Module 1 quality assessment (Part 7)
  → Determine if Module 2 is ready for Module 3

================================================================
SUCCESS CRITERIA
================================================================

Module 2 v5 is COMPLETE if ALL of the following hold:

[C1] PRNC-adaptive beats Standard LS in >=2/4 datasets by >3% CEP50
[C2] PRNC does NOT degrade DOP vs Standard LS (PDOP unchanged)
[C3] mu_nlos_module1 MAE < 0.3 km after Module 1 retraining
[C4] PRNC-adaptive beats WLS-MoG-linear in ALL 4 datasets
     (PRNC strictly dominates the best v3/v4 WLS approach)
[C5] PRNC+2A does NOT degrade vs PRNC-adaptive in any dataset
[C6] NLOS correction precision > 70% 
     (>70% of corrected satellites are actually NLOS)

Print pass/fail for each criterion at end of run_fusion.py.
If [C1] fails but [C4] passes: acceptable partial success.
If [C1] and [C4] both fail: escalate with full diagnostic dump.

================================================================
CONSTRAINTS
================================================================
- PRNC must use UNIFORM weights in final LS solve (no DOP change)
- Module 1 retraining uses existing 3-stage schedule, only adds 
  SupervisedMuRegressionLoss in NLL stage (epoch 34+)
- Do NOT change GAT architecture, input features, or training epochs
- Preserve all previous methods in evaluate_fusion.py
- Cache Module 1 outputs per model version 
  (cache/{dataset}_mog_outputs_exp040.pkl etc.)
- All 12 methods run on all 4 datasets without crash
- Print timing for each method (wall-clock seconds)
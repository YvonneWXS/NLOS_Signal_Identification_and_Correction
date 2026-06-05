Goal: Module 2 v6 — Fix clock bias contamination, implement 
LOS-anchored positioning, and achieve WLS-MoG better than 
Standard LS in at least 2/4 datasets.

The v5 verify_nlos_sign.py found NLOS mean errors of only 4-72m 
after clock absorption, but mu_empirical is 166-236m. This 3-10x 
discrepancy proves the clock estimate is ABSORBING the NLOS bias.
When ~40-48% of satellites are NLOS with positive delays, the 
standard median clock estimate is systematically biased upward,
which then makes NLOS residuals appear symmetric (zero-mean).

The consequence: ALL residual-based corrections and ALL WLS 
approaches use a contaminated clock, so the corrections fight 
their own contaminated reference frame.

================================================================
PART 0: VERIFY THE CLOCK CONTAMINATION HYPOTHESIS
================================================================

File: fusion/verify_clock_contamination.py

Step 1 — Contaminated vs clean clock comparison:
  For each dataset, for each epoch:
  
  Method A (current — contaminated):
    clk_contaminated = median(pr_mes_i - dist_i) over ALL satellites
    residuals_A_i = pr_mes_i - dist_i - clk_contaminated
    
  Method B (LOS-anchored):
    high_los_mask = (p_los_i > 0.7)  # use only confident LOS sats
    if sum(high_los_mask) >= 4:
      clk_los = median(pr_mes_i[high_los] - dist_i[high_los])
    else:
      clk_los = clk_contaminated  # fallback if not enough LOS sats
    residuals_B_i = pr_mes_i - dist_i - clk_los
  
  For each dataset, compute and print:
    a) mean(clk_contaminated) vs mean(clk_los)
    b) delta_clk = clk_los - clk_contaminated (expected: positive, 
       since contaminated clock is biased high by NLOS)
    c) For NLOS satellites only:
       mean(residuals_A) vs mean(residuals_B)  
       Expected: residuals_B should be LARGER and MORE POSITIVE
    d) For LOS satellites only:
       mean(residuals_A) vs mean(residuals_B)
       Expected: residuals_B should be close to 0 (LOS is clean)
    e) NLOS fraction with residuals_B > 0 (expected: >65% vs current 39-53%)
    f) NLOS mean(residuals_B) in meters (expected: >200m)

Step 2 — Minimum LOS satellite count:
  For each dataset, print histogram: 
    how many epochs have high_los_mask.sum() >= 4?
    how many have >= 5? >= 6?
  This tells us how often the LOS-anchored clock is usable.

Step 3 — Print diagnosis:
  If mean(residuals_B[NLOS]) > 150m and NLOS_frac_positive > 60%:
    print "CONFIRMED: Clock contamination is the root cause"
    print "LOS-anchored clock reveals true NLOS positive bias"
  Else:
    print "Clock contamination NOT the root cause, investigating further"
    print per-dataset diagnostics

Save results to cache/clock_contamination_analysis.json

================================================================
PART 1: LOS-ANCHORED ITERATIVE LS (Core Fix)
================================================================

File: fusion/los_anchored_ls.py (NEW FILE)

Implement 4 positioning methods that use LOS-anchored clock:

--- Method 1: LOS-Anchored Standard LS ---

def solve_los_anchored_ls(obs_list, sv_positions, p_los, 
                           sigma_los, max_iter=10):
  """
  Key idea: estimate clock using high-confidence LOS satellites only,
  then run LS with corrected residuals using the clean clock.
  """
  pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
  
  # Initialize position
  x = np.zeros(3)
  x[2] = 6371.0  # rough Earth radius km
  
  for iteration in range(max_iter):
    dists = np.linalg.norm(sv_positions - x[np.newaxis,:], axis=1)
    
    # LOS-anchored clock estimation
    high_los = (p_los > 0.7)
    if high_los.sum() >= 4:
      raw_residuals = pr_mes - dists
      clk = np.median(raw_residuals[high_los])
    else:
      # Fallback: use all sats but weight by p_los for median
      raw_residuals = pr_mes - dists
      # Weighted median approximation: sort by p_los, take upper half
      sorted_idx = np.argsort(p_los)[::-1]
      top_half = sorted_idx[:max(4, len(sorted_idx)//2)]
      clk = np.median(raw_residuals[top_half])
    
    residuals = pr_mes - dists - clk
    
    # Standard LS: uniform weights, all satellites
    los_vecs = (sv_positions - x[np.newaxis,:]) / dists[:,np.newaxis]
    H = np.hstack([-los_vecs, np.ones((len(pr_mes),1))])
    
    # Note: clock is pre-estimated, only solve for position
    # H_pos = H[:, :3]  but we still include clock for consistency
    try:
      delta = np.linalg.lstsq(H, residuals, rcond=None)[0]
    except:
      break
    
    x += delta[:3]
    # DO NOT update clk from LS delta[3] — keep LOS-anchored clock
    # The LS clock correction goes to position only
    
    if np.linalg.norm(delta[:3]) * 1000 < 0.1:
      break
  
  return x, clk

--- Method 2: LOS-Anchored WLS-MoG ---

def solve_los_anchored_wls_mog(obs_list, sv_positions, p_los, 
                                sigma_los, mu_nlos, max_iter=10):
  """
  Uses LOS-anchored clock + WLS weights from Module 1.
  Separates clock estimation from weighting.
  """
  pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
  x = np.zeros(3)
  x[2] = 6371.0
  
  for iteration in range(max_iter):
    dists = np.linalg.norm(sv_positions - x[np.newaxis,:], axis=1)
    
    # LOS-anchored clock (same as above)
    high_los = (p_los > 0.7)
    raw_residuals = pr_mes - dists
    if high_los.sum() >= 4:
      clk = np.median(raw_residuals[high_los])
    else:
      sorted_idx = np.argsort(p_los)[::-1]
      top_half = sorted_idx[:max(4, len(sorted_idx)//2)]
      clk = np.median(raw_residuals[top_half])
    
    residuals = pr_mes - dists - clk
    
    # WLS weights from Module 1
    weights = np.maximum(0.01, p_los) / np.maximum(0.01, sigma_los**2)
    W = np.diag(weights)
    
    los_vecs = (sv_positions - x[np.newaxis,:]) / dists[:,np.newaxis]
    H = np.hstack([-los_vecs, np.ones((len(pr_mes),1))])
    
    try:
      delta = np.linalg.lstsq(np.sqrt(W) @ H, 
                               np.sqrt(W) @ residuals, rcond=None)[0]
    except:
      break
    
    x += delta[:3]
    if np.linalg.norm(delta[:3]) * 1000 < 0.1:
      break
  
  return x, clk

--- Method 3: LOS-Anchored PRNC ---

def solve_los_anchored_prnc(obs_list, sv_positions, p_los, 
                              sigma_los, mu_nlos, max_iter=7):
  """
  LOS-anchored clock + pseudorange residual correction.
  With clean clock, NLOS residuals should be consistently positive.
  """
  pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
  x = np.zeros(3)
  x[2] = 6371.0
  
  for iteration in range(max_iter):
    dists = np.linalg.norm(sv_positions - x[np.newaxis,:], axis=1)
    
    # LOS-anchored clock
    high_los = (p_los > 0.7)
    raw_residuals = pr_mes - dists
    if high_los.sum() >= 4:
      clk = np.median(raw_residuals[high_los])
    else:
      sorted_idx = np.argsort(p_los)[::-1]
      top_half = sorted_idx[:max(4, len(sorted_idx)//2)]
      clk = np.median(raw_residuals[top_half])
    
    residuals = pr_mes - dists - clk
    
    # PRNC: now residuals should be positive for NLOS
    # Only correct positive residuals above LOS noise floor
    noise_floor = 2.0 * sigma_los
    excess = np.maximum(0.0, residuals - noise_floor)
    
    # Gate: p_los < 0.5 AND positive excess residual
    gate = ((1.0 - p_los) * (residuals > noise_floor)).clip(0, 1)
    correction = gate * excess * (1.0 - p_los)
    
    pr_corrected = pr_mes - correction
    
    # Recompute with corrected pseudoranges
    raw_residuals_corr = pr_corrected - dists
    if high_los.sum() >= 4:
      clk = np.median(raw_residuals_corr[high_los])
    else:
      clk = np.median(raw_residuals_corr[top_half])
    
    residuals_corr = pr_corrected - dists - clk
    
    los_vecs = (sv_positions - x[np.newaxis,:]) / dists[:,np.newaxis]
    H_pos = -los_vecs
    H_full = np.hstack([H_pos, np.ones((len(pr_mes),1))])
    
    try:
      delta = np.linalg.lstsq(H_full, residuals_corr, rcond=None)[0]
    except:
      break
    
    x += delta[:3]
    if np.linalg.norm(delta[:3]) * 1000 < 0.1:
      break
  
  return x, clk

--- Method 4: LOS-Anchored mu_nlos Debiased WLS ---

def solve_los_anchored_debiased_wls(obs_list, sv_positions, p_los, 
                                      sigma_los, mu_nlos, max_iter=10):
  """
  LOS-anchored clock + subtract expected NLOS bias + WLS weights.
  This is the theoretically optimal combination.
  """
  pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
  p_nlos = 1.0 - p_los
  
  # Correct pseudoranges by subtracting expected NLOS delay
  pr_corrected = pr_mes - p_nlos * mu_nlos
  
  x = np.zeros(3)
  x[2] = 6371.0
  
  for iteration in range(max_iter):
    dists = np.linalg.norm(sv_positions - x[np.newaxis,:], axis=1)
    
    # LOS-anchored clock on CORRECTED pseudoranges
    high_los = (p_los > 0.7)
    raw_residuals = pr_corrected - dists
    if high_los.sum() >= 4:
      clk = np.median(raw_residuals[high_los])
    else:
      sorted_idx = np.argsort(p_los)[::-1]
      top_half = sorted_idx[:max(4, len(sorted_idx)//2)]
      clk = np.median(raw_residuals[top_half])
    
    residuals = pr_corrected - dists - clk
    
    # WLS weights
    weights = np.maximum(0.01, p_los) / np.maximum(0.01, sigma_los**2)
    W = np.diag(weights)
    
    los_vecs = (sv_positions - x[np.newaxis,:]) / dists[:,np.newaxis]
    H = np.hstack([-los_vecs, np.ones((len(pr_mes),1))])
    
    try:
      delta = np.linalg.lstsq(np.sqrt(W) @ H, 
                               np.sqrt(W) @ residuals, rcond=None)[0]
    except:
      break
    
    x += delta[:3]
    if np.linalg.norm(delta[:3]) * 1000 < 0.1:
      break
  
  return x, clk

================================================================
PART 2: GEOMETRY-AWARE SATELLITE SELECTION
================================================================

File: fusion/los_anchored_ls.py (add to same file)

The v4 diagnosis found that DOP inflation is a major failure mode
(frankfurt1: 58.2% of epochs get worse DOP with WLS).
Add a geometry-aware satellite selection that prevents DOP degradation:

def select_satellites_geometry_aware(sv_positions, p_los, sigma_los, 
                                       rx_pos_approx, min_sats=5):
  """
  Returns a subset of satellites that:
    1. Does not increase PDOP by more than 20% vs using all sats
    2. Preferentially keeps high-p_los (LOS) satellites
    3. Always keeps >= min_sats satellites
  
  Algorithm:
    Start with all satellites. 
    Compute baseline PDOP.
    Try removing the satellite with lowest p_los (most likely NLOS).
    If PDOP after removal < 1.2 * baseline_PDOP:
      Accept removal, update baseline.
    Repeat until no more satellites can be removed without DOP penalty.
  """
  n = len(p_los)
  active = list(range(n))
  
  def compute_pdop(sat_indices, rx_pos):
    sv_sel = sv_positions[sat_indices]
    dists = np.linalg.norm(sv_sel - rx_pos[np.newaxis,:], axis=1)
    los_vecs = (sv_sel - rx_pos[np.newaxis,:]) / dists[:,np.newaxis]
    H = np.hstack([-los_vecs, np.ones((len(sat_indices),1))])
    try:
      P = np.linalg.inv(H.T @ H)
      return np.sqrt(P[0,0] + P[1,1] + P[2,2])
    except:
      return 999.0
  
  baseline_pdop = compute_pdop(active, rx_pos_approx)
  
  # Sort candidates by p_los (ascending = most likely NLOS first)
  candidates = sorted(active, key=lambda i: p_los[i])
  
  for candidate in candidates:
    if len(active) <= min_sats:
      break
    if p_los[candidate] > 0.5:
      break  # don't remove satellites Module 1 thinks are LOS
    
    trial = [i for i in active if i != candidate]
    trial_pdop = compute_pdop(trial, rx_pos_approx)
    
    if trial_pdop <= 1.2 * baseline_pdop:
      active = trial
      baseline_pdop = trial_pdop
  
  return np.array(active)

--- Method 5: Full Combined Method ---

def solve_los_anchored_combined(obs_list, sv_positions, p_los,
                                  sigma_los, mu_nlos, max_iter=10):
  """
  Best of everything:
    1. LOS-anchored clock
    2. Geometry-aware satellite selection (no DOP degradation)
    3. mu_nlos debiasing on selected satellites
    4. WLS weights on selected satellites
  """
  pr_mes = np.array([obs.pr_mes_km for obs in obs_list])
  p_nlos = 1.0 - p_los
  
  # Initial position estimate
  x = np.zeros(3); x[2] = 6371.0
  dists_init = np.linalg.norm(sv_positions - x[np.newaxis,:], axis=1)
  
  # Geometry-aware satellite selection
  selected = select_satellites_geometry_aware(
      sv_positions, p_los, sigma_los, x)
  
  sv_sel = sv_positions[selected]
  pr_sel = pr_mes[selected]
  p_los_sel = p_los[selected]
  p_nlos_sel = p_nlos[selected]
  sigma_sel = sigma_los[selected]
  mu_sel = mu_nlos[selected]
  
  # Debias selected pseudoranges
  pr_corrected = pr_sel - p_nlos_sel * mu_sel
  
  for iteration in range(max_iter):
    dists = np.linalg.norm(sv_sel - x[np.newaxis,:], axis=1)
    
    # LOS-anchored clock
    high_los = (p_los_sel > 0.7)
    raw_res = pr_corrected - dists
    if high_los.sum() >= 4:
      clk = np.median(raw_res[high_los])
    else:
      top = np.argsort(p_los_sel)[::-1][:max(4, len(p_los_sel)//2)]
      clk = np.median(raw_res[top])
    
    residuals = pr_corrected - dists - clk
    weights = np.maximum(0.01, p_los_sel) / np.maximum(0.01, sigma_sel**2)
    W = np.diag(weights)
    
    los_vecs = (sv_sel - x[np.newaxis,:]) / dists[:,np.newaxis]
    H = np.hstack([-los_vecs, np.ones((len(pr_corrected),1))])
    
    try:
      delta = np.linalg.lstsq(np.sqrt(W) @ H,
                               np.sqrt(W) @ residuals, rcond=None)[0]
    except:
      break
    
    x += delta[:3]
    if np.linalg.norm(delta[:3]) * 1000 < 0.1:
      break
  
  return x, clk

================================================================
PART 3: ROBUST CLOCK ESTIMATION ALTERNATIVES
================================================================

File: fusion/los_anchored_ls.py (add to same file)

Implement 2 more robust clock estimation strategies as fallbacks:

--- Clock Strategy A: Iterative sigma-clipping ---
def estimate_clock_sigma_clip(pr_mes, dists, p_los, n_iter=3):
  """Remove outliers iteratively before clock estimation."""
  residuals = pr_mes - dists
  active = np.ones(len(residuals), dtype=bool)
  
  for _ in range(n_iter):
    clk = np.median(residuals[active])
    res_centered = residuals - clk
    mad = np.median(np.abs(res_centered[active]))
    sigma = 1.4826 * mad  # MAD-based std estimate
    # Remove outliers: > 2.5 sigma AND low p_los
    outlier = (np.abs(res_centered) > 2.5 * sigma) & (p_los < 0.4)
    if active.sum() - outlier[active].sum() < 4:
      break  # don't remove too many
    active[outlier] = False
  
  return np.median(residuals[active])

--- Clock Strategy B: p_los weighted median ---
def estimate_clock_weighted_median(pr_mes, dists, p_los):
  """Approximate weighted median: replicate points by weight."""
  residuals = pr_mes - dists
  # Integer weights from p_los
  weights = np.maximum(1, np.round(p_los * 10).astype(int))
  repeated = np.repeat(residuals, weights)
  return np.median(repeated)

These strategies are used as alternatives to the strict p_los > 0.7
threshold when there are not enough confident LOS satellites.

================================================================
PART 4: FULL 16-METHOD EVALUATION
================================================================

File: fusion/evaluate_fusion.py — update to v6, 16 methods

Add to the existing 12 methods:
  13. LOS-Anchored-LS (Method 1 from Part 1)
  14. LOS-Anchored-WLS-MoG (Method 2)
  15. LOS-Anchored-PRNC (Method 3)
  16. LOS-Anchored-Combined (Method 5: geometry-aware + debiased + WLS)

For each new method, report:
  CEP50 (m), CEP95 (m), Mean 2D (m), RMSE 3D (m)
  % improvement vs Standard LS
  Mean clock bias estimate (contaminated vs LOS-anchored, for diagnostics)
  Mean fraction of high-confidence LOS sats available per epoch

Save to fusion/result/exp_v6/positioning_results_v6.json

Additional per-epoch analysis for LOS-Anchored methods:
  - Distribution of delta_clk = clk_los_anchored - clk_contaminated
    (this shows how much contamination was in the clock)
  - Correlation between |delta_clk| and positioning improvement
  - Fraction of epochs where LOS-anchored clock improved CEP50

================================================================
PART 5: FINAL DIAGNOSIS AND VERIFICATION
================================================================

File: fusion/verify_clock_contamination.py (extend with verification)

After running all 16 methods, add a verification section:

For each dataset, print:
  --- Clock contamination analysis ---
  Mean contaminated clock: {val} km
  Mean LOS-anchored clock: {val} km  
  Mean clock delta: {val} km ({val} m)
  Expected direction: LOS-anchored > contaminated (NLOS delays clock down)
  Actual direction: [CONFIRMED / REVERSED]
  
  --- NLOS residual analysis with clean clock ---
  Contaminated clock: NLOS mean residual = {val} m, frac>0 = {val}%
  LOS-anchored clock: NLOS mean residual = {val} m, frac>0 = {val}%
  Improvement: {val}x better NLOS signal visibility
  
  --- Positioning impact ---
  Standard LS CEP50: {val} m
  LOS-Anchored-LS CEP50: {val} m (delta: {val}%)
  LOS-Anchored-Combined CEP50: {val} m (delta: {val}%)

================================================================
PART 6: SUCCESS CRITERIA REPORT
================================================================

Print at the end of run_fusion.py:

=== Module 2 v6 Success Criteria ===

[PASS/FAIL] Clock contamination confirmed (LOS-anchored > contaminated)
  berlin1 delta_clk: {val}m | berlin2: {val}m | frk1: {val}m | frk2: {val}m

[PASS/FAIL] LOS-Anchored-LS beats Standard LS (contaminated clock) 
  in >=2/4 datasets by >3% CEP50
  berlin1: {val}% | berlin2: {val}% | frk1: {val}% | frk2: {val}%

[PASS/FAIL] LOS-Anchored-Combined beats WLS-MoG (contaminated)
  in >=3/4 datasets
  berlin1: {val}% | berlin2: {val}% | frk1: {val}% | frk2: {val}%

[PASS/FAIL] LOS-Anchored-PRNC NLOS residuals predominantly positive
  (frac_positive > 60% with clean clock)
  berlin1: {val}% | berlin2: {val}% | frk1: {val}% | frk2: {val}%

[PASS/FAIL] Geometry-aware selection: no DOP inflation
  (PDOP(selected) <= 1.0 * PDOP(all) in >90% of epochs)

[PASS/FAIL] LOS-Anchored-Combined beats Standard LS in >=2/4 datasets
  by > 5% CEP50 — PRIMARY SUCCESS CRITERION
  berlin1: {val}% | berlin2: {val}% | frk1: {val}% | frk2: {val}%

If PRIMARY criterion PASSES: 
  print "Module 2 COMPLETE — ready for Module 3"
  print summary table with best method per dataset
  
If PRIMARY criterion FAILS:
  print detailed per-epoch analysis for the highest-NLOS dataset
  print fraction of epochs with >= 4 high-confidence LOS sats
  print recommendation for Module 3 integration

================================================================
PART 7: COMPARATIVE TABLE FOR PAPER
================================================================

File: fusion/generate_paper_table.py (NEW)

Generate a LaTeX-formatted table comparing all meaningful methods.
Include only the non-redundant methods for clarity:
  Standard LS | WLS-MoG (v3, best WLS) | LOS-WLS-MoG | LOS-Combined
  
For each: CEP50, CEP95, improvement vs Standard LS in %

Also generate a summary of the overall trajectory:
  v3: WLS approaches (best: frankfurt1 +11.7%, others worse)
  v4: 6 WLS variants (all fail, root cause: DOP+clock)
  v5: PRNC correction (fails due to symmetric NLOS errors)
  v6: Clock-contamination fix (result: TBD)

This table will be used in the final paper to demonstrate the
analysis contribution (identifying clock contamination) even if
positioning improvement is modest.

================================================================
IMPLEMENTATION ORDER (STRICTLY FOLLOW)
================================================================

Step 1: Run verify_clock_contamination.py (PART 0)
  - If clock contamination NOT confirmed (delta_clk near 0):
    STOP and print "hypothesis rejected, need different approach"
  - If confirmed: proceed to Step 2

Step 2: Implement los_anchored_ls.py (PARTS 1-3)
  - Quick test on berlin1 only first (1377 epochs, fast)
  - Check: does LOS-Anchored-LS beat Standard LS on berlin1?
  - Check: are NLOS residuals now positive-biased?

Step 3: Full 16-method evaluation on all 4 datasets (PART 4)

Step 4: Run verify_clock_contamination verification section (PART 5)

Step 5: Print success criteria report (PART 6)

Step 6: Generate paper table (PART 7)

================================================================
CONSTRAINTS
================================================================
- Do NOT retrain Module 1 (use exp_040-043)
- Do NOT delete any previous result files
- Keep all 12 existing methods in evaluate_fusion.py
- LOS-anchored methods MUST fall back gracefully when
  high_los_mask.sum() < 4 (use sigma-clip clock instead)
- All 16 methods run on all 4 datasets without crash
- Cache Module 1 inference results (already done: exp_040-043 caches)
  Reuse existing caches — do NOT rebuild from scratch
- Print timing for each method
- The geometry-aware satellite selection must GUARANTEE
  at least min_sats=5 satellites are always retained
- Never remove a satellite with p_los > 0.5 
  (only remove confident NLOS sats from geometry)
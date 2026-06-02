Goal: Module 2 complete overhaul — fix factor graph instability,
verify pseudorange geometry, integrate 2A prior, and achieve
meaningful positioning improvement over Standard LS baseline.

================================================================
PART 1: PSEUDORANGE GEOMETRY VERIFICATION (P0, do this first)
================================================================

File: fusion/debug_geometry.py

The current system has CEP50 400-1000m which may contain systematic
bias. Before any optimization work, write a geometry verification
script that checks every step of the pseudorange observation model.

Step 1 — Single epoch sanity check:
  Pick epoch index 0 from berlin1 dataset.
  For each visible satellite i:
    a) Print sv_pos_i in ECEF (km) — loaded from SP3
    b) Print gt_rx_pos in ECEF (km) — from NAV-POSLLH converted
    c) Compute geometric_range = ||sv_pos_i - gt_rx_pos|| in km
    d) Print pr_mes_i from RXM-RAWX.csv in km (divide raw meters by 1000)
    e) Compute residual = pr_mes_i - geometric_range
    f) Print elevation_i, expected_residual_sign
  
  Expected: residuals should be in range [-2, +5] km for NLOS signals
  and [-0.5, +0.5] km for LOS signals (based on Module 1 analysis
  where error STD is 0.59-0.71 km). If residuals are in hundreds of km,
  there is a unit error or clock bias not being absorbed.

Step 2 — Clock bias estimation:
  The receiver clock bias should be estimated, not assumed zero.
  Compute initial clock bias estimate:
    clk_bias_0 = median(pr_mes_i - geometric_range_i) over all sats
  After subtracting clk_bias_0, check that remaining residuals
  are in [-2, +5] km range. Print this verification.

Step 3 — Jacobian sign verification:
  The Jacobian of pseudorange w.r.t. receiver position must be correct.
  For satellite i:
    predicted_pr = ||rx - sv_i|| + clk_bias
    d(predicted_pr)/d(rx) = (rx - sv_i) / ||rx - sv_i||  [unit vector FROM sv TO rx]
  
  In the WLS H matrix: H[i, 0:3] = (rx - sv_i) / dist  (positive, pointing away from SV)
  H[i, 3] = 1.0  (clock partial)
  residual[i] = pr_mes_i - predicted_pr_i
  correction = (H^T W H)^{-1} H^T W residual  (this adds to current estimate)
  
  Verify by: starting from gt_rx_pos + [0.1, 0, 0, 0] offset (100m east),
  run one WLS iteration, check that correction moves TOWARD gt_rx_pos.
  Print before/after distance to gt. This is the definitive Jacobian test.

Step 4 — SP3 clock correction decision:
  Test both cases on epoch 0:
    Case A: pr_corrected = pr_mes (no SP3 clock correction)
    Case B: pr_corrected = pr_mes - sp3_clock_correction_i
  For each case, compute residuals after clock bias absorption.
  Print RMS residual for both cases. Use whichever gives smaller RMS.
  Hard-code this decision in utils.py USE_SP3_CLOCK_CORRECTION = True/False.

================================================================
PART 2: FIX FACTOR GRAPH L-BFGS-B INSTABILITY (P0)
================================================================

File: fusion/factor_graph_fusion.py

The current L-BFGS-B optimizer diverges from a good WLS-MoG 
initialization. Root cause: NLL objective has very sharp gradients 
near sigma boundaries and p_los extremes. Fix with 4 changes:

Fix A — Robust MoG log-likelihood:
  Replace current log-likelihood with numerically stable version:
  
  def log_likelihood_robust(residual, p_los, mu_nlos, sigma_los, sigma_nlos):
    # Clip inputs to safe range
    p_los = np.clip(p_los, 0.02, 0.98)
    sigma_los = np.clip(sigma_los, 0.1, 5.0)    # tighter than before
    sigma_nlos = np.clip(sigma_nlos, 0.1, 10.0)
    
    log_comp_los  = np.log(p_los)   - 0.5*(residual/sigma_los)**2  - np.log(sigma_los)
    log_comp_nlos = np.log(1-p_los) - 0.5*((residual-mu_nlos)/sigma_nlos)**2 - np.log(sigma_nlos)
    
    # Clip individual components before logsumexp to prevent explosion
    log_comp_los  = np.clip(log_comp_los,  -30.0, 10.0)
    log_comp_nlos = np.clip(log_comp_nlos, -30.0, 10.0)
    
    # Numerically stable logsumexp
    max_val = np.maximum(log_comp_los, log_comp_nlos)
    log_mix = max_val + np.log(np.exp(log_comp_los - max_val) 
                               + np.exp(log_comp_nlos - max_val))
    return np.clip(log_mix, -30.0, 10.0)

Fix B — Huberized NLL objective:
  Satellites with |residual| > 3*sigma_nlos are outliers that
  create misleading gradients. Downweight them:
  
  def total_neg_log_likelihood(x, observations):
    total = 0.0
    for obs in observations:
      residual = obs.pr_mes - (np.linalg.norm(x[:3] - obs.sv_pos) + x[3])
      ll = log_likelihood_robust(residual, ...)
      
      # Huber-style: cap contribution of extreme outliers
      # Satellites with very large residuals shouldn't dominate gradient
      max_ll_contribution = -0.5  # corresponds to 1-sigma residual
      ll = np.maximum(ll, max_ll_contribution * obs.sigma_nlos**2)
      total += ll
    return -total

Fix C — Multi-start optimization with best-result selection:
  The NLL surface has local minima. Use 3 starting points:
    start_0: WLS-MoG solution (current warm start)
    start_1: WLS-elevation solution  
    start_2: Standard LS solution
  
  Run L-BFGS-B from each start (max 50 iterations each).
  Accept result only if:
    a) optimization converged (success=True or iterations>10)
    b) final position is within 50km of ANY starting point
    c) final NLL < initial NLL at that starting point
  Select the result with lowest NLL among valid solutions.
  If no valid solution: fall back to WLS-MoG (not Standard LS).

Fix D — Gradient verification in solve_epoch():
  Add one-time numerical gradient check (only for first epoch):
    grad_analytic = compute_jacobian(x0, observations)
    grad_numeric  = scipy.optimize.approx_fprime(x0, neg_ll_func, 1e-5)
    rel_error = |grad_analytic - grad_numeric| / (|grad_numeric| + 1e-8)
    if rel_error.max() > 0.01:
        print("WARNING: Jacobian error > 1%, max relative error:", rel_error.max())
  This catches sign errors automatically.

================================================================
PART 3: INTEGRATE MODULE 2A PRIOR INTO FULL PIPELINE (P1)
================================================================

File: fusion/motion_geometry_predictor.py (already written, needs fixes)
File: fusion/evaluate_fusion.py (needs 2A integration)

The TCN predictor exists but was never connected. Connect it:

Step 1 — Fix TCN training data construction:
  In motion_geometry_predictor.py, build_training_sequences():
    For each dataset, load all epochs in order.
    For each epoch t (starting at t=10):
      input_seq: epochs [t-10, t-1] → extract features:
        per epoch: [rx_vel_x, rx_vel_y, rx_vel_z] (3D, finite diff of ECEF pos)
                   + per satellite slot: [elevation/90, azimuth/360, p_los_from_module1]
                   padded to MAX_SV=20 satellites → total dim = 3 + 20*3 = 63 per timestep
      target: at epoch t, for each satellite: (1 - p_los_from_module1)
              → shape (20,), masked by visibility
    
    p_los_from_module1 = run MoG inference using best_model.pth for that dataset
    Cache inference results to avoid re-running Module 1 repeatedly.
    Save to: fusion/cache/{dataset}_module1_outputs.pkl

Step 2 — TCN training loop:
  Train one TCN model per dataset (not cross-dataset).
  Loss: BCE(p_nlos_pred * visibility_mask, target * visibility_mask)
  Optimizer: Adam lr=1e-3
  Epochs: 30, batch_size=64 sequences
  Save model to: fusion/models/tcn_{dataset}.pth

Step 3 — Prior injection at evaluation time:
  In evaluate_fusion.py, add method "FactorGraph-MoG+2A":
    For each epoch t:
      If t >= 10 and tcn_model loaded:
        p_nlos_prior, confidence = tcn_model.predict(history[t-10:t])
        For satellite i where confidence[i] > 0.5:
          # Bayesian update: posterior ∝ likelihood × prior
          p_los_updated = p_los_gat[i] * (1 - p_nlos_prior[i])
          p_los_updated /= (p_los_updated + (1-p_los_gat[i]) * p_nlos_prior[i])
          observations[i].p_los = p_los_updated
      Run FactorGraph-MoG with updated p_los values.

================================================================
PART 4: COMPLETE EVALUATION TABLE (P1)
================================================================

File: fusion/evaluate_fusion.py

Generate the following 6-method comparison table for all 4 datasets.
Save results to fusion/results/positioning_results.json and print:

Methods to evaluate:
  1. Standard LS (no Module 1)
  2. WLS-elevation (no Module 1)  
  3. WLS-MoG (Module 1 weights)
  4. Hard-threshold p_los>0.5 (Module 1)
  5. FactorGraph-MoG (fixed L-BFGS-B)
  6. FactorGraph-MoG+2A (with TCN prior)

Metrics per method per dataset:
  - CEP50 (m), CEP95 (m), Mean 2D error (m), RMSE 3D (m)
  - % epochs with error < 5m, < 10m, < 20m, < 50m, < 100m
  - Mean num_satellites used (for Hard-threshold)
  - Mean convergence iterations (for FactorGraph methods)
  - % epochs where FactorGraph improved over WLS-MoG init

Also compute and print:
  - Improvement of FactorGraph-MoG over WLS-MoG: delta_CEP50 (%)
  - Improvement of FactorGraph-MoG+2A over FactorGraph-MoG: delta_CEP50 (%)
  - For each dataset: correlation between p_los_gap and positioning error

Expected targets (if geometry is correct):
  FactorGraph-MoG should beat WLS-MoG by >5% CEP50 in >=2/4 datasets.
  FactorGraph-MoG+2A should beat FactorGraph-MoG by >3% in berlin2/frankfurt1.
  If these targets are NOT met, print diagnostic info:
    - Distribution of p_los values (are they all near 0.5?)
    - Distribution of sigma_nlos/sigma_los ratios
    - Fraction of epochs where L-BFGS-B converged successfully

================================================================
CONSTRAINTS
================================================================
- Run Part 1 debug script FIRST before any optimization changes
- Do not modify Module 1 files (GAT_V2025.py, config.py)
- Reuse existing coordinate conversion from Radio_Depth_Generate.py
- All 4 datasets must complete without crash
- Print intermediate diagnostics at each major step
- Save all results to fusion/results/ directory
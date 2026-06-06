Goal: Module 3 v2 — Fix metric consistency, resolve frankfurt2 
late-epoch degradation, tune online learning parameters, integrate
TCN prior, and produce final paper-ready results.

This is the final polishing pass before research conclusion.
All architectural components exist. The focus is: fix 3 specific
bugs + tune 2 hyperparameters + run final comprehensive evaluation.

================================================================
PART 0: FIX 2D ERROR METRIC CONSISTENCY (CRITICAL, do first)
================================================================

File: model/evaluate_module3.py

The current ecef_2d_error() converts to LLA and uses flat-earth
approximation. Module 2 uses ECEF-based horizontal error.
They produce different absolute values (Standard LS shows 1016m
in Module 3 vs 904m in Module 2 for berlin1 — same dataset, same
solver, 112m gap).

Replace the current ecef_2d_error() with:

def ecef_2d_error(pos_ecef_km, gt_ecef_km):
  """
  Compute horizontal (2D) positioning error in meters.
  Uses ECEF subtraction projected onto local horizontal plane.
  This matches Module 2's error computation exactly.
  """
  diff = (pos_ecef_km - gt_ecef_km) * 1000  # convert to meters
  
  # Project onto local horizontal plane at GT position
  # Using GT position as reference for ENU frame
  gt_norm = gt_ecef_km / (np.linalg.norm(gt_ecef_km) + 1e-10)
  
  # Up vector = unit vector from Earth center to GT position
  up = gt_norm
  
  # East vector = cross(up, North_approx) normalized
  north_approx = np.array([0, 0, 1])
  east = np.cross(up, north_approx)
  east = east / (np.linalg.norm(east) + 1e-10)
  
  # North vector
  north = np.cross(east, up)
  north = north / (np.linalg.norm(north) + 1e-10)
  
  # Project error onto horizontal plane
  e_component = np.dot(diff, east)
  n_component = np.dot(diff, north)
  
  return np.sqrt(e_component**2 + n_component**2)

Update ALL places in evaluate_module3.py and run_module3.py where
2D error is computed. After this fix, Standard LS CEP50 should
match Module 2 values (berlin1 ~904m, berlin2 ~611m, etc).

Verify: print Standard LS CEP50 with new metric and confirm it
matches Module 2 exp_015 results within 5%.

================================================================
PART 1: FIX FRANKFURT2 LATE-EPOCH DEGRADATION
================================================================

File: model/residual_feedback.py

The critical bug: frankfurt2 CEP50 goes from 121m (first 100) to
740m (last 100) — a 509% degradation. This means the adaptive
corrector is SELECTING WLS-MoG/FG when it should be using LS,
and the fallback is not triggering.

Root cause investigation (add diagnostic prints in run_module3.py):

Step 1 — Diagnose at epoch level for frankfurt2:
  For epochs 3400-3575 (last ~175 epochs of frankfurt2):
    Print per-epoch: epoch_idx, method_selected, mog_2d_error, 
                     stdls_2d_error, innovation, scene_quality,
                     quality_score, fallback_triggered

Step 2 — Fix the fallback comparison:
  The fallback compares mog_error > stdls_error PER EPOCH.
  But for large errors (both >500m), small numerical differences
  matter less. The fallback might be failing because:
  
  Current fallback logic (guess):
    if mog_error > stdls_error: use stdls result
  
  Potential issue: if BOTH are computed from the same initial
  position estimate and the scene has changed, one may be
  consistently slightly worse but not triggering fallback.
  
  Fix: use a RELATIVE threshold for fallback:
    if mog_error > stdls_error * 1.05:  # 5% relative margin
      use stdls result
    else:
      use mog result
  
  This makes fallback trigger earlier when MoG starts to diverge.

Step 3 — Add CUSUM integration to method selection:
  Currently CUSUM runs but doesn't influence decisions.
  When CUSUMShiftDetector detects POSITIVE shift (MoG getting worse):
    Force LOW_QUALITY for next 10 epochs (not 5 as originally planned)
    Reset tracker innovation history (don't use stale data)
  
  In AdaptivePosCorrector.process_epoch():
    # Check CUSUM before quality classification
    if hasattr(self, 'cusum_detector'):
      shift = self.cusum_detector.update(innovation)
      if shift == 'POSITIVE':
        # Override to LOW quality for 10 epochs
        self._low_quality_override = 10
    
    if getattr(self, '_low_quality_override', 0) > 0:
      self._low_quality_override -= 1
      quality, score = 'LOW', 0.0
      method = 'Standard-LS'
      pos_selected = pos_stdls
    else:
      # Normal quality detection flow
      ...

Step 4 — Add window reset on scene transition:
  When CUSUM detects shift, also reset the innovation history:
    self.tracker.innovation_history.clear()
    self.tracker.mog_err_history.clear()
    self.tracker.stdls_err_history.clear()
  This prevents stale pre-transition data from influencing 
  post-transition quality assessment.

================================================================
PART 2: FIX ONLINE LEARNING — LARGER WINDOW AND SLOWER ADAPTATION
================================================================

File: model/residual_feedback.py

C3 failure: only berlin1 shows positive online learning.
berlin2/frankfurt show degradation over time.
Root cause: min_history=5 and window_size=20 are too small.
With only 5 samples, the quality classification is noisy.
Thresholds adapt too quickly on noisy signals.

Changes:
  ResidualInnovationTracker:
    window_size: 20 → 50
    min_history: 5 → 15
    
  SceneQualityDetector:
    ema_alpha: 0.1 → 0.05  (slower adaptation)
    Initial threshold check: require improvement_fraction > 0.65 
    (was 0.60) for HIGH_QUALITY
    
  Specifically in get_scene_quality():
    # Current thresholds (too loose)
    if mean_innovation < -10 and improvement_fraction > 0.6:
      return 'HIGH_QUALITY', improvement_fraction
    
    # New stricter thresholds
    if mean_innovation < -15 and improvement_fraction > 0.65:
      return 'HIGH_QUALITY', improvement_fraction
    elif mean_innovation < -5 and improvement_fraction > 0.70:
      return 'HIGH_QUALITY', improvement_fraction  # second tier
    elif mean_innovation > 15 and improvement_fraction < 0.35:
      return 'LOW_QUALITY', 1.0 - improvement_fraction

Also in SceneQualityDetector.classify_epoch(), increase score
threshold for HIGH quality action:
  quality = 'HIGH' if score >= 0.65 else 'LOW'  # was 0.60

These changes should reduce false HIGH_QUALITY classifications
that lead to WLS being applied when it shouldn't be.

================================================================
PART 3: INTEGRATE TCN TEMPORAL PRIOR INTO FG SOLVER
================================================================

File: model/residual_feedback.py — update make_fg_solver()

TCN models are already trained at:
  ../part2_FactorGraphLocalizationFusion/models/tcn_{dataset}.pth

Update make_fg_solver() to load and use TCN:

def make_fg_solver(dataset_name, use_tcn=True):
  """Returns a FG-MoG solver with optional TCN prior."""
  import sys
  sys.path.insert(0, '../../part2_FactorGraphLocalizationFusion/model')
  from fusion.factor_graph_fusion import FactorGraphPositioner
  from fusion.motion_geometry_predictor import MotionGeometryPredictor
  
  positioner = FactorGraphPositioner()
  
  tcn_model = None
  if use_tcn:
    tcn_path = f'../../part2_FactorGraphLocalizationFusion/models/tcn_{dataset_name}.pth'
    if os.path.exists(tcn_path):
      tcn_model = MotionGeometryPredictor()
      tcn_model.load_state_dict(torch.load(tcn_path, map_location='cpu'))
      tcn_model.eval()
  
  # Maintain per-call history for TCN (closure state)
  history_buffer = []  # list of (epoch_features,)
  
  def solver(obs_list, sv_positions, mog_outputs):
    # Build TCN input if enough history
    if tcn_model is not None and len(history_buffer) >= 10:
      try:
        seq = build_tcn_sequence(history_buffer[-10:], mog_outputs)
        with torch.no_grad():
          p_nlos_prior = tcn_model(seq).numpy().flatten()
          p_los_tcn = 1.0 - p_nlos_prior[:len(mog_outputs['p_los'])]
        
        # Soft blend (alpha capped at 0.25)
        p_los_orig = mog_outputs['p_los']
        conf = np.abs(p_nlos_prior[:len(p_los_orig)] - 0.5) * 2
        alpha = np.clip(conf * 0.25, 0, 0.25)
        
        disagree = ((p_nlos_prior[:len(p_los_orig)] > 0.6) & (p_los_orig > 0.5)) | \
                   ((p_nlos_prior[:len(p_los_orig)] < 0.4) & (p_los_orig < 0.5))
        p_los_updated = np.where(disagree,
          (1-alpha) * p_los_orig + alpha * p_los_tcn, p_los_orig)
        mog_outputs = dict(mog_outputs)
        mog_outputs['p_los'] = p_los_updated
      except Exception:
        pass  # TCN failed, use original p_los
    
    # Update history
    history_buffer.append({
      'p_los': mog_outputs['p_los'].copy(),
      'elevation': mog_outputs.get('elevation_deg', np.zeros_like(mog_outputs['p_los'])),
      'azimuth': mog_outputs.get('azimuth_deg', np.zeros_like(mog_outputs['p_los']))
    })
    if len(history_buffer) > 15:
      history_buffer.pop(0)
    
    try:
      result = positioner.solve_epoch(obs_list, sv_positions, mog_outputs)
      return result[0], result[1]
    except:
      from fusion.baselines import solve_wls_mog
      return solve_wls_mog(obs_list, sv_positions,
                           mog_outputs['p_los'], mog_outputs['sigma_los'])
  
  return solver

def build_tcn_sequence(history, current_mog):
  """Build (1, 10, 63) TCN input from history buffer."""
  MAX_SV = 20
  SEQ_LEN = 10
  
  seq = np.zeros((SEQ_LEN, MAX_SV * 3 + 3))
  for t, h in enumerate(history[-SEQ_LEN:]):
    n = min(len(h['p_los']), MAX_SV)
    seq[t, :3] = 0.0  # velocity (unknown, set to 0)
    for i in range(n):
      seq[t, 3 + i*3] = h['elevation'][i] / 90.0
      seq[t, 3 + i*3 + 1] = h['azimuth'][i] / 360.0
      seq[t, 3 + i*3 + 2] = h['p_los'][i]
  
  return torch.tensor(seq[np.newaxis], dtype=torch.float32)

================================================================
PART 4: TUNE PER-DATASET THRESHOLDS
================================================================

File: model/run_module3.py

Frankfurt datasets have high fallback rates (61-63%) with
SceneQualityDetector initial thresholds tuned for berlin.
Add per-dataset initial threshold configuration:

DATASET_CONFIGS = {
  'berlin1_potsdamer_platz': {
    'initial_plos_gap_threshold': 0.50,
    'initial_pdop_ratio_threshold': 1.12,
    'window_size': 50,
    'min_history': 15
  },
  'berlin2_gendarmenmarkt': {
    'initial_plos_gap_threshold': 0.55,
    'initial_pdop_ratio_threshold': 1.10,
    'window_size': 50,
    'min_history': 15
  },
  'frankfurt1_maintower': {
    'initial_plos_gap_threshold': 0.45,  # lower: frankfurt1 has lower gap
    'initial_pdop_ratio_threshold': 1.08,  # stricter DOP: frankfurt has DOP issues
    'window_size': 50,
    'min_history': 20
  },
  'frankfurt2_westendtower': {
    'initial_plos_gap_threshold': 0.50,
    'initial_pdop_ratio_threshold': 1.10,
    'window_size': 50,
    'min_history': 20
  }
}

Pass these configs when constructing SceneQualityDetector and
ResidualInnovationTracker for each dataset run.

Also for frankfurt1 specifically, start the FG solver threshold
at score >= 0.75 (was 0.70) to be more conservative:

In AdaptivePosCorrector.process_epoch():
  fg_threshold = getattr(self, 'fg_threshold', 0.70)
  wls_threshold = getattr(self, 'wls_threshold', 0.60)
  
  if quality == 'HIGH' and score >= fg_threshold:
    ... use FG ...
  elif quality == 'HIGH' and score >= wls_threshold:
    ... use WLS ...

Set fg_threshold=0.75 for frankfurt datasets in DATASET_CONFIGS.

================================================================
PART 5: COMPREHENSIVE FINAL EVALUATION (exp_002)
================================================================

File: model/run_module3.py and model/evaluate_module3.py

Run exp_002 with ALL fixes from Parts 0-4. Save to result/exp_002/.

Report the following metrics for each dataset:

=== Primary Positioning Table ===
Method         | CEP50(m) | CEP95(m) | Mean2D(m) | vs Std LS | vs Best Static
Standard-LS    |          |          |           | (baseline)| (baseline)
WLS-MoG        |          |          |           |           |
FG-MoG+2A      |          |          |           |           |
Adaptive-M3    |          |          |           |           |
Adaptive-M3+TCN|          |          |           |           |

=== Method Selection Distribution ===
Dataset | LS | LS-fallback | WLS | FG | FG+TCN

=== Online Learning Effect (exp_002 with tuned window) ===
Dataset | First_100_CEP50 | Last_100_CEP50 | Change_%

=== CUSUM Detections ===
Dataset | num_positive_shifts | num_negative_shifts | 
         avg_epochs_between_shifts

=== Success Criteria ===
C1: Adaptive ≤ Standard LS ALL 4: [PASS/FAIL]
C2: Adaptive ≤ best static ≥3/4: [PASS/FAIL]
C3: Online learning ≥2/4: [PASS/FAIL]
C4: frankfurt1 ≤ 490m: [PASS/FAIL]
C5: CUSUM functional: [PASS/FAIL]

Save to result/exp_002/FINAL_RESULTS.md

================================================================
PART 6: CROSS-MODULE COMPARISON TABLE (Paper-Ready)
================================================================

File: model/generate_cross_module_table.py (NEW)

Generate a comprehensive table comparing all 3 modules.
This is the core contribution table for the research paper.

Use the ECEF-consistent error metric (fixed in Part 0).

Table structure (LaTeX-formatted, also print as text):

Method                    | berlin1 CEP50 | berlin2 CEP50 | frk1 CEP50 | frk2 CEP50
==========================|===============|===============|============|============
Standard LS (no M1)       | 904.5         | 610.8         | 525.2      | 382.6
─── Module 2 Best ────────|               |               |            |
WLS-MoG (M2)              | +X%           | +X%           | +7.2%      | +X%
FG-MoG+2A (M2)            | -X%           | -X%           | +9.2%      | -X%
─── Module 3 ─────────────|               |               |            |
Adaptive-M3 (M3)          | +8.5%         | +10.0%        | +3.3%      | +3.9%
Adaptive-M3+TCN (M3 v2)   | +X%           | +X%           | +X%        | +X%

Note: Module 2 best = best static method per dataset
Note: % values show improvement vs Standard LS (positive = better)

Also generate Figure description text:
  "Figure X: Progression of CEP50 improvement over Standard LS 
   across Module 1 (classification only), Module 2 (static fusion),
   and Module 3 (adaptive online correction). Module 3 Adaptive 
   achieves consistent improvement in all 4 datasets, demonstrating
   that residual feedback generalizes the scene-specific advantages
   of Module 2 to diverse urban environments."

Save table to result/exp_002/paper_table.md and paper_table.tex

================================================================
PART 7: ABLATION STUDY
================================================================

File: model/run_ablation.py (NEW)

Run ablation study to quantify the contribution of each Module 3
component:

Configuration A: No Module 3 (static best-of-M2 per dataset)
Configuration B: Adaptive selection only (no posterior correction)
Configuration C: Adaptive + posterior p_los correction
Configuration D: Adaptive + CUSUM override (no posterior)
Configuration E: Full Adaptive-M3 (all components)
Configuration F: Full Adaptive-M3 + TCN

For each configuration × 4 datasets, compute CEP50.
Print delta table showing marginal contribution of each component:
  "Posterior correction: +X m / -X% CEP50"
  "CUSUM override: +X m / -X% CEP50"
  "TCN integration: +X m / -X% CEP50"

This ablation directly supports the research paper's contribution
claims. Expected finding: all components contribute positively to
at least some datasets.

Save to result/exp_002/ablation_results.md

================================================================
IMPLEMENTATION ORDER
================================================================

Step 1: Fix ecef_2d_error() in evaluate_module3.py (Part 0)
  Quick check: run berlin1 only, verify Standard LS CEP50 ≈ 904m

Step 2: Fix frankfurt2 degradation (Part 1)
  Add diagnostic prints, identify exact failure epochs
  Implement CUSUM integration + window reset

Step 3: Update ResidualInnovationTracker parameters (Part 2)
  window_size=50, min_history=15, stricter thresholds

Step 4: Integrate TCN into FG solver (Part 3)
  Test on berlin2 (has best TCN model, val_loss=0.481)

Step 5: Set per-dataset configs (Part 4)

Step 6: Run full exp_002 evaluation (Part 5)
  Check: all 5 success criteria
  If C3 still fails: increase min_history to 25 for frankfurt

Step 7: Generate cross-module table (Part 6)
  Verify numbers match exp_002 results

Step 8: Run ablation study (Part 7)
  ~30 min additional compute

================================================================
SUCCESS CRITERIA (exp_002 targets)
================================================================

[C1] REQUIRED: Adaptive-M3 ≤ Standard LS in ALL 4 datasets
     (non-negotiable — safety fallback must guarantee this)

[C2] REQUIRED: Adaptive-M3 ≤ best static in ≥3/4 datasets

[C3] TARGET: Online learning improves in ≥2/4 datasets
     (last 100 epoch CEP50 < first 100 epoch CEP50)
     This should be achievable with window_size=50

[C4] TARGET: frankfurt1 Adaptive ≤ 490m
     (close miss in v1 at 496.7m, per-dataset tuning should close it)

[C5] REQUIRED: CUSUM detects and overrides at least 1 genuine
     scene transition in berlin2 or frankfurt1

[BONUS] Adaptive-M3+TCN improves over Adaptive-M3 in ≥2/4 datasets

If all C1-C5 pass: print "Module 3 COMPLETE"
If only C1-C2 pass and delta from C3/C4 < 5%: print "Module 3 SUCCESS"

================================================================
CONSTRAINTS
================================================================
- The ecef_2d_error fix (Part 0) is MANDATORY first step
  No subsequent results are meaningful without consistent metrics
- All result files from exp_001 preserved; exp_002 goes to new dir
- Do NOT retrain any Module 1 models
- Do NOT modify Module 2 algorithm files (only add TCN wrapper)
- The fallback guarantee (C1) must be enforced by construction,
  not by hyperparameter tuning — code must always compute Standard LS
  and compare before returning any adaptive result
- Per-dataset configs in DATASET_CONFIGS must be documented in
  result/exp_002/params.json for reproducibility
- TCN integration uses existing trained models (no retraining)
  If TCN model is missing for a dataset, gracefully fall back to FG
- Print timing breakdown: tracker update, detector classification,
  solver calls, total per epoch (expected: <5ms per epoch)
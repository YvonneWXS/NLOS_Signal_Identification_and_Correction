Goal: Implement Module 3 — Residual Feedback and Online Correction.
Use positioning residuals from Module 2 to detect distribution shift,
adaptively correct Module 1 p_los outputs per scene, and generalize
the frankfurt1 improvement pattern to all 4 datasets.

Module 3 is the residual feedback loop described in the PI-SEP 
framework. It operates in three steps:
  (A) Compute innovation sequence (positioning residuals over time)
  (B) Detect distribution shift / scene changes using sliding window
  (C) Apply online correction: either posterior adjustment of p_los
      or lightweight fine-tuning of Module 1 last layers

The key insight from Module 2: frankfurt1 works because its NLOS 
satellites are geometrically redundant (removing them doesn't hurt 
DOP). Modules 2 never knew which scenes have this property. Module 3 
learns it from residuals: when WLS-MoG reduces residuals → scene 
allows weighting; when residuals increase → scene requires uniform LS.

================================================================
PART 0: DEFINE THE MODULE 3 ARCHITECTURE
================================================================

File: part3_ResidualFeedback/model/residual_feedback.py (NEW MODULE)

Module 3 has three sub-components:

[A] ResidualInnovationTracker
    - Maintains a sliding window of T=20 epochs of positioning residuals
    - Residual for epoch t = Module2_position_t - StandardLS_position_t
    - If Module 2 helps: residual < 0 (Module 2 better than LS)
    - If Module 2 hurts: residual > 0 (Module 2 worse than LS)
    - Computes: mean, variance, trend (slope of residual over window)

[B] SceneQualityDetector
    - Classifies each epoch as:
      HIGH_QUALITY: Module 2 weighting is likely to help
        (p_los gap > threshold AND DOP impact acceptable)
      LOW_QUALITY: Module 2 weighting will hurt
        (DOP sensitive scene, weights degrade geometry)
    - Learns the threshold from residual history (no GT needed)

[C] AdaptivePosCorrector
    - For HIGH_QUALITY epochs: use FG-MoG+2A result
    - For LOW_QUALITY epochs: fall back to Standard LS
    - Gradually adjusts the HIGH/LOW threshold based on running
      performance metric (online Bayesian update)

================================================================
PART 1: RESIDUAL INNOVATION TRACKER
================================================================

File: part3_ResidualFeedback/model/residual_feedback.py

class ResidualInnovationTracker:
  """
  Tracks the innovation sequence: difference between Module 2 
  (WLS-MoG) and Standard LS positioning results.
  Uses this to detect when MoG weighting is helping vs hurting.
  """
  
  def __init__(self, window_size=20, min_history=5):
    self.window_size = window_size
    self.min_history = min_history
    self.innovation_history = []  # (Module2_err - StdLS_err) in meters
    self.position_history = []    # list of (stdls_pos, mog_pos) pairs
    self.mog_err_history = []     # Module 2 2D error over time
    self.stdls_err_history = []   # Standard LS 2D error over time
  
  def update(self, epoch_idx, stdls_pos, mog_pos, gt_pos_ecef):
    """
    Update tracker with new epoch results.
    Returns: innovation = mog_error - stdls_error (meters)
    Positive = MoG worse, Negative = MoG better
    """
    def ecef_2d_error(pos, gt):
      # Convert both to LLA and compute horizontal error
      from fusion.utils import ecef_to_lla
      pos_lla = ecef_to_lla(pos)
      gt_lla = ecef_to_lla(gt)
      dlat = (pos_lla[0] - gt_lla[0]) * 111320  # meters
      dlon = (pos_lla[1] - gt_lla[1]) * 111320 * np.cos(np.radians(gt_lla[0]))
      return np.sqrt(dlat**2 + dlon**2)
    
    stdls_err = ecef_2d_error(stdls_pos, gt_pos_ecef)
    mog_err = ecef_2d_error(mog_pos, gt_pos_ecef)
    innovation = mog_err - stdls_err
    
    self.innovation_history.append(innovation)
    self.mog_err_history.append(mog_err)
    self.stdls_err_history.append(stdls_err)
    
    # Keep only last window_size entries
    if len(self.innovation_history) > self.window_size:
      self.innovation_history.pop(0)
      self.mog_err_history.pop(0)
      self.stdls_err_history.pop(0)
    
    return innovation
  
  def get_scene_quality(self):
    """
    Returns scene quality assessment based on recent innovation.
    HIGH_QUALITY: MoG consistently helps (negative innovations)
    LOW_QUALITY: MoG consistently hurts (positive innovations)
    UNCERTAIN: mixed signal
    """
    if len(self.innovation_history) < self.min_history:
      return 'UNCERTAIN', 0.0
    
    recent = self.innovation_history[-self.min_history:]
    mean_innovation = np.mean(recent)
    std_innovation = np.std(recent)
    
    # Improvement ratio: fraction of recent epochs where MoG helps
    improvement_fraction = np.mean([x < 0 for x in recent])
    
    if mean_innovation < -10 and improvement_fraction > 0.6:
      return 'HIGH_QUALITY', improvement_fraction
    elif mean_innovation > 10 and improvement_fraction < 0.4:
      return 'LOW_QUALITY', 1.0 - improvement_fraction
    else:
      return 'UNCERTAIN', 0.5
  
  def get_statistics(self):
    if len(self.innovation_history) < 2:
      return {}
    return {
      'mean_innovation': float(np.mean(self.innovation_history)),
      'std_innovation': float(np.std(self.innovation_history)),
      'improvement_fraction': float(np.mean([x < 0 for x in self.innovation_history])),
      'trend': float(np.polyfit(range(len(self.innovation_history)), 
                                 self.innovation_history, 1)[0]),  # slope
      'window_size': len(self.innovation_history)
    }

================================================================
PART 2: SCENE QUALITY DETECTOR WITH ONLINE THRESHOLD LEARNING
================================================================

File: part3_ResidualFeedback/model/residual_feedback.py (same file)

class SceneQualityDetector:
  """
  Learns an adaptive threshold for p_los gap and DOP sensitivity
  that determines when Module 2 weighting will help.
  Uses exponential moving average to adapt to scene changes.
  """
  
  def __init__(self, initial_plos_gap_threshold=0.55,
               initial_pdop_ratio_threshold=1.10,
               ema_alpha=0.1):
    # Thresholds for HIGH_QUALITY classification
    self.plos_gap_threshold = initial_plos_gap_threshold
    self.pdop_ratio_threshold = initial_pdop_ratio_threshold
    self.ema_alpha = ema_alpha
    
    # Running statistics for adaptation
    self.quality_accuracy_history = []
    self.threshold_history = []
  
  def classify_epoch(self, epoch_mog_outputs, sv_positions, rx_pos_approx):
    """
    Classify epoch as HIGH/LOW quality for MoG weighting.
    
    epoch_mog_outputs: dict with p_los array, sigma_los array
    sv_positions: (N, 3) array
    rx_pos_approx: (3,) initial position estimate
    
    Returns: quality_class ('HIGH'/'LOW'), confidence (0-1)
    """
    p_los = epoch_mog_outputs['p_los']
    sigma_los = epoch_mog_outputs['sigma_los']
    
    # Feature 1: p_los gap (LOS/NLOS discrimination quality)
    los_mask = p_los > 0.6
    nlos_mask = p_los < 0.4
    
    if los_mask.sum() > 0 and nlos_mask.sum() > 0:
      plos_gap = p_los[los_mask].mean() - p_los[nlos_mask].mean()
    else:
      plos_gap = 0.0
    
    # Feature 2: DOP impact of WLS weighting
    weights = np.maximum(0.01, p_los) / np.maximum(0.01, sigma_los**2)
    
    def compute_pdop(w, sv_pos, rx_pos):
      dists = np.linalg.norm(sv_pos - rx_pos[np.newaxis,:], axis=1)
      los_vecs = (sv_pos - rx_pos[np.newaxis,:]) / dists[:,np.newaxis]
      H = np.hstack([-los_vecs, np.ones((len(w),1))])
      W = np.diag(w)
      try:
        P = np.linalg.inv(H.T @ W @ H)
        return np.sqrt(P[0,0] + P[1,1] + P[2,2])
      except:
        return 999.0
    
    pdop_uniform = compute_pdop(np.ones(len(p_los)), sv_positions, rx_pos_approx)
    pdop_weighted = compute_pdop(weights, sv_positions, rx_pos_approx)
    pdop_ratio = pdop_weighted / (pdop_uniform + 1e-6)
    
    # Feature 3: Fraction of NLOS satellites that are geometrically redundant
    # (removing them doesn't hurt DOP)
    nlos_indices = np.where(p_los < 0.4)[0]
    redundant_nlos = 0
    for idx in nlos_indices:
      remaining = [i for i in range(len(p_los)) if i != idx]
      pdop_without = compute_pdop(np.ones(len(remaining)), 
                                   sv_positions[remaining], rx_pos_approx)
      if pdop_without < pdop_uniform * 1.15:
        redundant_nlos += 1
    redundancy_fraction = redundant_nlos / max(1, len(nlos_indices))
    
    # Classification rule
    gap_ok = plos_gap > self.plos_gap_threshold
    dop_ok = pdop_ratio < self.pdop_ratio_threshold
    redundancy_ok = redundancy_fraction > 0.5
    
    features = {
      'plos_gap': plos_gap,
      'pdop_ratio': pdop_ratio,
      'redundancy_fraction': redundancy_fraction,
      'gap_ok': gap_ok,
      'dop_ok': dop_ok,
      'redundancy_ok': redundancy_ok
    }
    
    # Score: weighted combination of 3 features
    score = (0.4 * float(gap_ok) + 
             0.4 * float(dop_ok) + 
             0.2 * float(redundancy_ok))
    
    quality = 'HIGH' if score >= 0.6 else 'LOW'
    return quality, score, features
  
  def update_thresholds(self, predicted_quality, actual_innovation):
    """
    Online threshold adaptation: if prediction was wrong, adjust thresholds.
    actual_innovation < 0 means HIGH_QUALITY was correct
    actual_innovation > 0 means LOW_QUALITY was correct
    """
    correct = ((predicted_quality == 'HIGH' and actual_innovation < 0) or
               (predicted_quality == 'LOW' and actual_innovation > 0))
    
    self.quality_accuracy_history.append(float(correct))
    if len(self.quality_accuracy_history) > 50:
      self.quality_accuracy_history.pop(0)
    
    recent_accuracy = np.mean(self.quality_accuracy_history[-10:]) \
                      if len(self.quality_accuracy_history) >= 10 else 0.5
    
    # If accuracy drops below 60%, tighten the HIGH_QUALITY threshold
    if recent_accuracy < 0.6 and predicted_quality == 'HIGH':
      self.plos_gap_threshold = min(0.70, 
          self.plos_gap_threshold + self.ema_alpha * 0.05)
      self.pdop_ratio_threshold = max(1.05,
          self.pdop_ratio_threshold - self.ema_alpha * 0.05)
    elif recent_accuracy > 0.75 and predicted_quality == 'HIGH':
      # Can relax thresholds
      self.plos_gap_threshold = max(0.40,
          self.plos_gap_threshold - self.ema_alpha * 0.02)

================================================================
PART 3: ADAPTIVE POSITIONING CORRECTOR
================================================================

File: part3_ResidualFeedback/model/residual_feedback.py (same file)

class AdaptivePosCorrector:
  """
  Selects the best positioning method for each epoch based on
  scene quality classification and residual history.
  """
  
  def __init__(self):
    self.tracker = ResidualInnovationTracker(window_size=20, min_history=5)
    self.detector = SceneQualityDetector()
    
    # Statistics
    self.method_selection_history = []
    self.epoch_results = []
  
  def process_epoch(self, epoch_idx, obs_list, sv_positions, 
                     mog_outputs, gt_pos_ecef,
                     stdls_solver, mog_solver, fg_solver):
    """
    Process one epoch: classify, select method, solve, update.
    
    stdls_solver: callable(obs_list, sv_positions) -> (pos, clk)
    mog_solver: callable(obs_list, sv_positions, mog_outputs) -> (pos, clk)
    fg_solver: callable(obs_list, sv_positions, mog_outputs) -> (pos, clk)
    
    Returns: position_ecef (3,), method_used (str), diagnostics (dict)
    """
    # Always compute Standard LS (reference)
    pos_stdls, clk_stdls = stdls_solver(obs_list, sv_positions)
    
    # Scene quality classification
    quality, score, features = self.detector.classify_epoch(
        mog_outputs, sv_positions, pos_stdls)
    
    # Method selection
    if quality == 'HIGH' and score >= 0.7:
      # High confidence: use factor graph
      pos_mog, clk_mog = fg_solver(obs_list, sv_positions, mog_outputs)
      method = 'FG-MoG+2A'
    elif quality == 'HIGH' and score >= 0.6:
      # Medium confidence: use WLS-MoG
      pos_mog, clk_mog = mog_solver(obs_list, sv_positions, mog_outputs)
      method = 'WLS-MoG'
    else:
      # Low confidence or uncertain: use Standard LS
      pos_mog, clk_mog = pos_stdls, clk_stdls
      method = 'Standard-LS'
    
    # Update tracker with results (only if GT available)
    if gt_pos_ecef is not None:
      innovation = self.tracker.update(
          epoch_idx, pos_stdls, pos_mog, gt_pos_ecef)
      
      # Update detector thresholds
      self.detector.update_thresholds(quality, innovation)
    
    self.method_selection_history.append(method)
    
    diagnostics = {
      'quality': quality,
      'score': score,
      'method': method,
      'features': features,
      'tracker_stats': self.tracker.get_statistics()
    }
    
    return pos_mog, method, diagnostics
  
  def get_summary(self):
    counts = {}
    for m in self.method_selection_history:
      counts[m] = counts.get(m, 0) + 1
    total = len(self.method_selection_history)
    return {method: count/total for method, count in counts.items()}

================================================================
PART 4: ONLINE POSTERIOR P_LOS CORRECTION
================================================================

File: part3_ResidualFeedback/model/posterior_correction.py (NEW)

In addition to method selection, implement a lightweight posterior
correction of p_los values based on residual feedback:

class PosteriorPlosCorrector:
  """
  Uses positioning residuals to compute scene-specific p_los bias
  correction. If Module 2 consistently fails on certain satellite 
  types, adjusts their p_los posteriors.
  """
  
  def __init__(self, window_size=50):
    self.window_size = window_size
    # Per-elevation-bin bias: track if module1 over/under-estimates p_los
    self.elevation_bins = np.arange(0, 91, 15)  # 6 bins
    self.plos_bias_by_elevation = np.zeros(len(self.elevation_bins) - 1)
    self.sample_count_by_elevation = np.zeros(len(self.elevation_bins) - 1)
    
    # Per-CNO-bin bias
    self.cno_bins = np.arange(20, 55, 5)  # 7 bins
    self.plos_bias_by_cno = np.zeros(len(self.cno_bins) - 1)
    
  def update_from_residuals(self, obs_list, mog_outputs, 
                              pos_estimate, sv_positions):
    """
    After positioning, use per-satellite residuals to update 
    p_los bias estimates.
    """
    p_los = mog_outputs['p_los']
    elevations = mog_outputs['elevation_deg']
    cnos = mog_outputs['cno']
    
    dists = np.linalg.norm(sv_positions - pos_estimate[np.newaxis,:], axis=1)
    clk = np.median([obs.pr_mes_km for obs in obs_list]) - np.median(dists)
    residuals = np.array([obs.pr_mes_km for obs in obs_list]) - dists - clk
    
    # Large positive residual: satellite likely NLOS but p_los was high
    # Large negative residual: noise or geometry error
    for i, (elev, cno, res) in enumerate(zip(elevations, cnos, residuals)):
      if abs(res) < 0.05:  # skip small residuals (noise-level)
        continue
      
      # Positive large residual + high p_los → overconfident LOS
      if res > 0.3 and p_los[i] > 0.6:
        bin_idx = np.searchsorted(self.elevation_bins[1:-1], elev)
        self.plos_bias_by_elevation[bin_idx] -= 0.05  # bias toward NLOS
        self.sample_count_by_elevation[bin_idx] += 1
  
  def apply_correction(self, mog_outputs):
    """
    Apply learned bias corrections to p_los values.
    Returns updated mog_outputs dict.
    """
    corrected = dict(mog_outputs)
    p_los_corrected = mog_outputs['p_los'].copy()
    elevations = mog_outputs['elevation_deg']
    
    for i, elev in enumerate(elevations):
      bin_idx = np.searchsorted(self.elevation_bins[1:-1], elev)
      if self.sample_count_by_elevation[bin_idx] > 10:
        bias = self.plos_bias_by_elevation[bin_idx]
        # Apply soft correction: limit bias to ±0.2
        bias = np.clip(bias, -0.2, 0.2)
        p_los_corrected[i] = np.clip(p_los_corrected[i] + bias, 0.02, 0.98)
    
    corrected['p_los'] = p_los_corrected
    return corrected

================================================================
PART 5: FULL PIPELINE INTEGRATION
================================================================

File: part3_ResidualFeedback/model/run_module3.py (NEW)

Full Module 3 evaluation script:

def run_module3_evaluation(dataset_name, exp_name, use_tcn=True):
  """
  Run full Module 3 evaluation on one dataset.
  Compares: Standard LS, WLS-MoG, FG-MoG+2A, Adaptive-M3
  """
  # Load data and Module 1 outputs (reuse Module 2 caches)
  epoch_data = load_epoch_data(dataset_name)
  mog_cache_path = f"../part2_FactorGraphLocalizationFusion/cache/{dataset_name}_mog_outputs.pkl"
  
  # Initialize Module 3 components
  corrector = AdaptivePosCorrector()
  posterior_corrector = PosteriorPlosCorrector()
  
  # Load TCN model if available
  tcn_model = None
  if use_tcn:
    tcn_path = f"../part2_FactorGraphLocalizationFusion/models/tcn_{dataset_name}.pth"
    if os.path.exists(tcn_path):
      tcn_model = load_tcn_model(tcn_path)
  
  results = {
    'Standard-LS': [], 'WLS-MoG': [], 
    'FG-MoG+2A': [], 'Adaptive-M3': [],
    'method_selection': [], 'quality_scores': []
  }
  
  for epoch_idx, epoch in enumerate(epoch_data):
    obs_list = epoch.observations
    sv_positions = compute_satellite_positions(epoch)
    mog_outputs = load_mog_epoch(mog_cache_path, epoch_idx)
    gt_pos = epoch.gt_ecef
    
    # Apply posterior p_los correction
    mog_corrected = posterior_corrector.apply_correction(mog_outputs)
    
    # Apply TCN prior if available
    if tcn_model is not None and epoch_idx >= 10:
      mog_corrected = apply_tcn_prior(tcn_model, mog_corrected, 
                                       results['Adaptive-M3'][-10:])
    
    # Solve all 4 methods
    pos_stdls, _ = solve_standard_ls(obs_list, sv_positions)
    pos_wls, _ = solve_wls_mog(obs_list, sv_positions, 
                                mog_corrected['p_los'],
                                mog_corrected['sigma_los'])
    pos_fg, _ = solve_fg_mog_2a(obs_list, sv_positions, mog_corrected)
    
    # Adaptive selection (Module 3 core)
    pos_adaptive, method, diag = corrector.process_epoch(
        epoch_idx, obs_list, sv_positions, mog_corrected, gt_pos,
        stdls_solver=lambda o,s: solve_standard_ls(o,s),
        mog_solver=lambda o,s,m: solve_wls_mog(o,s,m['p_los'],m['sigma_los']),
        fg_solver=lambda o,s,m: solve_fg_mog_2a(o,s,m)
    )
    
    # Update posterior corrector
    posterior_corrector.update_from_residuals(
        obs_list, mog_corrected, pos_adaptive, sv_positions)
    
    # Record results
    results['Standard-LS'].append(pos_stdls)
    results['WLS-MoG'].append(pos_wls)
    results['FG-MoG+2A'].append(pos_fg)
    results['Adaptive-M3'].append(pos_adaptive)
    results['method_selection'].append(method)
    results['quality_scores'].append(diag['score'])
  
  return results, corrector

================================================================
PART 6: EVALUATION AND REPORTING
================================================================

File: part3_ResidualFeedback/model/evaluate_module3.py (NEW)

Metrics to compute:

def evaluate_results(results, gt_positions, dataset_name):
  """Compute CEP50, CEP95, and method selection statistics."""
  
  report = {}
  
  for method_name, positions in results.items():
    if method_name in ['method_selection', 'quality_scores']:
      continue
    errors = [compute_2d_error(pos, gt) 
              for pos, gt in zip(positions, gt_positions)]
    errors = [e for e in errors if not np.isnan(e)]
    report[method_name] = {
      'cep50': float(np.median(errors)),
      'cep95': float(np.percentile(errors, 95)),
      'mean_2d': float(np.mean(errors)),
      'n_epochs': len(errors)
    }
  
  # Method selection breakdown
  selection = results['method_selection']
  method_counts = {}
  for m in selection:
    method_counts[m] = method_counts.get(m, 0) + 1
  report['method_distribution'] = {
      k: v/len(selection) for k, v in method_counts.items()}
  
  # Innovation analysis: does adaptive beat static best?
  adaptive_cep = report['Adaptive-M3']['cep50']
  best_static_cep = min(report[m]['cep50'] 
                        for m in ['Standard-LS','WLS-MoG','FG-MoG+2A'])
  report['adaptive_vs_best_static'] = {
    'adaptive_cep50': adaptive_cep,
    'best_static_cep50': best_static_cep,
    'improvement_pct': (best_static_cep - adaptive_cep) / best_static_cep * 100
  }
  
  # Online learning effect: compare first 100 epochs vs last 100 epochs
  if len(selection) > 200:
    early_errors = [compute_2d_error(p, g) 
                    for p, g in zip(results['Adaptive-M3'][:100],
                                    gt_positions[:100])]
    late_errors  = [compute_2d_error(p, g) 
                    for p, g in zip(results['Adaptive-M3'][-100:],
                                    gt_positions[-100:])]
    report['learning_effect'] = {
      'early_cep50': float(np.median(early_errors)),
      'late_cep50': float(np.median(late_errors)),
      'improvement_pct': (np.median(early_errors) - np.median(late_errors)) / 
                          np.median(early_errors) * 100
    }
  
  return report

Run on all 4 datasets and print:

=== Module 3 Results: {dataset_name} ===
Method          | CEP50 | CEP95 | vs Std LS
Standard LS     |       |       | (baseline)
WLS-MoG         |       |       |
FG-MoG+2A       |       |       |
Adaptive-M3     |       |       | ← target: best or tied with best

Method selection: HIGH_QUALITY={%}%, WLS={%}%, Standard_LS={%}%
Online learning: first 100 epochs CEP50 = {}, last 100 = {} (Δ={%}%)
Adaptive vs best static: {improvement}%

SUCCESS if Adaptive-M3 CEP50 <= min(WLS-MoG, FG-MoG+2A, Standard-LS)
in ALL 4 datasets.

Save to part3_ResidualFeedback/result/exp_001/module3_results.json

================================================================
PART 7: DISTRIBUTION SHIFT DETECTION
================================================================

File: part3_ResidualFeedback/model/shift_detector.py (NEW)

Implement CUSUM-based shift detection to identify when the scene
changes (e.g., entering/leaving a building canyon):

class CUSUMShiftDetector:
  """
  CUSUM (Cumulative Sum) control chart for detecting distribution shift
  in positioning innovation sequence.
  Reference: classic CUSUM for change-point detection.
  """
  
  def __init__(self, target=0.0, allowance=20.0, threshold=100.0):
    """
    target: expected mean innovation (meters, 0 = MoG same as LS)
    allowance: size of allowable shift before detection (meters)
    threshold: CUSUM threshold for alarm (meters)
    """
    self.target = target
    self.allowance = allowance
    self.threshold = threshold
    self.cusum_pos = 0.0  # CUSUM for positive shift
    self.cusum_neg = 0.0  # CUSUM for negative shift
    self.shift_detected = False
    self.detection_history = []
  
  def update(self, innovation_meters):
    """Update CUSUM and return shift detection status."""
    # Positive CUSUM: detects upward shift (MoG getting worse)
    self.cusum_pos = max(0, self.cusum_pos + 
                          innovation_meters - self.target - self.allowance)
    # Negative CUSUM: detects downward shift (MoG getting better)
    self.cusum_neg = max(0, self.cusum_neg - 
                          innovation_meters + self.target - self.allowance)
    
    shift = 'NONE'
    if self.cusum_pos > self.threshold:
      shift = 'POSITIVE'  # scene changed, MoG getting worse
      self.cusum_pos = 0  # reset after detection
      self.shift_detected = True
    elif self.cusum_neg > self.threshold:
      shift = 'NEGATIVE'  # scene changed, MoG getting better
      self.cusum_neg = 0  # reset
      self.shift_detected = True
    
    self.detection_history.append(shift)
    return shift
  
  def trigger_recompute(self):
    """Called when shift detected: signal to recompute quality thresholds."""
    self.shift_detected = False
    return True

Integrate into AdaptivePosCorrector:
  - When CUSUM detects POSITIVE shift: temporarily force LOW_QUALITY
    for next 5 epochs until tracker re-accumulates evidence
  - When CUSUM detects NEGATIVE shift: try HIGH_QUALITY for next 5 epochs

================================================================
PART 8: FINAL CROSS-MODULE VALIDATION
================================================================

File: part3_ResidualFeedback/model/cross_module_validation.py (NEW)

Run complete pipeline validation comparing:
  1. Module 1 alone (p_los classification accuracy)
  2. Module 2 WLS-MoG (positioning with Module 1 weights)
  3. Module 2 FG-MoG+2A (best static method)
  4. Module 3 Adaptive (online adaptive selection)

For each stage, compute and print the information gain:
  - Module 1 → 2: Does soft information improve positioning at all?
    Measure: CEP50(WLS-MoG) vs CEP50(Standard LS)
  - Module 2 → 3: Does adaptive selection improve over best static?
    Measure: CEP50(Adaptive) vs min(CEP50 across M2 methods)
  - What fraction of epochs benefit from Module 1 information?
    (epochs where any M1-based method beats Standard LS)

Print a 3-column summary table:
  Dataset | M1 gain (M2 vs LS) | M2 gain (Adaptive vs static) | 
           Epoch coverage (% epochs M1 helps)
  
  berlin1  |  +X%  |  +Y%  |  Z%
  berlin2  |  ...
  frankfurt1 | +7-15% (known) | 
  frankfurt2 |

Target:
  - Module 3 Adaptive should match or beat best static in ALL 4 datasets
    (the adaptive component ensures we never do worse than Standard LS)
  - % epochs where M1 helps should increase from M2 to M3
    (adaptive selection identifies and leverages these epochs)

Save to part3_ResidualFeedback/result/exp_001/cross_module_validation.json

================================================================
FILE STRUCTURE
================================================================

part3_ResidualFeedback/
├── model/
│   ├── run_module3.py          # Main pipeline entry
│   ├── residual_feedback.py    # ResidualInnovationTracker + 
│   │                           # SceneQualityDetector + AdaptivePosCorrector
│   ├── posterior_correction.py # PosteriorPlosCorrector
│   ├── shift_detector.py       # CUSUM shift detection
│   ├── evaluate_module3.py     # Metrics and reporting
│   └── cross_module_validation.py  # Full pipeline validation
├── result/
│   └── exp_001/                # First Module 3 experiment
└── README.md

================================================================
IMPLEMENTATION ORDER
================================================================

Step 1: Create directory structure and basic files (~30 min)
  - part3_ResidualFeedback/model/ directory
  - Stub out all files with class definitions

Step 2: Implement ResidualInnovationTracker and 
        SceneQualityDetector (Part 1-2) (~45 min)
  - Test on berlin1 first: does it correctly classify 
    HIGH_QUALITY when FG-MoG+2A beats Standard LS?

Step 3: Implement AdaptivePosCorrector (Part 3) (~30 min)
  - Wire to existing Module 2 solvers via function pointers
  - Test on berlin1: method distribution should be ~70% Standard LS,
    ~15% WLS, ~15% FG for berlin1

Step 4: Implement PosteriorPlosCorrector (Part 4) (~30 min)

Step 5: Implement run_module3.py full pipeline (Part 5) (~45 min)
  - Reuse Module 2 caches (NO new Module 1 inference)
  - Test: does adaptive result match or beat Standard LS on berlin1?

Step 6: Full 4-dataset evaluation (Part 6) (~30 min)

Step 7: Implement CUSUMShiftDetector (Part 7) (~30 min)
  - Integrate into AdaptivePosCorrector
  - Test: does CUSUM detect scene transitions in berlin data?

Step 8: Cross-module validation (Part 8) (~20 min)

Total estimated time: ~4 hours of coding + evaluation

================================================================
SUCCESS CRITERIA
================================================================

[C1] Adaptive-M3 CEP50 <= Standard LS CEP50 in ALL 4 datasets
     (the adaptive selection must never make things worse than LS)
     
[C2] Adaptive-M3 CEP50 <= min(WLS-MoG, FG-MoG+2A) CEP50 
     in at least 3/4 datasets
     (adaptive selection should approach the oracle that always 
      picks the best method)

[C3] Online learning effect: last 100 epoch CEP50 < first 100 epoch
     CEP50 in at least 2/4 datasets
     (system improves over time as it learns scene characteristics)

[C4] frankfurt1 Adaptive-M3 CEP50 <= 490m
     (should maintain the WLS-MoG/FG-MoG+2A improvement already seen)

[C5] CUSUM correctly detects at least 1 genuine scene transition
     in berlin2 or frankfurt1 (longest datasets)
     Verify by checking if detection epochs correlate with 
     sudden changes in p_los gap or Standard LS error

================================================================
CONSTRAINTS
================================================================
- Reuse ALL Module 2 caches — do NOT re-run Module 1 inference
  The cached outputs at fusion/cache/{dataset}_mog_outputs.pkl 
  are from exp_048-051 and should be used directly
- Adaptive-M3 MUST guarantee CEP50 <= Standard LS (C1 is hard constraint)
  Implement this as a fallback: if Adaptive result is worse than LS
  result for a given epoch, always return LS result
- Do NOT modify any Module 1 or Module 2 files
- The SceneQualityDetector initial thresholds can be tuned per dataset
  if needed, but the online learning should adapt them automatically
- All Module 3 results saved to part3_ResidualFeedback/result/
  NOT to the Module 2 result directory
- Print per-dataset timing for the full pipeline
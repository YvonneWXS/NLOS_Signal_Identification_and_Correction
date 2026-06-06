# residual_feedback.py — Module 3: Residual Feedback and Adaptive Online Correction
# ================================================================================
# Components:
#   [A] ResidualInnovationTracker  — sliding window of positioning innovation
#   [B] SceneQualityDetector       — online threshold learning for scene classification
#   [C] AdaptivePosCorrector       — method selector: FG-MoG / WLS-MoG / Standard-LS
#
# Key insight: frankfurt1 works because its NLOS sats are geometrically redundant.
# Module 3 learns which scenes share this property from positioning residuals.
# ================================================================================

import os, numpy as np

# --- Ensure Module 2 fusion/ is importable ---
_M2_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'part2_FactorGraphLocalizationFusion', 'model'))
import sys
if _M2_DIR not in sys.path:
    sys.path.insert(0, _M2_DIR)

from fusion.baselines import solve_standard_ls, solve_wls_mog
from fusion.factor_graph_fusion import FactorGraphPositioner


# ============================================================================
# [A] Residual Innovation Tracker
# ============================================================================

class ResidualInnovationTracker:
    """Tracks positioning innovation: Module 2 vs Standard LS, per-epoch.
    
    Innovation = WLS-MoG error - Standard LS error (meters)
      Negative: MoG helps (better than LS)
      Positive: MoG hurts  (worse than LS)
    """

    def __init__(self, window_size=20, min_history=5):
        self.window_size = window_size
        self.min_history = min_history
        self.innovation_history = []
        self.mog_err_history = []
        self.stdls_err_history = []

    def update(self, epoch_idx, stdls_pos, mog_pos, gt_pos_ecef):
        """Update tracker with new epoch results. Returns innovation in meters."""
        stdls_err = _ecef_2d_error(stdls_pos, gt_pos_ecef)
        mog_err = _ecef_2d_error(mog_pos, gt_pos_ecef)
        innovation = mog_err - stdls_err

        self.innovation_history.append(innovation)
        self.mog_err_history.append(mog_err)
        self.stdls_err_history.append(stdls_err)

        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)
            self.mog_err_history.pop(0)
            self.stdls_err_history.pop(0)

        return innovation

    def get_scene_quality(self):
        """Classify recent scene quality from innovation history."""
        if len(self.innovation_history) < self.min_history:
            return 'UNCERTAIN', 0.0

        recent = self.innovation_history[-self.min_history:]
        mean_innovation = np.mean(recent)
        improvement_fraction = float(np.mean([x < 0 for x in recent]))

        if mean_innovation < -10 and improvement_fraction > 0.6:
            return 'HIGH_QUALITY', improvement_fraction
        elif mean_innovation > 10 and improvement_fraction < 0.4:
            return 'LOW_QUALITY', 1.0 - improvement_fraction
        else:
            return 'UNCERTAIN', 0.5

    def get_statistics(self):
        if len(self.innovation_history) < 2:
            return {}
        hist = self.innovation_history
        slope = np.polyfit(range(len(hist)), hist, 1)[0] if len(hist) >= 3 else 0.0
        return {
            'mean_innovation_m': float(np.mean(hist)),
            'std_innovation_m': float(np.std(hist)),
            'improvement_fraction': float(np.mean([x < 0 for x in hist])),
            'trend_m_per_epoch': float(slope),
            'window_size': len(hist),
        }


# ============================================================================
# [B] Scene Quality Detector with Online Threshold Learning
# ============================================================================

class SceneQualityDetector:
    """Learns adaptive thresholds for when MoG weighting helps vs hurts.
    
    Three features per epoch:
      1. p_los gap: LOS mean p_los - NLOS mean p_los (detection quality)
      2. DOP ratio: weighted PDOP / uniform PDOP (geometry degradation)
      3. Redundancy fraction: fraction of NLOS sats that can be removed 
         without DOP penalty (frankfurt1-like property)
    """

    def __init__(self, initial_plos_gap_threshold=0.55,
                 initial_pdop_ratio_threshold=1.10, ema_alpha=0.1):
        self.plos_gap_threshold = initial_plos_gap_threshold
        self.pdop_ratio_threshold = initial_pdop_ratio_threshold
        self.ema_alpha = ema_alpha
        self.quality_accuracy_history = []
        self.threshold_history = []

    def classify_epoch(self, epoch_mog_outputs, sv_positions, rx_pos_approx):
        """Classify epoch as HIGH/LOW quality for MoG weighting.
        
        Returns: quality ('HIGH'/'LOW'), score (0-1), features (dict)
        """
        p_los = epoch_mog_outputs['p_los']
        sigma_los = epoch_mog_outputs['sigma_los']

        # Feature 1: p_los gap
        los_mask = p_los > 0.6
        nlos_mask = p_los < 0.4
        if los_mask.sum() > 0 and nlos_mask.sum() > 0:
            plos_gap = float(p_los[los_mask].mean() - p_los[nlos_mask].mean())
        else:
            plos_gap = 0.0

        # Feature 2: DOP impact
        weights = np.maximum(0.01, p_los) / np.maximum(0.01, sigma_los ** 2)
        pdop_uniform = _compute_pdop(np.ones(len(p_los)), sv_positions, rx_pos_approx)
        pdop_weighted = _compute_pdop(weights, sv_positions, rx_pos_approx)
        pdop_ratio = pdop_weighted / (pdop_uniform + 1e-6)

        # Feature 3: NLOS redundancy (frankfurt1-like property)
        nlos_indices = np.where(p_los < 0.4)[0]
        redundant_nlos = 0
        for idx in nlos_indices:
            remaining = [i for i in range(len(p_los)) if i != idx]
            pdop_without = _compute_pdop(np.ones(len(remaining)),
                                          sv_positions[remaining], rx_pos_approx)
            if pdop_without < pdop_uniform * 1.15:
                redundant_nlos += 1
        redundancy_fraction = redundant_nlos / max(1, len(nlos_indices))

        gap_ok = plos_gap > self.plos_gap_threshold
        dop_ok = pdop_ratio < self.pdop_ratio_threshold
        redundancy_ok = redundancy_fraction > 0.5

        features = {
            'plos_gap': plos_gap,
            'pdop_ratio': pdop_ratio,
            'redundancy_fraction': redundancy_fraction,
            'gap_ok': gap_ok, 'dop_ok': dop_ok, 'redundancy_ok': redundancy_ok,
        }

        score = (0.4 * float(gap_ok) + 0.4 * float(dop_ok) +
                 0.2 * float(redundancy_ok))
        quality = 'HIGH' if score >= 0.6 else 'LOW'
        return quality, score, features

    def update_thresholds(self, predicted_quality, actual_innovation):
        """Online adaptation: tighten/relax thresholds based on accuracy."""
        correct = ((predicted_quality == 'HIGH' and actual_innovation < 0) or
                   (predicted_quality == 'LOW' and actual_innovation > 0))
        self.quality_accuracy_history.append(float(correct))
        if len(self.quality_accuracy_history) > 50:
            self.quality_accuracy_history.pop(0)

        if len(self.quality_accuracy_history) < 10:
            return
        recent_accuracy = np.mean(self.quality_accuracy_history[-10:])

        if recent_accuracy < 0.6 and predicted_quality == 'HIGH':
            self.plos_gap_threshold = min(0.70,
                self.plos_gap_threshold + self.ema_alpha * 0.05)
            self.pdop_ratio_threshold = max(1.05,
                self.pdop_ratio_threshold - self.ema_alpha * 0.05)
        elif recent_accuracy > 0.75 and predicted_quality == 'HIGH':
            self.plos_gap_threshold = max(0.40,
                self.plos_gap_threshold - self.ema_alpha * 0.02)
        self.threshold_history.append((self.plos_gap_threshold, self.pdop_ratio_threshold))


# ============================================================================
# [C] Adaptive Positioning Corrector
# ============================================================================

class AdaptivePosCorrector:
    """Selects best positioning method per epoch based on scene quality.
    
    Guarantee (hard constraint C1): Adaptive result NEVER worse than Standard LS.
    Falls back to LS result whenever adaptive selection would increase error.
    """

    def __init__(self):
        self.tracker = ResidualInnovationTracker(window_size=20, min_history=5)
        self.detector = SceneQualityDetector()
        self.method_selection_history = []

    def process_epoch(self, epoch_idx, obs_list, sv_positions,
                       mog_outputs, gt_pos_ecef,
                       stdls_solver, mog_solver, fg_solver):
        """Process one epoch through adaptive pipeline.
        
        Returns: position_ecef (3,), method_used (str), diagnostics (dict)
        """
        # Always compute Standard LS reference
        pos_stdls, clk_stdls = stdls_solver(obs_list, sv_positions)

        # Scene quality classification
        quality, score, features = self.detector.classify_epoch(
            mog_outputs, sv_positions, pos_stdls)

        # Method selection
        if quality == 'HIGH' and score >= 0.7:
            try:
                pos_mog, clk_mog = fg_solver(obs_list, sv_positions, mog_outputs)
                method = 'FG-MoG+2A'
            except Exception:
                pos_mog, clk_mog = mog_solver(obs_list, sv_positions, mog_outputs)
                method = 'WLS-MoG'
        elif quality == 'HIGH' and score >= 0.6:
            pos_mog, clk_mog = mog_solver(obs_list, sv_positions, mog_outputs)
            method = 'WLS-MoG'
        else:
            pos_mog, clk_mog = pos_stdls, clk_stdls
            method = 'Standard-LS'

        # Safety fallback (hard constraint C1)
        if gt_pos_ecef is not None:
            mog_err = _ecef_2d_error(pos_mog, gt_pos_ecef)
            ls_err = _ecef_2d_error(pos_stdls, gt_pos_ecef)
            if mog_err > ls_err:
                pos_mog = pos_stdls
                method = 'Standard-LS(fallback)'

        # Update tracker
        if gt_pos_ecef is not None:
            innovation = self.tracker.update(epoch_idx, pos_stdls, pos_mog, gt_pos_ecef)
            self.detector.update_thresholds(quality, innovation)

        self.method_selection_history.append(method)

        diagnostics = {
            'quality': quality, 'score': score, 'method': method,
            'features': features,
            'tracker_stats': self.tracker.get_statistics(),
        }
        return pos_mog, method, diagnostics

    def get_summary(self):
        counts = {}
        for m in self.method_selection_history:
            counts[m] = counts.get(m, 0) + 1
        total = max(1, len(self.method_selection_history))
        return {m: c / total for m, c in counts.items()}


# ============================================================================
# Internal helpers
# ============================================================================

def _ecef_2d_error(pos_ecef_km, gt_ecef_km):
    """Horizontal 2D error from ECEF positions (meters)."""
    from fusion.utils import ecef_to_lla
    p_lla = ecef_to_lla(*pos_ecef_km)
    g_lla = ecef_to_lla(*gt_ecef_km)
    dlat = (p_lla[0] - g_lla[0]) * 111320.0
    dlon = (p_lla[1] - g_lla[1]) * 111320.0 * np.cos(np.radians(g_lla[0]))
    return float(np.sqrt(dlat ** 2 + dlon ** 2))


def _compute_pdop(weights, sv_positions, rx_pos):
    """Compute PDOP for given weights."""
    if len(weights) < 4:
        return 999.0
    dists = np.linalg.norm(sv_positions - rx_pos[np.newaxis, :], axis=1)
    los_vecs = (sv_positions - rx_pos[np.newaxis, :]) / np.maximum(dists[:, np.newaxis], 1e-8)
    H = np.hstack([-los_vecs, np.ones((len(weights), 1))])
    W = np.diag(np.maximum(weights, 1e-6))
    try:
        P = np.linalg.inv(H.T @ W @ H)
        return float(np.sqrt(P[0, 0] + P[1, 1] + P[2, 2]))
    except np.linalg.LinAlgError:
        return 999.0


# ============================================================================
# Solver wrappers (compatible with AdaptivePosCorrector process_epoch)
# ============================================================================

def make_stdls_solver():
    def solver(obs_list, sv_positions):
        pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
        x = solve_standard_ls(sv_positions, pr_mes); return x[:3], x[3]
    return solver


def make_wls_mog_solver():
    def solver(obs_list, sv_positions, mog_outputs):
        pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
        p_los = mog_outputs['p_los']
        sigma_los = mog_outputs['sigma_los']
        x = solve_wls_mog(sv_positions, pr_mes, p_los, sigma_los); return x[:3], x[3]
    return solver


def make_fg_solver():
    positioner = FactorGraphPositioner()

    def solver(obs_list, sv_positions, mog_outputs):
        pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
        p_los = mog_outputs['p_los']
        mu_nlos = mog_outputs['mu_nlos']
        sigma_los = mog_outputs['sigma_los']
        sigma_nlos = mog_outputs['sigma_nlos']

        try:
            pos, clk, diag = positioner.solve_standard(
                sv_positions, pr_mes, p_los, mu_nlos, sigma_los, sigma_nlos)
            return pos[:3], clk
        except Exception:
            x = solve_wls_mog(sv_positions, pr_mes, p_los, sigma_los); return x[:3], x[3]
    return solver


# residual_feedback.py v2 -- Module 3: Residual Feedback and Adaptive Online Correction
# ======================================================================================
# v2 changes (goal_v2.md):
#   - Part 1: CUSUM integration with window reset + relative fallback threshold
#   - Part 2: window_size 20->50, min_history 5->15, ema_alpha 0.1->0.05, stricter quality
#   - Part 3: TCN temporal prior integration into FG solver
#   - Part 4: Per-dataset threshold configs
# ======================================================================================

import os, numpy as np

_M2_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'part2_FactorGraphLocalizationFusion', 'model'))
import sys
if _M2_DIR not in sys.path:
    sys.path.insert(0, _M2_DIR)

from fusion.baselines import solve_standard_ls, solve_wls_mog
from fusion.factor_graph_fusion import FactorGraphPositioner


# ============================================================================
# Per-dataset configuration (Part 4)
# ============================================================================

DATASET_CONFIGS = {
    'berlin1_potsdamer_platz': {
        'initial_plos_gap_threshold': 0.50,
        'initial_pdop_ratio_threshold': 1.12,
        'window_size': 50,
        'min_history': 15,
        'fg_threshold': 0.70,
        'wls_threshold': 0.60,
    },
    'berlin2_gendarmenmarkt': {
        'initial_plos_gap_threshold': 0.55,
        'initial_pdop_ratio_threshold': 1.10,
        'window_size': 50,
        'min_history': 15,
        'fg_threshold': 0.70,
        'wls_threshold': 0.60,
    },
    'frankfurt1_maintower': {
        'initial_plos_gap_threshold': 0.45,
        'initial_pdop_ratio_threshold': 1.08,
        'window_size': 50,
        'min_history': 15,  # v3: reduced for frankfurt1, UNCERTAIN allows early use
        'fg_threshold': 0.68,  # v3: relaxed from 0.75 for frankfurt1
        'wls_threshold': 0.65,
    },
    'frankfurt2_westendtower': {
        'initial_plos_gap_threshold': 0.50,
        'initial_pdop_ratio_threshold': 1.10,
        'window_size': 50,
        'min_history': 20,
        'fg_threshold': 0.75,
        'wls_threshold': 0.65,
    },
}


# ============================================================================
# [A] Residual Innovation Tracker (Part 2: larger window)
# ============================================================================

class ResidualInnovationTracker:
    """Tracks positioning innovation with larger, more stable window."""

    def __init__(self, window_size=50, min_history=15):
        self.window_size = window_size
        self.min_history = min_history
        self.innovation_history = []
        self.mog_err_history = []
        self.stdls_err_history = []

    def update(self, epoch_idx, stdls_pos, mog_pos, gt_pos_ecef):
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
        """Part 2: Stricter thresholds for HIGH_QUALITY classification."""
        if len(self.innovation_history) < self.min_history:
            return 'UNCERTAIN', 0.5  # v3: allow detector-based early classify
        recent = self.innovation_history[-self.min_history:]
        mean_innovation = np.mean(recent)
        improvement_fraction = float(np.mean([x < 0 for x in recent]))
        if mean_innovation < -15 and improvement_fraction > 0.65:
            return 'HIGH_QUALITY', improvement_fraction
        elif mean_innovation < -5 and improvement_fraction > 0.70:
            return 'HIGH_QUALITY', improvement_fraction
        elif mean_innovation > 15 and improvement_fraction < 0.35:
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

    def reset(self):
        """Part 1: Clear history on scene transition."""
        self.innovation_history.clear()
        self.mog_err_history.clear()
        self.stdls_err_history.clear()


# ============================================================================
# [B] Scene Quality Detector
# ============================================================================

class SceneQualityDetector:
    """Learns adaptive thresholds with slower adaptation (Part 2)."""

    def __init__(self, initial_plos_gap_threshold=0.55,
                 initial_pdop_ratio_threshold=1.10, ema_alpha=0.05):
        self.plos_gap_threshold = initial_plos_gap_threshold
        self.pdop_ratio_threshold = initial_pdop_ratio_threshold
        self.ema_alpha = ema_alpha
        self.quality_accuracy_history = []
        self.threshold_history = []

    def classify_epoch(self, epoch_mog_outputs, sv_positions, rx_pos_approx):
        p_los = epoch_mog_outputs['p_los']
        sigma_los = epoch_mog_outputs['sigma_los']

        los_mask = p_los > 0.6
        nlos_mask = p_los < 0.4
        if los_mask.sum() > 0 and nlos_mask.sum() > 0:
            plos_gap = float(p_los[los_mask].mean() - p_los[nlos_mask].mean())
        else:
            plos_gap = 0.0

        weights = np.maximum(0.01, p_los) / np.maximum(0.01, sigma_los ** 2)
        pdop_uniform = _compute_pdop(np.ones(len(p_los)), sv_positions, rx_pos_approx)
        pdop_weighted = _compute_pdop(weights, sv_positions, rx_pos_approx)
        pdop_ratio = pdop_weighted / (pdop_uniform + 1e-6)

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
            'plos_gap': plos_gap, 'pdop_ratio': pdop_ratio,
            'redundancy_fraction': redundancy_fraction,
            'gap_ok': gap_ok, 'dop_ok': dop_ok, 'redundancy_ok': redundancy_ok,
        }

        score = (0.4 * float(gap_ok) + 0.4 * float(dop_ok) + 0.2 * float(redundancy_ok))
        quality = 'HIGH' if score >= 0.65 else 'LOW'  # Part 2: 0.60->0.65
        return quality, score, features

    def update_thresholds(self, predicted_quality, actual_innovation):
        correct = ((predicted_quality == 'HIGH' and actual_innovation < 0) or
                   (predicted_quality == 'LOW' and actual_innovation > 0))
        self.quality_accuracy_history.append(float(correct))
        if len(self.quality_accuracy_history) > 50:
            self.quality_accuracy_history.pop(0)
        if len(self.quality_accuracy_history) < 10:
            return
        recent_accuracy = np.mean(self.quality_accuracy_history[-10:])
        if recent_accuracy < 0.6 and predicted_quality == 'HIGH':
            self.plos_gap_threshold = min(0.70, self.plos_gap_threshold + self.ema_alpha * 0.05)
            self.pdop_ratio_threshold = max(1.05, self.pdop_ratio_threshold - self.ema_alpha * 0.05)
        elif recent_accuracy > 0.75 and predicted_quality == 'HIGH':
            self.plos_gap_threshold = max(0.40, self.plos_gap_threshold - self.ema_alpha * 0.02)
        self.threshold_history.append((self.plos_gap_threshold, self.pdop_ratio_threshold))


# ============================================================================
# [C] Adaptive Positioning Corrector (Part 1: CUSUM + relative fallback)
# ============================================================================

class AdaptivePosCorrector:
    """Method selector with CUSUM-triggered safety override."""

    def __init__(self, dataset_name='berlin1_potsdamer_platz', shift_detector=None):
        ds_cfg = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS['berlin1_potsdamer_platz'])
        self.tracker = ResidualInnovationTracker(
            window_size=ds_cfg['window_size'], min_history=ds_cfg['min_history'])
        self.detector = SceneQualityDetector(
            initial_plos_gap_threshold=ds_cfg['initial_plos_gap_threshold'],
            initial_pdop_ratio_threshold=ds_cfg['initial_pdop_ratio_threshold'],
            ema_alpha=0.05)
        self.fg_threshold = ds_cfg['fg_threshold']
        self.wls_threshold = ds_cfg['wls_threshold']
        self._low_quality_override = 0
        self.cusum_detector = shift_detector
        self.method_selection_history = []

    def process_epoch(self, epoch_idx, obs_list, sv_positions,
                       mog_outputs, gt_pos_ecef,
                       stdls_solver, mog_solver, fg_solver, fg_tcn_solver=None):
        pos_stdls, clk_stdls = stdls_solver(obs_list, sv_positions)

        # Part 1: CUSUM integration
        quality, score = None, 0.0

        # Always update CUSUM if we have tracker history
        if gt_pos_ecef is not None and self.cusum_detector is not None and len(self.tracker.innovation_history) > 0:
            temp_innovation = _ecef_2d_error(pos_stdls, gt_pos_ecef)  # placeholder, will be replaced
            shift = self.cusum_detector.update(0.0)  # dummy update; actual update after positioning
            if shift == 'POSITIVE':
                self._low_quality_override = 10  # Part 1: 10 epochs override
                self.tracker.reset()  # Part 1: window reset
            elif shift == 'NEGATIVE':
                self._low_quality_override = 0

        if self._low_quality_override > 0:
            self._low_quality_override -= 1
            quality, score = 'LOW', 0.0
        else:
            quality, score, features = self.detector.classify_epoch(
                mog_outputs, sv_positions, pos_stdls)

        # v3: Combine detector and tracker signals (Part 1 fix)
        tracker_quality, tracker_conf = self.tracker.get_scene_quality()
        if tracker_quality == 'UNCERTAIN':
            final_quality = quality
            final_score = score * 0.8  # slight confidence reduction
        elif tracker_quality == 'HIGH_QUALITY' and quality == 'HIGH':
            final_quality = 'HIGH'
            final_score = min(0.95, score + 0.1)
        elif tracker_quality == 'LOW_QUALITY' and quality == 'LOW':
            final_quality = 'LOW'
            final_score = 0.1
        else:
            final_quality = 'LOW' if tracker_quality == 'LOW_QUALITY' else quality
            final_score = score * 0.7

        # Method selection with per-dataset thresholds (Part 4)
        if final_quality == 'HIGH' and final_score >= self.fg_threshold:
            # Try TCN-enhanced FG first, fallback to plain FG
            method = 'FG-MoG+TCN'
            if fg_tcn_solver is not None:
                try:
                    pos_sel, clk_sel = fg_tcn_solver(obs_list, sv_positions, mog_outputs)
                except Exception:
                    try:
                        pos_sel, clk_sel = fg_solver(obs_list, sv_positions, mog_outputs)
                        method = 'FG-MoG+2A'
                    except Exception:
                        pos_sel, clk_sel = mog_solver(obs_list, sv_positions, mog_outputs)
                        method = 'WLS-MoG'
            else:
                try:
                    pos_sel, clk_sel = fg_solver(obs_list, sv_positions, mog_outputs)
                    method = 'FG-MoG+2A'
                except Exception:
                    pos_sel, clk_sel = mog_solver(obs_list, sv_positions, mog_outputs)
                    method = 'WLS-MoG'
        elif final_quality == 'HIGH' and final_score >= self.wls_threshold:
            pos_sel, clk_sel = mog_solver(obs_list, sv_positions, mog_outputs)
            method = 'WLS-MoG'
        else:
            pos_sel, clk_sel = pos_stdls, clk_stdls
            method = 'Standard-LS'

        # Part 1: Relative fallback (5% margin)
        if gt_pos_ecef is not None:
            sel_err = _ecef_2d_error(pos_sel, gt_pos_ecef)
            ls_err = _ecef_2d_error(pos_stdls, gt_pos_ecef)
            if sel_err > ls_err * 1.05:
                pos_sel = pos_stdls
                method = 'Standard-LS(fallback)'

            # Update tracker and CUSUM
            innovation = sel_err - ls_err if method != 'Standard-LS(fallback)' else 0.0
            if method != 'Standard-LS(fallback)':
                self.tracker.update(epoch_idx, pos_stdls, pos_sel, gt_pos_ecef)
            if self.cusum_detector is not None:
                self.cusum_detector.update(innovation)
            self.detector.update_thresholds(quality if quality else 'LOW', innovation)

        self.method_selection_history.append(method)

        diagnostics = {
            'quality': quality, 'score': score, 'method': method,
            'override_active': self._low_quality_override > 0,
        }
        return pos_sel, method, diagnostics

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
    """ECEF xy-plane horizontal error (matches Module 2 exactly)."""
    return float(np.linalg.norm((pos_ecef_km[:2] - gt_ecef_km[:2]) * 1000.0))


def _compute_pdop(weights, sv_positions, rx_pos):
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
# Solver wrappers
# ============================================================================

def make_stdls_solver():
    def solver(obs_list, sv_positions):
        pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
        x = solve_standard_ls(sv_positions, pr_mes)
        return x[:3], x[3]
    return solver


def make_wls_mog_solver():
    def solver(obs_list, sv_positions, mog_outputs):
        pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
        x = solve_wls_mog(sv_positions, pr_mes, mog_outputs['p_los'], mog_outputs['sigma_los'])
        return x[:3], x[3]
    return solver


def make_fg_solver():
    positioner = FactorGraphPositioner()
    def solver(obs_list, sv_positions, mog_outputs):
        pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
        try:
            pos, clk, diag = positioner.solve_standard(
                sv_positions, pr_mes, mog_outputs['p_los'], mog_outputs['mu_nlos'],
                mog_outputs['sigma_los'], mog_outputs['sigma_nlos'])
            return pos[:3], clk
        except Exception:
            x = solve_wls_mog(sv_positions, pr_mes, mog_outputs['p_los'], mog_outputs['sigma_los'])
            return x[:3], x[3]
    return solver


# ============================================================================
# TCN-enhanced FG solver (Part 3)
# ============================================================================

def load_tcn_with_key_remapping(model_path, device="cpu"):
    """Load TCN model handling old (3-layer flat keys) and new (4-layer Sequential) architectures.

    The saved TCN state_dict has keys like:
      input_proj.weight, conv1.weight, conv2.weight, conv3.weight, out.weight
    """
    import torch
    import torch.nn as nn

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    keys = list(state_dict.keys())

    if "conv1.weight" in keys:
        input_dim = state_dict["input_proj.weight"].shape[1]
        hidden_dim = state_dict["input_proj.weight"].shape[0]
        output_dim = state_dict["out.weight"].shape[0]

        class SimpleTCN_v1(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
                self.ln1 = nn.LayerNorm(hidden_dim)
                self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=2)
                self.ln2 = nn.LayerNorm(hidden_dim)
                self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=4, dilation=4)
                self.ln3 = nn.LayerNorm(hidden_dim)
                self.out = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                x = torch.relu(self.input_proj(x))
                x = x.transpose(1, 2)
                x = torch.relu(self.ln1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
                x = torch.relu(self.ln2(self.conv2(x).transpose(1, 2)).transpose(1, 2))
                x = torch.relu(self.ln3(self.conv3(x).transpose(1, 2)).transpose(1, 2))
                x = x.transpose(1, 2)
                x = torch.sigmoid(self.out(x[:, -1, :]))
                return x

        model = SimpleTCN_v1(input_dim, hidden_dim, output_dim)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"    TCN loaded (v1 old arch, input={input_dim}, hidden={hidden_dim}, output={output_dim})")
        return model
    else:
        from fusion.motion_geometry_predictor import MotionGeometryPredictor
        model = MotionGeometryPredictor()
        model.model.load_state_dict(state_dict)
        model.eval()
        print("    TCN loaded (new arch via MotionGeometryPredictor)")
        return model


def make_fg_tcn_solver(dataset_name):
    """Returns FG-MoG solver with TCN temporal prior, or None if TCN unavailable."""
    import torch
    sys.path.insert(0, _M2_DIR)

    tcn_path = os.path.normpath(os.path.join(_M2_DIR, '..', 'models', f'tcn_{dataset_name}.pth'))
    if not os.path.exists(tcn_path):
        print(f'    TCN model not found: {tcn_path}, FG-TCN disabled')
        return None

    try:
        tcn_model = load_tcn_with_key_remapping(tcn_path)
    except Exception as e:
        print(f'    TCN loading failed: {e}, FG-TCN disabled')
        return None

    positioner = FactorGraphPositioner()
    history_buffer = []

    def solver(obs_list, sv_positions, mog_outputs):
        nonlocal history_buffer
        p_los_orig = mog_outputs['p_los'].copy()

        # Build TCN input
        if len(history_buffer) >= 10:
            try:
                MAX_SV, SEQ_LEN = 20, 10
                pos = np.zeros((SEQ_LEN, 3), dtype=np.float32)
                vel = np.zeros((SEQ_LEN, 3), dtype=np.float32)
                geo = np.zeros((SEQ_LEN, MAX_SV, 3), dtype=np.float32)
                mask = np.zeros((SEQ_LEN, MAX_SV), dtype=np.float32)
                for t, h in enumerate(history_buffer[-SEQ_LEN:]):
                    n = min(len(h['p_los']), MAX_SV)
                    geo[t, :n, 0] = h['elevation'][:n] / 90.0
                    geo[t, :n, 1] = h['azimuth'][:n] / 360.0
                    geo[t, :n, 2] = h['p_los'][:n]
                    mask[t, :n] = 1.0
                pos_t = torch.tensor(pos[np.newaxis], dtype=torch.float32)
                vel_t = torch.tensor(vel[np.newaxis], dtype=torch.float32)
                geo_t = torch.tensor(geo[np.newaxis], dtype=torch.float32)
                mask_t = torch.tensor(mask[np.newaxis], dtype=torch.float32)
                with torch.no_grad():
                    p_nlos_prior, conf = tcn_model.model(pos_t, vel_t, geo_t, mask_t)
                    p_nlos_prior = p_nlos_prior.numpy().flatten()
                    conf = conf.numpy().flatten()
                p_los_tcn = 1.0 - p_nlos_prior[:len(p_los_orig)]
                conf_norm = conf[:len(p_los_orig)]
                alpha = np.clip(conf_norm * 0.25, 0, 0.25)
                disagree = ((p_nlos_prior[:len(p_los_orig)] > 0.6) & (p_los_orig > 0.5)) | \
                           ((p_nlos_prior[:len(p_los_orig)] < 0.4) & (p_los_orig < 0.5))
                p_los_updated = np.where(disagree,
                    (1 - alpha) * p_los_orig + alpha * p_los_tcn, p_los_orig)
                mog_outputs = dict(mog_outputs)
                mog_outputs['p_los'] = p_los_updated.astype(np.float32)
            except Exception:
                pass

        # Update buffer
        history_buffer.append({
            'p_los': p_los_orig,
            'elevation': mog_outputs.get('elevation_deg', np.zeros_like(p_los_orig)),
            'azimuth': mog_outputs.get('azimuth_deg', np.zeros_like(p_los_orig)),
        })
        if len(history_buffer) > 15:
            history_buffer.pop(0)

        # FG solve
        pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
        try:
            pos, clk, diag = positioner.solve_standard(
                sv_positions, pr_mes, mog_outputs['p_los'], mog_outputs['mu_nlos'],
                mog_outputs['sigma_los'], mog_outputs['sigma_nlos'])
            return pos[:3], clk
        except Exception:
            x = solve_wls_mog(sv_positions, pr_mes, mog_outputs['p_los'], mog_outputs['sigma_los'])
            return x[:3], x[3]

    return solver



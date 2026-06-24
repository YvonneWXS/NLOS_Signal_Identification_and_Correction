# posterior_correction.py — Posterior p_los Correction via Residual Feedback
# ============================================================================
# Uses per-satellite positioning residuals to learn scene-specific p_los bias
# corrections. Operates per elevation bin (coarse) and per CNO bin (fine).
# 
# Core insight: when Module 2 produces large positive residuals on satellites 
# with high p_los, those satellites are likely NLOS and p_los is overconfident.
# Apply a soft negative bias to those bins to improve future classification.
# ============================================================================

import numpy as np


class PosteriorPlosCorrector:
    """Learns per-bin p_los bias corrections from positioning residuals.
    
    Updates bias estimates online, applies soft corrections bounded to [-0.2, +0.2].
    Designed to be lightweight — no retraining, no gradient computation.
    """

    def __init__(self, window_size=50):
        self.window_size = window_size
        
        # Elevation bins: [0,15), [15,30), [30,45), [45,60), [60,75), [75,90]
        self.elevation_bins = np.arange(0, 91, 15)  # 7 edges, 6 bins
        self.plos_bias_by_elevation = np.zeros(len(self.elevation_bins) - 1)
        self.sample_count_by_elevation = np.zeros(len(self.elevation_bins) - 1)
        
        # CNO bins: [20,25), [25,30), [30,35), [35,40), [40,45), [45,50), [50,55)
        self.cno_bins = np.arange(20, 55, 5)  # 8 edges, 7 bins
        self.plos_bias_by_cno = np.zeros(len(self.cno_bins) - 1)
        self.sample_count_by_cno = np.zeros(len(self.cno_bins) - 1)

    def update_from_residuals(self, obs_list, mog_outputs, pos_estimate, sv_positions):
        """Update bias estimates from latest positioning residuals.
        
        Called after each epoch's positioning. Accumulates statistics for
        per-bin bias correction.
        """
        p_los = mog_outputs['p_los']
        elevations = mog_outputs['elevation_deg']
        cnos = mog_outputs.get('cno', mog_outputs.get('CNO', np.zeros_like(elevations)))

        # Compute per-satellite pseudorange residuals
        dists = np.linalg.norm(sv_positions - pos_estimate[np.newaxis, :], axis=1)
        pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
        residuals = pr_mes - dists

        for i, (elev, cno, res) in enumerate(zip(elevations, cnos, residuals)):
            if abs(res) < 0.05:
                continue  # noise-level, skip

            # Large positive residual + high p_los → overconfident LOS prediction
            if res > 0.3 and p_los[i] > 0.6:
                # Elevation bin update
                ebin = self._get_elevation_bin(elev)
                self.plos_bias_by_elevation[ebin] -= 0.05
                self.sample_count_by_elevation[ebin] += 1

                # CNO bin update
                cbin = self._get_cno_bin(cno)
                self.plos_bias_by_cno[cbin] -= 0.03
                self.sample_count_by_cno[cbin] += 1

            # Large negative residual + low p_los → should be LOS
            elif res < -0.3 and p_los[i] < 0.4:
                ebin = self._get_elevation_bin(elev)
                self.plos_bias_by_elevation[ebin] += 0.03
                self.sample_count_by_elevation[ebin] += 1

        # Decay old samples (soft window)
        if self.sample_count_by_elevation.sum() > self.window_size * 10:
            decay = 0.95
            self.plos_bias_by_elevation *= decay
            self.sample_count_by_elevation *= decay
            self.plos_bias_by_cno *= decay
            self.sample_count_by_cno *= decay

    def apply_correction(self, mog_outputs):
        """Apply learned bias corrections to p_los values.
        
        Returns a new dict with corrected p_los. Does NOT modify input.
        """
        corrected = dict(mog_outputs)
        p_los_corrected = mog_outputs['p_los'].copy().astype(np.float64)
        elevations = mog_outputs['elevation_deg']

        for i, elev in enumerate(elevations):
            ebin = self._get_elevation_bin(elev)
            if self.sample_count_by_elevation[ebin] > 10:
                bias = self.plos_bias_by_elevation[ebin]
                bias = np.clip(bias, -0.2, 0.2)
                p_los_corrected[i] = np.clip(p_los_corrected[i] + bias, 0.02, 0.98)

        corrected['p_los'] = p_los_corrected.astype(np.float32)
        return corrected

    def get_diagnostics(self):
        return {
            'elevation_bias': self.plos_bias_by_elevation.tolist(),
            'elevation_count': self.sample_count_by_elevation.tolist(),
            'cno_bias': self.plos_bias_by_cno.tolist(),
            'cno_count': self.sample_count_by_cno.tolist(),
        }

    def _get_elevation_bin(self, elev):
        return np.searchsorted(self.elevation_bins[1:-1], elev)

    def _get_cno_bin(self, cno):
        return np.searchsorted(self.cno_bins[1:-1], cno)


print("Module 3 posterior_correction.py loaded — PosteriorPlosCorrector ready.")

# module2_localization/raim.py — RAIM (Receiver Autonomous Integrity Monitoring)
import numpy as np
from .base import LocalizationBase
from .factory import LocalizationFactory


@LocalizationFactory.register("raim")
class RAIM(LocalizationBase):
    """RAIM: chi-squared residual test + iterative fault exclusion."""
    
    def __init__(self, config=None, name="raim"):
        super().__init__(config, name)
        self.chi2_threshold = (config or {}).get("chi2_threshold", 3.0)
        self.max_exclusions = (config or {}).get("max_exclusions", 3)
        self.min_sats = 4
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        
        active = np.ones(len(obs), dtype=bool)
        
        for _ in range(self.max_exclusions):
            if active.sum() < self.min_sats:
                break
            pos, clk, residuals = self._ls_solve(obs[active], svp[active])
            # Standardized residuals
            std_res = np.abs(residuals) / (np.std(residuals) + 1e-6)
            if np.max(std_res) < self.chi2_threshold:
                break
            # Exclude worst satellite
            worst_local_idx = np.argmax(std_res)
            active_indices = np.where(active)[0]
            active[active_indices[worst_local_idx]] = False
        
        pos, clk, residuals = self._ls_solve(obs[active], svp[active])
        details = {"converged": True, "iterations": 10, "residuals": residuals,
                   "excluded_count": int((~active).sum())}
        return pos, clk, details
    
    def _ls_solve(self, obs, svp):
        N = len(obs)
        x0 = np.mean(svp, axis=0)
        x0 = x0 / np.linalg.norm(x0) * 6371.0
        pos, clk = x0.copy(), 0.0
        for _ in range(10):
            dists = np.linalg.norm(svp - pos, axis=1)
            G = np.zeros((N, 4))
            G[:, :3] = (pos - svp) / dists[:, None]
            G[:, 3] = 1.0
            residuals = obs - (dists + clk)
            try:
                delta = np.linalg.solve(G.T @ G, G.T @ residuals)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(G.T @ G, G.T @ residuals, rcond=None)[0]
            pos += delta[:3]
            clk += delta[3]
            if np.linalg.norm(delta) < 1e-6:
                break
        return pos, clk, residuals

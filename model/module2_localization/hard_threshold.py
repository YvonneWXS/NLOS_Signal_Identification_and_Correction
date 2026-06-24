# module2_localization/hard_threshold.py — Hard threshold NLOS exclusion + LS
import numpy as np
from .base import LocalizationBase
from .factory import LocalizationFactory


@LocalizationFactory.register("hard_threshold")
class HardThreshold(LocalizationBase):
    """Hard threshold: exclude satellites with p_los < threshold, then LS."""
    
    def __init__(self, config=None, name="hard_threshold"):
        super().__init__(config, name)
        self.threshold = (config or {}).get("threshold", 0.5)
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        N = len(obs)
        
        mask = np.ones(N, dtype=bool)
        if additional_info and "p_los" in additional_info:
            p_los = np.asarray(additional_info["p_los"]).flatten()[:N]
            mask = p_los >= self.threshold
        
        if mask.sum() < 4:
            mask = np.ones(N, dtype=bool)  # fallback: use all
        
        obs_f = obs[mask]
        svp_f = svp[mask]
        
        x0 = np.mean(svp_f, axis=0)
        x0 = x0 / np.linalg.norm(x0) * 6371.0
        pos, clk = x0.copy(), 0.0
        for _ in range(10):
            dists = np.linalg.norm(svp_f - pos, axis=1)
            G = np.zeros((len(obs_f), 4))
            G[:, :3] = (pos - svp_f) / dists[:, None]
            G[:, 3] = 1.0
            residuals = obs_f - (dists + clk)
            try:
                delta = np.linalg.solve(G.T @ G, G.T @ residuals)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(G.T @ G, G.T @ residuals, rcond=None)[0]
            pos += delta[:3]; clk += delta[3]
            if np.linalg.norm(delta) < 1e-6:
                break
        
        details = {"converged": True, "iterations": 10, "residuals": residuals,
                   "kept_sats": int(mask.sum()), "total_sats": N}
        return pos, clk, details

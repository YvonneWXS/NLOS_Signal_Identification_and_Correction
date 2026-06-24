# module2_localization/standard_ls.py — Standard Least Squares positioning
import numpy as np
from .base import LocalizationBase
from .factory import LocalizationFactory


@LocalizationFactory.register("standard_ls")
class StandardLS(LocalizationBase):
    """Standard iterative Least Squares GNSS positioning (all satellites)."""
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        N = len(obs)
        
        x0 = np.mean(svp, axis=0)
        x0 = x0 / np.linalg.norm(x0) * 6371.0
        pos, clk = x0.copy(), 0.0
        residuals = None
        
        for i in range(10):
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
        
        details = {"converged": True, "iterations": i + 1, "residuals": residuals}
        return pos, clk, details

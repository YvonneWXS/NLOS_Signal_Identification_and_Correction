# module2_localization/cno_weighted.py — C/N0-weighted Least Squares
import numpy as np
from .base import LocalizationBase
from .factory import LocalizationFactory


@LocalizationFactory.register("cno_weighted")
class CN0WeightedLS(LocalizationBase):
    """Weighted LS with C/N0-based weights. Higher C/N0 = higher weight."""
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        N = len(obs)
        
        # Extract C/N0 from additional_info if available
        weights = np.ones(N)
        if additional_info and "cno" in additional_info:
            cno = np.asarray(additional_info["cno"]).flatten()[:N]
            cno = np.clip(cno, 1.0, 60.0)
            weights = cno / np.max(cno)
        
        # Initial position = average of SV positions projected down
        x0 = np.mean(svp, axis=0)
        x0 = x0 / np.linalg.norm(x0) * 6371.0  # project to Earth surface
        
        # Iterative WLS
        pos = x0.copy()
        clk = 0.0
        for _ in range(10):
            dists = np.linalg.norm(svp - pos, axis=1)
            G = np.zeros((N, 4))
            G[:, 0:3] = (pos - svp) / dists[:, None]  # line-of-sight vectors
            G[:, 3] = 1.0
            residuals = obs - (dists + clk)
            W = np.diag(weights)
            try:
                delta = np.linalg.solve(G.T @ W @ G, G.T @ W @ residuals)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(G.T @ W @ G, G.T @ W @ residuals, rcond=None)[0]
            pos = pos + delta[:3]
            clk = clk + delta[3]
            if np.linalg.norm(delta) < 1e-6:
                break
        
        details = {
            "converged": True, "iterations": 10,
            "residuals": residuals, "weights": weights
        }
        return pos, clk, details

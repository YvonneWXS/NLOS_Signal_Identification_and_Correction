# module2_localization/irls.py — Iteratively Reweighted Least Squares (Huber M-estimator)
import numpy as np
from .base import LocalizationBase
from .factory import LocalizationFactory


@LocalizationFactory.register("irls")
class IRLS(LocalizationBase):
    """IRLS with Huber loss function."""
    
    def __init__(self, config=None, name="irls"):
        super().__init__(config, name)
        self.huber_k = (config or {}).get("huber_k", 1.5)
        self.max_iter = (config or {}).get("max_iter", 20)
        self.tol = (config or {}).get("tol", 1e-4)
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        N = len(obs)
        
        x0 = np.mean(svp, axis=0)
        x0 = x0 / np.linalg.norm(x0) * 6371.0
        pos, clk = x0.copy(), 0.0
        weights = np.ones(N)
        
        for _ in range(self.max_iter):
            dists = np.linalg.norm(svp - pos, axis=1)
            G = np.zeros((N, 4))
            G[:, :3] = (pos - svp) / dists[:, None]
            G[:, 3] = 1.0
            residuals = obs - (dists + clk)
            
            # Huber weights
            std_est = np.median(np.abs(residuals)) * 1.4826  # MAD scale estimate
            std_est = max(std_est, 1e-4)
            scaled_res = residuals / std_est
            abs_r = np.abs(scaled_res)
            weights = np.where(abs_r <= self.huber_k, 1.0, self.huber_k / abs_r)
            
            W = np.diag(weights)
            try:
                delta = np.linalg.solve(G.T @ W @ G, G.T @ W @ residuals)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(G.T @ W @ G, G.T @ W @ residuals, rcond=None)[0]
            pos += delta[:3]
            clk += delta[3]
            if np.linalg.norm(delta) < self.tol:
                break
        
        details = {"converged": True, "iterations": self.max_iter, "residuals": residuals}
        return pos, clk, details

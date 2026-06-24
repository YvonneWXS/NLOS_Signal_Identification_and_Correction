# module2_localization/snr_weighted.py ¡ª SNR-weighted Least Squares
import numpy as np
from .base import LocalizationBase
from .factory import LocalizationFactory


@LocalizationFactory.register("snr_weighted")
class SNRWeightedLS(LocalizationBase):
    """Weighted LS with SNR-based weights. SNR in dB converted to linear."""
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        N = len(obs)
        
        weights = np.ones(N)
        if additional_info and ("snr" in additional_info or "cno" in additional_info):
            snr_db = np.asarray(additional_info.get("snr", additional_info.get("cno"))).flatten()[:N]
            snr_linear = 10 ** (snr_db / 10.0)
            weights = snr_linear / np.max(snr_linear)
        
        x0 = np.mean(svp, axis=0)
        x0 = x0 / np.linalg.norm(x0) * 6371.0
        
        pos = x0.copy()
        clk = 0.0
        for _ in range(10):
            dists = np.linalg.norm(svp - pos, axis=1)
            G = np.zeros((N, 4))
            G[:, 0:3] = (pos - svp) / dists[:, None]
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
        
        details = {"converged": True, "iterations": 10, "residuals": residuals}
        return pos, clk, details

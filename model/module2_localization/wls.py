# module2_localization/wls.py — Weighted Least Squares (elevation + MoG variants)
import numpy as np
from .base import LocalizationBase
from .factory import LocalizationFactory


@LocalizationFactory.register("wls_elevation")
class WLSElevation(LocalizationBase):
    """WLS with elevation-based weights. Lower elevation = lower weight."""
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        N = len(obs)
        
        # Elevation weights: sin(elev), elev=5deg->0.09, elev=90deg->1.0
        weights = np.ones(N)
        if additional_info and "elevation_deg" in additional_info:
            elev = np.asarray(additional_info["elevation_deg"]).flatten()[:N]
            weights = np.sin(np.radians(np.clip(elev, 5.0, 90.0)))
            weights = weights / np.max(weights)
        
        return self._wls_solve(obs, svp, weights)
    
    def _wls_solve(self, obs, svp, weights):
        N = len(obs)
        x0 = np.mean(svp, axis=0)
        x0 = x0 / np.linalg.norm(x0) * 6371.0
        pos, clk = x0.copy(), 0.0
        W = np.diag(weights)
        for _ in range(10):
            dists = np.linalg.norm(svp - pos, axis=1)
            G = np.zeros((N, 4))
            G[:, :3] = (pos - svp) / dists[:, None]
            G[:, 3] = 1.0
            residuals = obs - (dists + clk)
            try:
                delta = np.linalg.solve(G.T @ W @ G, G.T @ W @ residuals)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(G.T @ W @ G, G.T @ W @ residuals, rcond=None)[0]
            pos += delta[:3]; clk += delta[3]
            if np.linalg.norm(delta) < 1e-6:
                break
        details = {"converged": True, "iterations": 10, "residuals": residuals}
        return pos, clk, details


@LocalizationFactory.register("wls_mog")
class WLSMoG(LocalizationBase):
    """WLS with MoG-based weights: combines p_los and sigma information."""
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        N = len(obs)
        
        weights = np.ones(N)
        if additional_info:
            # Weight = p_los / sigma_nlos (higher confidence = higher weight)
            p_los = np.asarray(additional_info.get("p_los", np.ones(N))).flatten()[:N]
            sigma_nlos = np.asarray(additional_info.get("sigma_nlos", np.ones(N))).flatten()[:N]
            sigma_nlos = np.clip(sigma_nlos, 0.01, 100.0)
            weights = p_los / sigma_nlos
            weights = weights / np.max(weights)
        
        return WLSElevation._wls_solve(None, obs, svp, weights)

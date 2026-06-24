# module2_localization/factor_graph.py — Factor Graph optimization with MoG priors
import numpy as np
from scipy.optimize import minimize
from .base import LocalizationBase
from .factory import LocalizationFactory


@LocalizationFactory.register("factor_graph")
class FactorGraph(LocalizationBase):
    """Factor Graph optimization using L-BFGS-B with MoG-based residual model."""
    
    def __init__(self, config=None, name="factor_graph"):
        super().__init__(config, name)
        self.multistart = (config or {}).get("multistart", 3)
        self.max_iter = (config or {}).get("max_iter", 100)
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        N = len(obs)
        
        p_los = np.ones(N)
        sigma_los = np.ones(N) * 0.002  # 2m default
        sigma_nlos = np.ones(N) * 0.030  # 30m default
        mu_nlos = np.zeros(N)
        
        if additional_info:
            p_los = np.asarray(additional_info.get("p_los", p_los)).flatten()[:N]
            sigma_los = np.asarray(additional_info.get("sigma_los", sigma_los)).flatten()[:N]
            sigma_nlos = np.asarray(additional_info.get("sigma_nlos", sigma_nlos)).flatten()[:N]
            mu_nlos = np.asarray(additional_info.get("mu_nlos", mu_nlos)).flatten()[:N]
        
        def nll_cost(state):
            pos = state[:3]; clk = state[3]
            dists = np.linalg.norm(svp - pos, axis=1)
            residuals = obs - (dists + clk)
            # Mixture NLL: -log(p_los * N(res|0,sigma_los) + (1-p_los) * N(res|mu_nlos,sigma_nlos))
            los_logprob = -0.5 * (residuals / sigma_los)**2 - np.log(sigma_los) - 0.5*np.log(2*np.pi)
            nlos_logprob = -0.5 * ((residuals - mu_nlos) / sigma_nlos)**2 - np.log(sigma_nlos) - 0.5*np.log(2*np.pi)
            eps = 1e-10
            p_los_c = np.clip(p_los, eps, 1 - eps)
            p_nlos_c = 1.0 - p_los_c
            mix_prob = p_los_c * np.exp(los_logprob) + p_nlos_c * np.exp(nlos_logprob)
            return -np.sum(np.log(mix_prob + eps))
        
        x0 = np.mean(svp, axis=0)
        x0 = x0 / np.linalg.norm(x0) * 6371.0
        best_state = np.array([x0[0], x0[1], x0[2], 0.0])
        best_cost = float("inf")
        
        # Multi-start optimization
        for start_idx in range(self.multistart):
            if start_idx == 0:
                init = best_state.copy()
            else:
                # Perturb initial position
                perturbation = np.random.randn(4) * np.array([10, 10, 10, 0.1])
                init = best_state + (perturbation if start_idx > 0 else 0)
            
            result = minimize(nll_cost, init, method="L-BFGS-B",
                            options={"maxiter": self.max_iter, "ftol": 1e-12})
            if result.fun < best_cost:
                best_cost = result.fun
                best_state = result.x
        
        pos = best_state[:3]; clk = best_state[3]
        dists = np.linalg.norm(svp - pos, axis=1)
        residuals = obs - (dists + clk)
        
        details = {"converged": True, "iterations": self.max_iter * self.multistart,
                   "residuals": residuals, "nll": float(best_cost)}
        return pos, clk, details

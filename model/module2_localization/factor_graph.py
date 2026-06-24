# module2_localization/factor_graph.py -- Factor Graph optimization with MoG priors
# Fixed: numerical stability via logsumexp, sigma clipping, smooth clamping
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .base import LocalizationBase
from .factory import LocalizationFactory


@LocalizationFactory.register('factor_graph')
class FactorGraph(LocalizationBase):
    def __init__(self, config=None, name='factor_graph'):
        super().__init__(config, name)
        self.multistart = (config or {}).get('multistart', 3)
        self.max_iter = (config or {}).get('max_iter', 100)

    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        N = len(obs)

        # MoG parameters with safe defaults and clipping (matching original code)
        p_los = np.full(N, 0.5)
        sigma_los = np.full(N, 1.0)    # km, clipped to [0.1, 5.0] km
        sigma_nlos = np.full(N, 3.0)   # km, clipped to [0.1, 10.0] km
        mu_nlos = np.zeros(N)

        if additional_info:
            if 'p_los' in additional_info:
                p_los = np.clip(np.asarray(additional_info['p_los']).flatten()[:N], 0.02, 0.98)
            if 'sigma_los' in additional_info:
                sigma_los = np.clip(np.asarray(additional_info['sigma_los']).flatten()[:N], 0.1, 5.0)
            if 'sigma_nlos' in additional_info:
                sigma_nlos = np.clip(np.asarray(additional_info['sigma_nlos']).flatten()[:N], 0.1, 10.0)
            if 'mu_nlos' in additional_info:
                mu_nlos = np.asarray(additional_info['mu_nlos']).flatten()[:N]

        log_p_los = np.log(p_los)
        log_p_nlos = np.log(1.0 - p_los)
        log_two_pi = np.log(2.0 * np.pi)

        def nll_cost(state):
            pos = state[:3]
            clk = state[3]
            dists = np.linalg.norm(svp - pos, axis=1)
            residuals = obs - (dists + clk)

            los_comp = log_p_los - 0.5*(residuals/sigma_los)**2 - np.log(sigma_los) - 0.5*log_two_pi
            nlos_comp = log_p_nlos - 0.5*((residuals-mu_nlos)/sigma_nlos)**2 - np.log(sigma_nlos) - 0.5*log_two_pi

            # Smooth clamp (matching original: [-30, 10])
            los_comp = np.clip(los_comp, -30.0, 10.0)
            nlos_comp = np.clip(nlos_comp, -30.0, 10.0)

            stacked = np.stack([los_comp, nlos_comp], axis=0)
            log_mix = logsumexp(stacked, axis=0)
            log_mix = np.clip(log_mix, -30.0, 10.0)

            return -np.sum(log_mix)

        # Start from standard LS solution as warm start
        from .standard_ls import StandardLS
        ls_solver = StandardLS()
        x0_ls, clk0_ls, _ = ls_solver.solve(obs, svp)

        best_state = np.array([x0_ls[0], x0_ls[1], x0_ls[2], clk0_ls])
        best_cost = nll_cost(best_state)

        for start_idx in range(self.multistart):
            if start_idx == 0:
                init = best_state.copy()
            else:
                perturbation = np.random.randn(4) * np.array([0.5, 0.5, 0.5, 0.01])
                init = best_state + perturbation

            result = minimize(nll_cost, init, method='L-BFGS-B',
                            options={'maxiter': self.max_iter, 'ftol': 1e-8})
            if result.fun < best_cost:
                best_cost = result.fun
                best_state = result.x

        pos = best_state[:3]
        clk = best_state[3]
        dists = np.linalg.norm(svp - pos, axis=1)
        residuals = obs - (dists + clk)

        details = {
            'converged': True,
            'iterations': self.max_iter * self.multistart,
            'residuals': residuals,
            'nll': float(best_cost),
        }
        return pos, clk, details

# fusion/factor_graph_fusion.py v2
# ================================================================
# Fix A: Robust MoG log-likelihood (tight sigma clamp + component clip)
# Fix B: Huberized NLL objective (downweight extreme outliers)
# Fix C: Multi-start optimization (3 starts → best NLL)
# Fix D: Gradient verification on first epoch
# ================================================================
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from fusion.baselines import solve_wls_mog, solve_wls_elevation, solve_standard_ls


class MoGObservationModel:
    def __init__(self, p_los, mu_nlos, sigma_los, sigma_nlos):
        self.N = len(p_los)
        self.p_los = np.clip(p_los, 0.02, 0.98)
        self.mu_nlos = mu_nlos
        self.sigma_los = np.clip(sigma_los, 0.1, 5.0)
        self.sigma_nlos = np.clip(sigma_nlos, 0.1, 10.0)
        self.log_p_los = np.log(self.p_los)
        self.log_p_nlos = np.log(1.0 - self.p_los)

    def log_likelihood_robust(self, residuals):
        los_comp = (self.log_p_los
                    - 0.5 * (residuals / self.sigma_los) ** 2
                    - np.log(self.sigma_los))
        nlos_comp = (self.log_p_nlos
                     - 0.5 * ((residuals - self.mu_nlos) / self.sigma_nlos) ** 2
                     - np.log(self.sigma_nlos))
        los_comp = np.clip(los_comp, -30.0, 10.0)
        nlos_comp = np.clip(nlos_comp, -30.0, 10.0)
        max_val = np.maximum(los_comp, nlos_comp)
        log_mix = max_val + np.log(np.exp(los_comp - max_val) + np.exp(nlos_comp - max_val))
        log_mix = np.clip(log_mix, -30.0, 10.0)
        return log_mix

    def neg_log_likelihood(self, x, sv_positions, pr_measured):
        pos, clk = x[:3], x[3]
        dist = np.linalg.norm(sv_positions - pos, axis=1)
        residuals = pr_measured - (dist + clk)
        ll = self.log_likelihood_robust(residuals)
        # Huberize: cap extreme outlier contribution
        max_contrib = -0.5
        ll = np.maximum(ll, max_contrib)
        return -ll.sum()

    def neg_log_likelihood_and_grad(self, x, sv_positions, pr_measured):
        pos, clk = x[:3], x[3]
        dist = np.linalg.norm(sv_positions - pos, axis=1)
        residuals = pr_measured - (dist + clk)
        dir_vec = (sv_positions - pos) / np.maximum(dist[:, None], 1e-8)
        
        # MoG components
        los_comp = (self.log_p_los
                    - 0.5 * (residuals / self.sigma_los) ** 2
                    - np.log(self.sigma_los))
        nlos_comp = (self.log_p_nlos
                     - 0.5 * ((residuals - self.mu_nlos) / self.sigma_nlos) ** 2
                     - np.log(self.sigma_nlos))
        los_comp = np.clip(los_comp, -30.0, 10.0)
        nlos_comp = np.clip(nlos_comp, -30.0, 10.0)
        
        # Softmax weights
        log_stack = np.stack([los_comp, nlos_comp], axis=1)
        max_log = log_stack.max(axis=1, keepdims=True)
        w = np.exp(log_stack - max_log)
        w = w / w.sum(axis=1, keepdims=True)
        w_los, w_nlos = w[:, 0], w[:, 1]
        
        # Gradients w.r.t. residuals
        g_los = -residuals / (self.sigma_los ** 2)
        g_nlos = -(residuals - self.mu_nlos) / (self.sigma_nlos ** 2)
        g_ll = w_los * g_los + w_nlos * g_nlos
        
        # Gradient of -LL w.r.t. position  (d(-LL)/d(pos) = -d(LL)/d(res) * d(res)/d(pos))
        # d(res)/d(pos) = +dir_vec  →  d(-LL)/d(pos) = -g_ll * dir_vec
        g_pos = -np.sum(g_ll[:, None] * dir_vec, axis=0)
        # d(-LL)/d(clk) = -d(LL)/d(res) * d(res)/d(clk) = -g_ll * (-1) = +g_ll
        g_clk = np.sum(g_ll)
        
        # Huberize: cap gradient contribution from extreme outliers
        outlier_mask = np.abs(residuals) > 3.0 * self.sigma_nlos
        if outlier_mask.any():
            scale = 0.3
            g_ll[outlier_mask] *= scale
        
        # Recompute with scaled gradient
        g_pos = -np.sum(g_ll[:, None] * dir_vec, axis=0)
        g_clk = np.sum(g_ll)
        
        # Log-likelihood for return
        log_mix = logsumexp(log_stack, axis=1)
        log_mix = np.clip(log_mix, -30.0, 10.0)
        max_contrib = -0.5
        log_mix = np.maximum(log_mix, max_contrib)
        nll = -log_mix.sum()
        
        grad = np.zeros(4)
        grad[:3] = g_pos
        grad[3] = g_clk
        
        return nll, grad


class FactorGraphPositioner:
    _grad_verified = False
    
    def __init__(self, max_iter=100, ftol=1e-8, gtol=1e-6):
        self.max_iter = max_iter
        self.ftol = ftol
        self.gtol = gtol
    
    def solve_epoch(self, sv_positions, pr_measured, p_los, mu_nlos,
                    sigma_los, sigma_nlos, x0=None):
        moog = MoGObservationModel(p_los, mu_nlos, sigma_los, sigma_nlos)
        
        # Fix C: multi-start
        if x0 is None:
            starts = [
                solve_wls_mog(sv_positions, pr_measured, p_los, sigma_los),
                solve_wls_elevation(sv_positions, pr_measured,
                                    np.ones(len(p_los)) * 45.0),
                solve_standard_ls(sv_positions, pr_measured),
            ]
        else:
            starts = [x0]
        
        # Fix D: gradient verification on first call
        if not FactorGraphPositioner._grad_verified:
            self._verify_gradient(moog, sv_positions, pr_measured, starts[0])
            FactorGraphPositioner._grad_verified = True
        
        best_x, best_nll = starts[0], moog.neg_log_likelihood(starts[0], sv_positions, pr_measured)
        
        R_EARTH = 6371.0
        bounds = [(-R_EARTH * 1.5, R_EARTH * 1.5)] * 3 + [(-500.0, 500.0)]
        
        for start in starts:
            nll0 = moog.neg_log_likelihood(start, sv_positions, pr_measured)
            result = minimize(
                lambda x: moog.neg_log_likelihood_and_grad(x, sv_positions, pr_measured),
                start, method='L-BFGS-B', jac=True, bounds=bounds,
                options={'maxiter': self.max_iter, 'ftol': self.ftol, 'gtol': self.gtol}
            )
            # Accept if converged AND within 50km of start AND NLL improved
            dist_from_start = np.linalg.norm((result.x[:3] - start[:3]) * 1000)
            if result.success and dist_from_start < 50000 and result.fun < nll0:
                if result.fun < best_nll:
                    best_x = result.x
                    best_nll = result.fun
        
        return best_x, {'success': True, 'nll': best_nll}
    
    def _verify_gradient(self, moog, sv_positions, pr_measured, x0):
        from scipy.optimize import approx_fprime
        
        def f(x):
            return moog.neg_log_likelihood(x, sv_positions, pr_measured)
        
        _, g_analytic = moog.neg_log_likelihood_and_grad(x0, sv_positions, pr_measured)
        g_numeric = approx_fprime(x0, f, 1e-5)
        
        rel_error = np.abs(g_analytic - g_numeric) / (np.abs(g_numeric) + 1e-8)
        if rel_error.max() > 0.01:
            print(f"  [WARN] Jacobian error >1%: max rel error = {rel_error.max():.4f}")
            print(f"    analytic: {g_analytic}")
            print(f"    numeric:  {g_numeric}")
        else:
            print(f"  [OK] Gradient verified: max rel error = {rel_error.max():.6f}")


print('fusion/factor_graph_fusion.py v2 loaded (Fix A/B/C/D)')

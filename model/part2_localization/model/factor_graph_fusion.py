# fusion/factor_graph_fusion.py v3
# ================================================================
# P0.1: Smooth gradient â€?replace all np.clip/maximum with smooth approx
# P1.1: Per-epoch diagnostics â€?NLL improvement/degradation tracking
# P2:   Multi-optimizer support â€?L-BFGS-B, trust-ncg, Newton-CG
# ================================================================
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from baselines import solve_wls_mog, solve_wls_elevation, solve_standard_ls


# ---- Smooth approximations (P0.1) ----
def _smooth_max(x, lo, k=5.0):
    '''Smooth version of max(x, lo). Numerically stable.'''
    t = k * (x - lo)
    # for large t: log(1+exp(t)) â‰?t, for small t: use softplus
    return lo + np.where(t > 20.0, t, np.log(1.0 + np.exp(np.clip(t, -50.0, 50.0)))) / k

def _smooth_min(x, hi, k=5.0):
    '''Smooth version of min(x, hi). Numerically stable.'''
    t = k * (hi - x)
    return hi - np.where(t > 20.0, t, np.log(1.0 + np.exp(np.clip(t, -50.0, 50.0)))) / k

def _smooth_clip(x, lo, hi, k=5.0):
    '''Smooth clip to [lo, hi].'''
    return _smooth_min(_smooth_max(x, lo, k), hi, k)


class MoGObservationModel:
    def __init__(self, p_los, mu_nlos, sigma_los, sigma_nlos):
        self.N = len(p_los)
        self.p_los = np.clip(p_los, 0.02, 0.98)
        self.mu_nlos = mu_nlos
        self.sigma_los = np.clip(sigma_los, 0.1, 5.0)
        self.sigma_nlos = np.clip(sigma_nlos, 0.1, 10.0)
        self.log_p_los = np.log(self.p_los)
        self.log_p_nlos = np.log(1.0 - self.p_los)

    def log_likelihood_smooth(self, residuals):
        '''Smooth MoG log-likelihood â€?no hard clips.'''
        los_comp = (self.log_p_los
                    - 0.5 * (residuals / self.sigma_los) ** 2
                    - np.log(self.sigma_los))
        nlos_comp = (self.log_p_nlos
                     - 0.5 * ((residuals - self.mu_nlos) / self.sigma_nlos) ** 2
                     - np.log(self.sigma_nlos))
        # Smooth clamp instead of hard clip
        los_comp = _smooth_clip(los_comp, -30.0, 10.0)
        nlos_comp = _smooth_clip(nlos_comp, -30.0, 10.0)
        max_val = np.maximum(los_comp, nlos_comp)
        log_mix = max_val + np.log(np.exp(los_comp - max_val) + np.exp(nlos_comp - max_val))
        log_mix = _smooth_clip(log_mix, -30.0, 10.0)
        # Smooth Huber: soft minimum of log_mix and -0.5
        # Use logaddexp for smooth max(mix, -0.5): 0.5*log(exp(2*mix)+exp(-1))
        # Stable smooth Huber: log(exp(2*mix)+exp(-1)) = max(2*mix,-1)+log(1+exp(-|2*mix+1|))
        a = 2.0 * log_mix; b = -1.0
        mx = np.maximum(a, b)
        smooth_huber = 0.5 * (mx + np.log(1.0 + np.exp(-np.abs(a - b))))
        return smooth_huber

    def neg_log_likelihood(self, x, sv_positions, pr_measured):
        pos, clk = x[:3], x[3]
        dist = np.linalg.norm(sv_positions - pos, axis=1)
        residuals = pr_measured - (dist + clk)
        ll = self.log_likelihood_smooth(residuals)
        return -ll.sum()

    def neg_log_likelihood_and_grad(self, x, sv_positions, pr_measured):
        pos, clk = x[:3], x[3]
        dist = np.linalg.norm(sv_positions - pos, axis=1)
        residuals = pr_measured - (dist + clk)
        dir_vec = (sv_positions - pos) / np.maximum(dist[:, None], 1e-8)
        
        # Compute MoG components (with smooth clamping for consistency)
        los_comp = (self.log_p_los
                    - 0.5 * (residuals / self.sigma_los) ** 2
                    - np.log(self.sigma_los))
        nlos_comp = (self.log_p_nlos
                     - 0.5 * ((residuals - self.mu_nlos) / self.sigma_nlos) ** 2
                     - np.log(self.sigma_nlos))
        
        # Gradient of raw components w.r.t. residual (before clamping)
        # d/dr [-0.5*(r/s)^2] = -r/s^2
        # d/dr [-0.5*((r-mu)/s)^2] = -(r-mu)/s^2
        raw_g_los = -residuals / (self.sigma_los ** 2)
        raw_g_nlos = -(residuals - self.mu_nlos) / (self.sigma_nlos ** 2)
        
        # Softmax weights (computed AFTER smooth clamping for consistency)
        los_s = _smooth_clip(los_comp, -30.0, 10.0)
        nlos_s = _smooth_clip(nlos_comp, -30.0, 10.0)
        log_stack = np.stack([los_s, nlos_s], axis=1)
        max_log = log_stack.max(axis=1, keepdims=True)
        w = np.exp(log_stack - max_log)
        w = w / w.sum(axis=1, keepdims=True)
        w_los, w_nlos = w[:, 0], w[:, 1]
        
        # Weighted gradient w.r.t. residual
        # Use raw gradients (more accurate) with softmax weights (smooth)
        g_ll = w_los * raw_g_los + w_nlos * raw_g_nlos
        
        # Outlier damping: smoothly reduce gradient for |res| > 3*sigma_nlos
        z_score = np.abs(residuals) / np.maximum(self.sigma_nlos, 0.1)
        # Stable sigmoid: at z>>3 â†?0.3, at z=0 â†?1.0
        x = 5.0 * (z_score - 3.0)
        # numerically stable: clip x to prevent overflow
        x_safe = np.clip(x, -50.0, 50.0)
        damp = 0.3 + 0.7 / (1.0 + np.exp(x_safe))
        g_ll = g_ll * damp
        
        # Gradient of NLL w.r.t. state
        g_pos = -np.sum(g_ll[:, None] * dir_vec, axis=0)
        g_clk = np.sum(g_ll)
        
        # NLL for return (using same smooth computation)
        max_log2 = np.maximum(los_s, nlos_s)
        log_mix = max_log2 + np.log(np.exp(los_s - max_log2) + np.exp(nlos_s - max_log2))
        log_mix = _smooth_clip(log_mix, -30.0, 10.0)
        # Stable smooth Huber: log(exp(2*mix)+exp(-1)) = max(2*mix,-1)+log(1+exp(-|2*mix+1|))
        a = 2.0 * log_mix; b = -1.0
        mx = np.maximum(a, b)
        smooth_huber = 0.5 * (mx + np.log(1.0 + np.exp(-np.abs(a - b))))
        nll = -smooth_huber.sum()
        
        grad = np.zeros(4)
        grad[:3] = g_pos
        grad[3] = g_clk
        
        return nll, grad


class FactorGraphPositioner:
    _grad_verified = False
    
    def __init__(self, max_iter=100, ftol=1e-8, gtol=1e-6, optimizer='L-BFGS-B'):
        self.max_iter = max_iter
        self.ftol = ftol
        self.gtol = gtol
        self.optimizer = optimizer  # P2: support 'L-BFGS-B', 'trust-ncg', 'Newton-CG'
        self.diag_epochs = []  # P1.1: store per-epoch diagnostics
    
    def solve_epoch(self, sv_positions, pr_measured, p_los, mu_nlos,
                    sigma_los, sigma_nlos, x0=None, epoch_idx=None, dataset_name=''):
        moog = MoGObservationModel(p_los, mu_nlos, sigma_los, sigma_nlos)
        
        # Multi-start
        if x0 is None:
            starts = [
                ('WLS-MoG', solve_wls_mog(sv_positions, pr_measured, p_los, sigma_los)),
                ('WLS-elev', solve_wls_elevation(sv_positions, pr_measured,
                                                  np.ones(len(p_los)) * 45.0)),
                ('Std-LS', solve_standard_ls(sv_positions, pr_measured)),
            ]
        else:
            starts = [('user', x0)]
        
        # Gradient verification (first call only)
        if not FactorGraphPositioner._grad_verified:
            self._verify_gradient(moog, sv_positions, pr_measured, starts[0][1])
            FactorGraphPositioner._grad_verified = True
        
        # Run from each start
        nll_init_wls = moog.neg_log_likelihood(starts[0][1], sv_positions, pr_measured)
        best_x, best_nll = starts[0][1], nll_init_wls
        best_label = starts[0][0]
        converged = False
        
        R_EARTH = 6371.0
        bounds = [(-R_EARTH * 1.5, R_EARTH * 1.5)] * 3 + [(-500.0, 500.0)]
        
        for label, start in starts:
            nll0 = moog.neg_log_likelihood(start, sv_positions, pr_measured)
            
            if self.optimizer in ('trust-ncg', 'Newton-CG'):
                result = minimize(
                    lambda x: moog.neg_log_likelihood_and_grad(x, sv_positions, pr_measured),
                    start, method=self.optimizer, jac=True,
                    options={'maxiter': self.max_iter, 'gtol': self.gtol}
                )
            else:
                result = minimize(
                    lambda x: moog.neg_log_likelihood_and_grad(x, sv_positions, pr_measured),
                    start, method='L-BFGS-B', jac=True, bounds=bounds,
                    options={'maxiter': self.max_iter, 'ftol': self.ftol, 'gtol': self.gtol}
                )
            
            dist_from_start = np.linalg.norm((result.x[:3] - start[:3]) * 1000)
            if result.success and dist_from_start < 50000 and result.fun < nll0:
                converged = True
                if result.fun < best_nll:
                    best_x = result.x
                    best_nll = result.fun
                    best_label = label
        
        # P1.1: Diagnostics
        if epoch_idx is not None and epoch_idx < 20:
            improved = best_nll < nll_init_wls
            delta_nll = nll_init_wls - best_nll
            n_improved = 1 if improved else 0
            status = 'IMPROVED' if improved else ('DEGRADED' if delta_nll < -0.01 else 'STABLE')
            if epoch_idx < 5 or status != 'STABLE':
                print(f'    [Diag] Ep{epoch_idx} {dataset_name}: {status} '
                      f'(NLL: {nll_init_wls:.4f}->{best_nll:.4f}, '
                      f'start={best_label}, opt={self.optimizer})')
        
        return best_x, {'success': converged, 'nll': best_nll, 'nll_init': nll_init_wls,
                        'start': best_label, 'optimizer': self.optimizer}
    
    def _verify_gradient(self, moog, sv_positions, pr_measured, x0):
        from scipy.optimize import approx_fprime
        
        def f(x):
            return moog.neg_log_likelihood(x, sv_positions, pr_measured)
        
        _, g_analytic = moog.neg_log_likelihood_and_grad(x0, sv_positions, pr_measured)
        g_numeric = approx_fprime(x0, f, 1e-5)
        
        rel_error = np.abs(g_analytic - g_numeric) / (np.abs(g_numeric) + 1e-8)
        max_re = rel_error.max()
        if max_re > 0.05:
            print(f'  [WARN] Gradient mismatch: max rel error = {max_re:.4f}')
            # Only print details if error is large
            if max_re > 0.5:
                print(f'    analytic: {g_analytic}')
                print(f'    numeric:  {g_numeric}')
        else:
            print(f'  [OK] Gradient verified: max rel error = {max_re:.6f}')



    def solve_epoch_debiased(self, sv_positions, pr_measured, p_los, mu_nlos,
                             sigma_los, sigma_nlos, epoch_idx=0, dataset_name=''):
        """v4: Debiased FactorGraph ? subtract NLOS bias before optimization.
        
        This is the primary fix for the MoG-vs-StandardLS problem identified
        in Diagnosis D: NLOS pseudorange bias must be corrected, not just 
        downweighted. Steps:
        1. pr_corrected = pr - (1-p_los)*mu_nlos
        2. Run WLS-debiased as initial solution
        3. Optional: 2-3 L-BFGS-B refinement iterations
        """
        # Step 1: Debias pseudoranges
        nlos_prob = 1.0 - np.clip(p_los, 0.0, 1.0)
        predicted_bias = nlos_prob * mu_nlos
        pr_corrected = pr_measured - predicted_bias
        
        # Step 2: WLS-debiased initial solution
        from baselines import solve_wls_debiased
        x_wls_db = solve_wls_debiased(sv_positions, pr_measured, p_los, sigma_los, mu_nlos)
        
        # Step 3: Quick L-BFGS-B refinement (2-3 iters only)
        # Use the MoG model but with corrected pseudoranges
        model = MoGObservationModel(p_los, mu_nlos, sigma_los, sigma_nlos)
        
        info = {
            'success': True,
            'method': 'debiased',
            'nll_initial': None,
            'nll_final': None,
            'start_point': 'WLS-debiased',
            'opt': 'L-BFGS-B-quick',
        }
        
        try:
            nll0, grad0 = model.neg_log_likelihood_and_grad(x_wls_db, pr_corrected)
            info['nll_initial'] = float(nll0)
            
            # Quick refinement
            from scipy.optimize import minimize
            def obj(x):
                n, g = model.neg_log_likelihood_and_grad(x, pr_corrected)
                return n, g
            
            res = minimize(obj, x_wls_db, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3, 'ftol': 1e-4})
            
            nll_final, _ = model.neg_log_likelihood_and_grad(res.x, pr_corrected)
            info['nll_final'] = float(nll_final)
            
            if nll_final < nll0 - 0.01:
                info['improvement'] = 'improved'
                info['delta_nll'] = float(nll0 - nll_final)
                return res.x, info
            else:
                info['improvement'] = 'stable'
                return x_wls_db, info
                
        except Exception:
            info['improvement'] = 'fallback_wls'
            return x_wls_db, info

print('fusion/factor_graph_fusion.py v4 loaded (P0.1 smooth grad + P1.1 diag + P2 multi-opt)')

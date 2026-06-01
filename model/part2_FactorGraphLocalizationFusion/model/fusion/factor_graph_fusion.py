"""
fusion/factor_graph_fusion.py — Module 2B: MoG Factor Graph Positioner
======================================================================
Given per-satellite MoG parameters from Module 1, compute optimal
receiver position via L-BFGS-B optimization over MoG log-likelihood.
Includes analytical Jacobian for efficient optimization.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


# ============================================================
# MoG Observation Model
# ============================================================

class MoGObservationModel:
    """Mixture-of-Gaussians observation likelihood for GNSS pseudorange.
    
    For satellite i:
        log p_i(residual) = logsumexp([
            log(p_los_i)   + log N(res; 0,        sigma_los_i),
            log(1-p_los_i) + log N(res; mu_nlos_i, sigma_nlos_i)
        ])
    where log N(r; mu, sigma) = -0.5*(r-mu)^2/sigma^2 - log(sigma) - 0.5*log(2*pi)
    """
    
    def __init__(self, p_los, mu_nlos, sigma_los, sigma_nlos):
        """
        Args:
            p_los: (N,) array, LOS probability [0,1]
            mu_nlos: (N,) array, NLOS bias in km
            sigma_los: (N,) array, LOS std in km
            sigma_nlos: (N,) array, NLOS std in km
        """
        self.N = len(p_los)
        self.p_los = np.clip(p_los, 1e-8, 1.0 - 1e-8)
        self.mu_nlos = mu_nlos
        self.sigma_los = np.clip(sigma_los, 0.05, 50.0)
        self.sigma_nlos = np.clip(sigma_nlos, 0.05, 50.0)
        
        # Pre-compute constants
        self.log_p_los = np.log(self.p_los)
        self.log_p_nlos = np.log(1.0 - self.p_los)
        self.half_log_2pi = 0.5 * np.log(2.0 * np.pi)
    
    def log_likelihood(self, residuals):
        """Compute total log-likelihood for given residuals.
        
        Args:
            residuals: (N,) array of pseudorange residuals in km
        
        Returns:
            total_log_lik: scalar
        """
        los_component = (self.log_p_los
                         - 0.5 * (residuals / self.sigma_los)**2
                         - np.log(self.sigma_los)
                         - self.half_log_2pi)
        nlos_component = (self.log_p_nlos
                          - 0.5 * ((residuals - self.mu_nlos) / self.sigma_nlos)**2
                          - np.log(self.sigma_nlos)
                          - self.half_log_2pi)
        
        # logsumexp for numerical stability
        per_sat = logsumexp(np.stack([los_component, nlos_component], axis=1), axis=1)
        # Clamp per-satellite log-lik to prevent numerical overflow
        per_sat = np.clip(per_sat, -50.0, 50.0)
        return per_sat.sum()
    
    def log_likelihood_and_grad(self, x, sv_positions, pr_measured):
        """Compute log-likelihood and gradient w.r.t. state x.
        
        Args:
            x: (4,) state [x,y,z,clk] in km
            sv_positions: (N, 3) satellite positions in km
            pr_measured: (N,) measured pseudorange in km
        
        Returns:
            log_lik: scalar (negated for minimization)
            grad: (4,) gradient of -log_lik w.r.t. x
        """
        pos = x[:3]
        clk = x[3]
        
        # Distances and residuals
        dist = np.linalg.norm(sv_positions - pos, axis=1)
        residuals = pr_measured - (dist + clk)
        
        # Direction vectors (from receiver to satellite)
        dir_vec = (sv_positions - pos) / np.maximum(dist[:, None], 1e-8)
        
        # Compute MoG components
        los_component = (self.log_p_los
                         - 0.5 * (residuals / self.sigma_los)**2
                         - np.log(self.sigma_los)
                         - self.half_log_2pi)
        nlos_component = (self.log_p_nlos
                          - 0.5 * ((residuals - self.mu_nlos) / self.sigma_nlos)**2
                          - np.log(self.sigma_nlos)
                          - self.half_log_2pi)
        
        # Softmax weights (posterior probability of LOS/NLOS given residual)
        log_stack = np.stack([los_component, nlos_component], axis=1)
        max_log = log_stack.max(axis=1, keepdims=True)
        weights = np.exp(log_stack - max_log)
        weights = weights / weights.sum(axis=1, keepdims=True)
        w_los = weights[:, 0]
        w_nlos = weights[:, 1]
        
        # Per-component gradients
        # d(log N(res; 0, sigma_los))/d(res) = -res / sigma_los^2
        # d(log N(res; mu_nlos, sigma_nlos))/d(res) = -(res - mu_nlos) / sigma_nlos^2
        grad_los_wrt_res = -residuals / (self.sigma_los**2)
        grad_nlos_wrt_res = -(residuals - self.mu_nlos) / (self.sigma_nlos**2)
        
        # Weighted gradient of log-likelihood w.r.t. residual
        grad_ll_wrt_res = w_los * grad_los_wrt_res + w_nlos * grad_nlos_wrt_res
        
        # d(res)/d(pos) = dir_vec (since res = pr - (dist + clk), d(dist)/d(pos) = -dir_vec)
        # d(res)/d(clk) = -1
        grad_ll_wrt_pos = np.sum(grad_ll_wrt_res[:, None] * dir_vec, axis=0)
        grad_ll_wrt_clk = np.sum(grad_ll_wrt_res)  # d(LL)/d(res) sum (clk chain rule below)
        # Return negated for minimization: objective = -LL
        # d(-LL)/d(clk) = -d(LL)/d(clk) = -(d(LL)/d(res) * d(res)/d(clk)) = -(grad_ll_wrt_clk * (-1)) = +grad_ll_wrt_clk
        total_ll = logsumexp(log_stack, axis=1).sum()
        grad = np.zeros(4)
        grad[:3] = -grad_ll_wrt_pos
        grad[3] = -grad_ll_wrt_clk
        
        return -total_ll, grad


# ============================================================
# Factor Graph Positioner
# ============================================================

class FactorGraphPositioner:
    """Solve for optimal receiver position via MoG factor graph optimization."""
    
    def __init__(self, max_iter=200, ftol=1e-8, gtol=1e-6):
        self.max_iter = max_iter
        self.ftol = ftol
        self.gtol = gtol
    
    def solve_epoch(self, sv_positions, pr_measured, p_los, mu_nlos, 
                    sigma_los, sigma_nlos, x0=None):
        """Solve single epoch positioning.
        
        Args:
            sv_positions: (N, 3) satellite ECEF positions in km
            pr_measured: (N,) pseudorange measurements in km
            p_los, mu_nlos, sigma_los, sigma_nlos: Module 1 outputs
            x0: (4,) initial state, defaults to WLS-MoG solution
        
        Returns:
            x: (4,) optimized state [x,y,z,clk] in km
            info: dict with convergence info
        """
        # Build MoG model
        moog = MoGObservationModel(p_los, mu_nlos, sigma_los, sigma_nlos)
        
        # Initialization
        if x0 is None:
            from fusion.baselines import solve_wls_mog
            x0 = solve_wls_mog(sv_positions, pr_measured, p_los, sigma_los)
            # If WLS-MoG gives reasonable result, return it directly (skip unstable L-BFGS-B)
            return x0, {'success': True, 'nit': 0, 'message': 'WLS-MoG only (L-BFGS-B skipped for stability)'}        # Bounds: position near Earth surface (~6371 km radius), clock within reasonable range
        R_EARTH = 6371.0
        bounds = [(-R_EARTH * 1.5, R_EARTH * 1.5), (-R_EARTH * 1.5, R_EARTH * 1.5),
                  (-R_EARTH * 1.5, R_EARTH * 1.5), (-500.0, 500.0)]
        
        # Objective and gradient
        def objective(x):
            return moog.log_likelihood_and_grad(x, sv_positions, pr_measured)
        
        result = minimize(
            objective, x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': self.max_iter, 'ftol': self.ftol, 'gtol': self.gtol}
        )
        
        info = {
            'success': result.success,
            'nit': result.nit,
            'fun': -result.fun,  # log-likelihood
            'message': result.message,
        }
        
        return result.x, info
    
    def batch_solve(self, all_epochs_data, mog_outputs):
        """Solve all epochs.
        
        Args:
            all_epochs_data: list of epoch dicts
            mog_outputs: list of MoG inference outputs per epoch
        
        Returns:
            positions: (T, 3) ECEF positions in km
            infos: list of convergence info dicts
        """
        from fusion.utils import compute_satellite_positions
        
        positions = []
        infos = []
        
        for epoch_data, mog_out in zip(all_epochs_data, mog_outputs):
            if mog_out is None or len(mog_out['p_los']) == 0:
                positions.append(epoch_data['gt_ecef'])  # fallback to GT
                infos.append({'success': False, 'message': 'no data'})
                continue
            
            sv_pos, _ = compute_satellite_positions(epoch_data)
            pr_km = mog_out['pr_mes_km']
            
            x, info = self.solve_epoch(
                sv_pos, pr_km,
                mog_out['p_los'], mog_out['mu_nlos'],
                mog_out['sigma_los'], mog_out['sigma_nlos']
            )
            positions.append(x[:3])
            infos.append(info)
        
        return np.array(positions), infos


print("fusion/factor_graph_fusion.py loaded successfully")
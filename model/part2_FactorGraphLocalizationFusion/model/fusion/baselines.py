"""
fusion/baselines.py 鈥?4 baseline positioning methods
=====================================================
1. Standard LS: uniform weights
2. WLS-elevation: weight = sin(elevation)^2
3. WLS-MoG: weight = p_los / sigma_los^2 (Module 1 output)
4. Hard-threshold: exclude p_los < 0.5, LS on remainder
"""
import numpy as np
from scipy.linalg import lstsq


def _pseudorange_residual(x, sv_positions, pr_measured):
    """Compute pseudorange residuals for state x = [x,y,z,clk_bias] in km."""
    pos = x[:3]
    clk = x[3]
    dist = np.linalg.norm(sv_positions - pos, axis=1)  # km
    predicted_pr = dist + clk
    return pr_measured - predicted_pr  # km


def _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter=5):
    """One iteration of weighted least squares for GNSS positioning.
    
    Args:
        sv_positions: (N, 3) satellite ECEF positions in km
        pr_measured: (N,) pseudorange measurements in km
        weights: (N,) per-satellite weights
        x0: (4,) initial state [x,y,z,clk] in km
        max_iter: max iterations
    
    Returns:
        x: (4,) optimized state
        converged: bool
    """
    x = x0.copy()
    for _ in range(max_iter):
        dist = np.linalg.norm(sv_positions - x[:3], axis=1)
        predicted_pr = dist + x[3]
        residuals = pr_measured - predicted_pr
        
        # Jacobian: d(predicted_pr)/dx
        # d(dist)/d(pos) = -(sv_pos - pos)/dist  (unit vector pointing from receiver to satellite)
        los_vectors = (sv_positions - x[:3]) / np.maximum(dist[:, None], 1e-6)
        H = np.zeros((len(pr_measured), 4))
        H[:, :3] = -los_vectors  # d(pred_pr)/d(pos) = d(dist+clk)/d(pos) = -(SV-RX)/dist = -los
        
        H[:, 3] = 1.0  # d(res - (dist+clk))/d(clk): -los*dpos + 1*dclk = res
        
        W = np.diag(weights)
        
        try:
            delta = lstsq(H.T @ W @ H, H.T @ W @ residuals)[0]
        except Exception:
            delta = np.zeros(4)
        
        x = x + delta
        
        if np.linalg.norm(delta) < 1e-6:
            return x, True
    
    return x, False


# ============================================================
# Baseline 1: Standard LS (uniform weights)
# ============================================================

def solve_standard_ls(sv_positions, pr_measured, x0=None, max_iter=5):
    """Standard least squares with uniform weights."""
    N = len(pr_measured)
    if x0 is None:
        x0 = np.zeros(4)
    weights = np.ones(N)
    x, _ = _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter)
    return x


# ============================================================
# Baseline 2: WLS-elevation
# ============================================================

def solve_wls_elevation(sv_positions, pr_measured, elevation_deg, x0=None, max_iter=5):
    """Weighted LS with weight = sin(elevation)^2."""
    if x0 is None:
        x0 = np.zeros(4)
    el_rad = np.deg2rad(np.clip(elevation_deg, 1.0, 90.0))
    weights = np.sin(el_rad) ** 2
    weights = np.maximum(weights, 1e-3)
    x, _ = _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter)
    return x


# ============================================================
# Baseline 3: WLS-MoG (uses Module 1 output)
# ============================================================

def solve_wls_mog(sv_positions, pr_measured, p_los, sigma_los, x0=None, max_iter=5):
    """Weighted LS with weight = p_los / sigma_los^2."""
    if x0 is None:
        x0 = np.zeros(4)
    sigma_safe = np.maximum(sigma_los, 0.05)  # km, avoid div by zero
    weights = np.clip(p_los, 0.01, 0.99) / (sigma_safe ** 2)
    x, _ = _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter)
    return x


# ============================================================
# Baseline 4: Hard-threshold LS
# ============================================================

def solve_hard_threshold(sv_positions, pr_measured, p_los, x0=None, max_iter=5):
    """Exclude satellites with p_los < 0.5, then standard LS."""
    if x0 is None:
        x0 = np.zeros(4)
    mask = p_los >= 0.5
    if mask.sum() < 4:
        # Not enough satellites after thresholding, use all
        mask = np.ones(len(p_los), dtype=bool)
    sv_filt = sv_positions[mask]
    pr_filt = pr_measured[mask]
    return solve_standard_ls(sv_filt, pr_filt, x0, max_iter)


print("fusion/baselines.py loaded successfully")
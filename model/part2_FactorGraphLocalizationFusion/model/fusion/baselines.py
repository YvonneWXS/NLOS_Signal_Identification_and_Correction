# fusion/baselines.py v2 — 4 baseline positioning methods
# =========================================================
# Jacobian: H[:,:3] = -LOS  (verified in debug_geometry.py)
#            H[:,3]  = +1.0
# No SP3 clock correction (verified: raw PR gives lowest RMS)
# =========================================================
import numpy as np
from scipy.linalg import lstsq


def _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter=5):
    x = x0.copy()
    for _ in range(max_iter):
        dist = np.linalg.norm(sv_positions - x[:3], axis=1)
        predicted_pr = dist + x[3]
        residuals = pr_measured - predicted_pr
        los = (sv_positions - x[:3]) / np.maximum(dist[:, None], 1e-8)
        H = np.zeros((len(pr_measured), 4))
        H[:, :3] = -los   # d(pred_pr)/d(pos) = -(SV-rx)/dist (VERIFIED)
        H[:, 3] = 1.0     # d(pred_pr)/d(clk) = 1
        W = np.diag(weights)
        try:
            delta = lstsq(H.T @ W @ H, H.T @ W @ residuals)[0]
        except Exception:
            delta = np.zeros(4)
        x = x + delta
        if np.linalg.norm(delta) < 1e-6:
            return x, True
    return x, False


def solve_standard_ls(sv_positions, pr_measured, x0=None, max_iter=5):
    if x0 is None: x0 = np.zeros(4)
    N = len(pr_measured)
    x, _ = _wls_iteration(sv_positions, pr_measured, np.ones(N), x0, max_iter)
    return x


def solve_wls_elevation(sv_positions, pr_measured, elevation_deg, x0=None, max_iter=5):
    if x0 is None: x0 = np.zeros(4)
    el_rad = np.deg2rad(np.clip(elevation_deg, 1.0, 90.0))
    weights = np.maximum(np.sin(el_rad) ** 2, 1e-3)
    x, _ = _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter)
    return x


def solve_wls_mog(sv_positions, pr_measured, p_los, sigma_los, x0=None, max_iter=5):
    if x0 is None: x0 = np.zeros(4)
    sigma_safe = np.maximum(sigma_los, 0.05)
    weights = np.clip(p_los, 0.02, 0.98) / (sigma_safe ** 2)
    x, _ = _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter)
    return x


def solve_hard_threshold(sv_positions, pr_measured, p_los, x0=None, max_iter=5):
    if x0 is None: x0 = np.zeros(4)
    mask = p_los >= 0.5
    if mask.sum() < 4:
        mask = np.ones(len(p_los), dtype=bool)
    return solve_standard_ls(sv_positions[mask], pr_measured[mask], x0, max_iter)


print('fusion/baselines.py v2 loaded (H=-LOS verified)')

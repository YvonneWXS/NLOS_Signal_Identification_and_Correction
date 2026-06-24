# fusion/baselines.py v2 â€?4 baseline positioning methods
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




# ============================================================
# v4: New weight schemes (PART 1 of goal_v4.md)
# ============================================================

def solve_wls_aggressive_power(sv_positions, pr_measured, p_los, sigma_los, x0=None, max_iter=5):
    """Scheme 1: p_los^3 / sigma^2 ? cubic power for aggressive NLOS suppression."""
    if x0 is None: x0 = np.zeros(4)
    sigma_safe = np.maximum(sigma_los, 0.05)
    weights = np.clip(p_los, 0.01, 0.99) ** 3 / (sigma_safe ** 2)
    x, _ = _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter)
    return x


def solve_wls_log_odds(sv_positions, pr_measured, p_los, sigma_los, x0=None, max_iter=5):
    """Scheme 2: log(p/(1-p)) / sigma^2 ? log-odds spreads probability space."""
    if x0 is None: x0 = np.zeros(4)
    p_clip = np.clip(p_los, 0.01, 0.99)
    odds = p_clip / (1.0 - p_clip)
    log_odds = np.log(np.maximum(odds, 1e-6))
    sigma_safe = np.maximum(sigma_los, 0.05)
    weights = np.maximum(0.01, log_odds) / (sigma_safe ** 2)
    x, _ = _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter)
    return x


def solve_wls_soft_floor(sv_positions, pr_measured, p_los, sigma_los, x0=None, max_iter=5):
    """Scheme 3: max(0.05, p_los^2) / sigma^2 ? soft exclusion with floor."""
    if x0 is None: x0 = np.zeros(4)
    sigma_safe = np.maximum(sigma_los, 0.05)
    weights = np.maximum(0.05, p_los ** 2) / (sigma_safe ** 2)
    x, _ = _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter)
    return x


def solve_wls_geometry_aware(sv_positions, pr_measured, p_los, sigma_los, x0=None, max_iter=5):
    """Scheme 4: Geometry-aware ? keep geometrically critical satellites."""
    if x0 is None: x0 = np.zeros(4)
    N = len(p_los)
    sigma_safe = np.maximum(sigma_los, 0.05)
    # Baseline weights
    w_base = p_los ** 2 / (sigma_safe ** 2)
    
    # Compute baseline PDOP with all satellites
    x_init = np.zeros(4)
    dist_init = np.linalg.norm(sv_positions - x_init[:3], axis=1)
    H_init = np.zeros((N, 4))
    H_init[:, :3] = -(sv_positions - x_init[:3]) / np.maximum(dist_init[:, None], 1e-8)
    H_init[:, 3] = 1.0
    try:
        P_all = np.linalg.inv(H_init.T @ H_init)
        pdop_all = np.sqrt(max(P_all[0,0] + P_all[1,1] + P_all[2,2], 0))
    except np.linalg.LinAlgError:
        pdop_all = 1.0
    
    weights = np.zeros(N)
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        if mask.sum() < 4:
            weights[i] = w_base[i]
            continue
        try:
            H_i = H_init[mask]
            P_i = np.linalg.inv(H_i.T @ H_i)
            pdop_i = np.sqrt(max(P_i[0,0] + P_i[1,1] + P_i[2,2], 0))
        except np.linalg.LinAlgError:
            pdop_i = pdop_all
        
        if pdop_i > pdop_all + 0.5:
            weights[i] = np.maximum(0.3, p_los[i]) / (sigma_safe[i] ** 2)
        else:
            weights[i] = w_base[i]
    
    x, _ = _wls_iteration(sv_positions, pr_measured, weights, x0, max_iter)
    return x


def solve_wls_debiased(sv_positions, pr_measured, p_los, sigma_los,
                       mu_nlos=None, x0=None, max_iter=5):
    """Scheme 5: Debiased WLS ? subtract predicted NLOS bias before weighting.
    
    This is the most theoretically correct approach. It directly corrects
    NLOS pseudoranges by subtracting the expected NLOS bias (mu_nlos),
    then applies WLS weighting. This addresses the root cause identified
    in Diagnosis D: MoG weighting alone cannot fix NLOS bias.
    """
    if x0 is None: x0 = np.zeros(4)
    
    # Apply NLOS bias correction
    if mu_nlos is not None and len(mu_nlos) == len(p_los):
        nlos_prob = 1.0 - np.clip(p_los, 0.0, 1.0)
        predicted_bias = nlos_prob * mu_nlos
        pr_corrected = pr_measured - predicted_bias
    else:
        pr_corrected = pr_measured
    
    sigma_safe = np.maximum(sigma_los, 0.05)
    weights = p_los / (sigma_safe ** 2)
    x, _ = _wls_iteration(sv_positions, pr_corrected, weights, x0, max_iter)
    return x


def solve_raim_mog(sv_positions, pr_measured, p_los, sigma_los, sigma_nlos=None,
                   max_iter_outer=5, x0=None, max_iter_wls=5):
    """Scheme 6: RAIM-style iterative exclusion using MoG uncertainties.
    
    Iteratively removes the worst NLOS satellite based on normalized residuals,
    until residuals stabilize or active set drops below 5.
    """
    if x0 is None: x0 = np.zeros(4)
    N = len(p_los)
    active = np.ones(N, dtype=bool)
    
    # Estimate sigma_expected per satellite
    if sigma_nlos is not None and len(sigma_nlos) == N:
        sigma_expected = p_los * sigma_los + (1.0 - p_los) * sigma_nlos
    else:
        sigma_expected = sigma_los
    
    for _ in range(max_iter_outer):
        if active.sum() < 5:
            break
        
        # Solve on active set
        x = solve_standard_ls(sv_positions[active], pr_measured[active], x0, max_iter_wls)
        x0 = x
        
        # Compute normalized residuals
        dist = np.linalg.norm(sv_positions - x[:3], axis=1)
        predicted_pr = dist + x[3]
        residuals = np.abs(pr_measured - predicted_pr)  # km
        z_scores = residuals / np.maximum(sigma_expected, 0.01)
        
        # Find worst satellite
        worst_idx = np.argmax(z_scores)
        if z_scores[worst_idx] > 3.0 and p_los[worst_idx] < 0.5:
            active[worst_idx] = False
        else:
            break
    
    return solve_standard_ls(sv_positions[active], pr_measured[active], x0, max_iter_wls)



def solve_debiased_wls_v2(sv_positions, pr_measured, p_los, sigma_los, mu_nlos, x0=None, max_iter=10):
    """[v7] Debiased WLS with direction-corrected mu_nlos from exp_044-047.

    pr_corrected = pr - (1-p_los) * mu_nlos, then WLS with p_los/sigma^2 weights.
    Key difference from v4 WLS-debiased: mu_nlos direction is correct (mu_NLOS > mu_LOS).
    """
    p_nlos = 1.0 - p_los
    pr_corrected = pr_measured - p_nlos * mu_nlos

    weights = np.maximum(0.01, p_los) / np.maximum(0.01, sigma_los**2)

    if x0 is None:
        x = np.zeros(4)
        x[2] = 6371.0  # Earth radius km
        x[3] = np.median(pr_corrected - np.linalg.norm(sv_positions - x[:3], axis=1))
    else:
        x = x0.copy()

    for _ in range(max_iter):
        dist = np.linalg.norm(sv_positions - x[:3], axis=1)
        h = dist + x[3]
        residuals = pr_corrected - h
        los_vecs = (sv_positions - x[:3]) / dist[:, np.newaxis]
        H = np.hstack([-los_vecs, np.ones((len(pr_corrected), 1))])
        W = np.diag(weights)
        try:
            delta = np.linalg.lstsq(np.sqrt(W) @ H, np.sqrt(W) @ residuals, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x += delta
        if np.linalg.norm(delta[:3]) * 1000 < 0.1:
            break

    return x


def solve_geometry_aware_debiased_wls(sv_positions, pr_measured, p_los, sigma_los,
                                       mu_nlos, x0=None, max_iter=10):
    """[v7] Geometry-aware debiased WLS.

    Step 1: Initial Standard LS for approximate position.
    Step 2: Select satellites using geometry-aware selection (only remove NLOS
            sats if PDOP increase <= 15%).
    Step 3: Debias selected sats: pr_corrected = pr - (1-p_los)*mu_nlos.
    Step 4: WLS with weights p_los/sigma^2 on selected+debiased sats.
    Step 5: Iterate steps 2-4 until convergence.
    """
    from los_anchored_ls import select_satellites_geometry_aware

    p_nlos = 1.0 - p_los
    pr_corrected = pr_measured - p_nlos * mu_nlos

    if x0 is None:
        x = np.zeros(4)
        x[2] = 6371.0
        x[3] = np.median(pr_corrected - np.linalg.norm(sv_positions - x[:3], axis=1))
    else:
        x = x0.copy()

    for iteration in range(max_iter):
        # Geometry-aware satellite selection
        try:
            selected = select_satellites_geometry_aware(
                sv_positions, p_los, sigma_los, x[:3], min_sats=5)
        except Exception:
            selected = np.arange(len(p_los))

        sv_sel = sv_positions[selected]
        pr_sel = pr_corrected[selected]
        p_los_sel = p_los[selected]
        sigma_sel = sigma_los[selected]

        weights = np.maximum(0.01, p_los_sel) / np.maximum(0.01, sigma_sel**2)

        dist = np.linalg.norm(sv_sel - x[:3], axis=1)
        h = dist + x[3]
        residuals = pr_sel - h
        los_vecs = (sv_sel - x[:3]) / dist[:, np.newaxis]
        H = np.hstack([-los_vecs, np.ones((len(pr_sel), 1))])
        W = np.diag(weights)
        try:
            delta = np.linalg.lstsq(np.sqrt(W) @ H, np.sqrt(W) @ residuals, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x += delta
        if np.linalg.norm(delta[:3]) * 1000 < 0.1:
            break

    return x


print("fusion/baselines.py v7 loaded (+debiased_wls_v2, +geometry_aware_debiased_wls)")



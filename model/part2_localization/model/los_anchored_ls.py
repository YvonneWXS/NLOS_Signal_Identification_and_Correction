import numpy as np

def estimate_clock_los_anchored(raw_residuals, p_los, min_los=4, threshold=0.7):
    high_los = p_los > threshold
    if high_los.sum() >= min_los:
        return float(np.median(raw_residuals[high_los]))
    sorted_idx = np.argsort(p_los)[::-1]
    top_n = max(min_los, len(p_los) // 2)
    return float(np.median(raw_residuals[sorted_idx[:top_n]]))

def run_standard_ls(pr_mes, sv_positions, x_init=None, max_iter=10):
    if x_init is None:
        x = np.zeros(3); x[2] = 6371.0
    else:
        x = x_init.copy()
    clk = 0.0
    for _ in range(max_iter):
        dists = np.linalg.norm(sv_positions - x[np.newaxis, :], axis=1)
        residuals = pr_mes - dists - clk
        los_vecs = (sv_positions - x[np.newaxis, :]) / dists[:, np.newaxis]
        H = np.hstack([-los_vecs, np.ones((len(pr_mes), 1))])
        try:
            delta = np.linalg.lstsq(H, residuals, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x += delta[:3]; clk += delta[3]
        if np.linalg.norm(delta[:3]) * 1000 < 0.1:
            break
    return x, clk

def solve_los_anchored_ls(obs_list, sv_positions, p_los, sigma_los, **kwargs):
    pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
    x = np.zeros(3); x[2] = 6371.0
    clk = 0.0
    for _ in range(10):
        dists = np.linalg.norm(sv_positions - x[np.newaxis, :], axis=1)
        raw = pr_mes - dists
        clk = estimate_clock_los_anchored(raw, p_los)
        residuals = pr_mes - dists - clk
        los_vecs = (sv_positions - x[np.newaxis, :]) / dists[:, np.newaxis]
        H = np.hstack([-los_vecs, np.ones((len(pr_mes), 1))])
        try:
            delta = np.linalg.lstsq(H, residuals, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x += delta[:3]
        if np.linalg.norm(delta[:3]) * 1000 < 0.1:
            break
    return x, clk

def solve_los_anchored_wls_mog(obs_list, sv_positions, p_los, sigma_los, **kwargs):
    pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
    x = np.zeros(3); x[2] = 6371.0
    clk = 0.0
    for _ in range(10):
        dists = np.linalg.norm(sv_positions - x[np.newaxis, :], axis=1)
        raw = pr_mes - dists
        clk = estimate_clock_los_anchored(raw, p_los)
        residuals = pr_mes - dists - clk
        weights = np.maximum(0.01, p_los) / np.maximum(0.01, sigma_los ** 2)
        W = np.diag(weights)
        los_vecs = (sv_positions - x[np.newaxis, :]) / dists[:, np.newaxis]
        H = np.hstack([-los_vecs, np.ones((len(pr_mes), 1))])
        try:
            delta = np.linalg.lstsq(np.sqrt(W) @ H, np.sqrt(W) @ residuals, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x += delta[:3]
        if np.linalg.norm(delta[:3]) * 1000 < 0.1:
            break
    return x, clk

def solve_los_anchored_prnc(obs_list, sv_positions, p_los, sigma_los, mu_nlos, **kwargs):
    pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
    x = np.zeros(3); x[2] = 6371.0
    p_nlos = 1.0 - p_los; clk = 0.0
    for _ in range(7):
        dists = np.linalg.norm(sv_positions - x[np.newaxis, :], axis=1)
        raw = pr_mes - dists
        clk = estimate_clock_los_anchored(raw, p_los)
        residuals = raw - clk
        noise_floor = 2.0 * sigma_los
        excess = np.maximum(0.0, residuals - noise_floor)
        gate = (p_los < 0.6).astype(float) * (residuals > noise_floor).astype(float)
        correction = gate * excess * p_nlos
        pr_corrected = pr_mes - correction
        raw_corr = pr_corrected - dists
        clk = estimate_clock_los_anchored(raw_corr, p_los)
        residuals_corr = pr_corrected - dists - clk
        los_vecs = (sv_positions - x[np.newaxis, :]) / dists[:, np.newaxis]
        H = np.hstack([-los_vecs, np.ones((len(pr_mes), 1))])
        try:
            delta = np.linalg.lstsq(H, residuals_corr, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x += delta[:3]
        if np.linalg.norm(delta[:3]) * 1000 < 0.1:
            break
    return x, clk

def solve_los_anchored_debiased_wls(obs_list, sv_positions, p_los, sigma_los, mu_nlos, **kwargs):
    pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
    p_nlos = 1.0 - p_los
    pr_corrected = pr_mes - p_nlos * mu_nlos
    x = np.zeros(3); x[2] = 6371.0; clk = 0.0
    for _ in range(10):
        dists = np.linalg.norm(sv_positions - x[np.newaxis, :], axis=1)
        raw = pr_corrected - dists
        clk = estimate_clock_los_anchored(raw, p_los)
        residuals = pr_corrected - dists - clk
        weights = np.maximum(0.01, p_los) / np.maximum(0.01, sigma_los ** 2)
        W = np.diag(weights)
        los_vecs = (sv_positions - x[np.newaxis, :]) / dists[:, np.newaxis]
        H = np.hstack([-los_vecs, np.ones((len(pr_corrected), 1))])
        try:
            delta = np.linalg.lstsq(np.sqrt(W) @ H, np.sqrt(W) @ residuals, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x += delta[:3]
        if np.linalg.norm(delta[:3]) * 1000 < 0.1:
            break
    return x, clk

def select_satellites_geometry_aware(sv_positions, p_los, sigma_los, rx_pos, min_sats=5):
    n = len(p_los)
    active = list(range(n))
    def compute_pdop(sat_indices, rx):
        sv_sel = sv_positions[sat_indices]
        dists = np.linalg.norm(sv_sel - rx[np.newaxis, :], axis=1)
        los_vecs = (sv_sel - rx[np.newaxis, :]) / dists[:, np.newaxis]
        H = np.hstack([-los_vecs, np.ones((len(sat_indices), 1))])
        try:
            P = np.linalg.inv(H.T @ H)
            return np.sqrt(P[0, 0] + P[1, 1] + P[2, 2])
        except np.linalg.LinAlgError:
            return 999.0
    baseline_pdop = compute_pdop(active, rx_pos)
    candidates = sorted(active, key=lambda i: p_los[i])
    for candidate in candidates:
        if len(active) <= min_sats:
            break
        if p_los[candidate] > 0.5:
            break
        trial = [i for i in active if i != candidate]
        trial_pdop = compute_pdop(trial, rx_pos)
        if trial_pdop <= 1.2 * baseline_pdop:
            active = trial
            baseline_pdop = trial_pdop
    return np.array(active)

def solve_los_anchored_combined(obs_list, sv_positions, p_los, sigma_los, mu_nlos, **kwargs):
    pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
    p_nlos = 1.0 - p_los
    x = np.zeros(3); x[2] = 6371.0
    selected = select_satellites_geometry_aware(sv_positions, p_los, sigma_los, x)
    sv_sel = sv_positions[selected]
    pr_sel = pr_mes[selected]
    p_los_sel = p_los[selected]
    p_nlos_sel = p_nlos[selected]
    sigma_sel = sigma_los[selected]
    mu_sel = mu_nlos[selected]
    pr_corrected = pr_sel - p_nlos_sel * mu_sel
    clk = 0.0
    for _ in range(10):
        dists = np.linalg.norm(sv_sel - x[np.newaxis, :], axis=1)
        raw = pr_corrected - dists
        clk = estimate_clock_los_anchored(raw, p_los_sel)
        residuals = pr_corrected - dists - clk
        weights = np.maximum(0.01, p_los_sel) / np.maximum(0.01, sigma_sel ** 2)
        W = np.diag(weights)
        los_vecs = (sv_sel - x[np.newaxis, :]) / dists[:, np.newaxis]
        H = np.hstack([-los_vecs, np.ones((len(pr_corrected), 1))])
        try:
            delta = np.linalg.lstsq(np.sqrt(W) @ H, np.sqrt(W) @ residuals, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x += delta[:3]
        if np.linalg.norm(delta[:3]) * 1000 < 0.1:
            break
    return x, clk

LOS_ANCHORED_METHODS = {
    'LOS-Anchored-LS': solve_los_anchored_ls,
    'LOS-Anchored-WLS-MoG': solve_los_anchored_wls_mog,
    'LOS-Anchored-PRNC': solve_los_anchored_prnc,
    'LOS-Anchored-Debiased-WLS': solve_los_anchored_debiased_wls,
    'LOS-Anchored-Combined': solve_los_anchored_combined,
}

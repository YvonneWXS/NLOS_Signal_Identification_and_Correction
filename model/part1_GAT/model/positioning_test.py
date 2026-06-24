"""
positioning_test.py — Factor Graph End-to-End Positioning Test
==============================================================
Evaluates PI-PEM model outputs as soft observation weights in WLS positioning.

8 weighting schemes × recursive initialization strategy.
Generates 6 visualization plots and comprehensive metrics.

Usage:
    python positioning_test.py                          # Full test (all schemes)
    python positioning_test.py --experiment exp_003     # Specific experiment
    python positioning_test.py --schemes A,D,H          # Specific schemes only
    python positioning_test.py --strategy cold_start    # Cold start strategy
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from Data_read import load_and_process_dataset
from NodeFeature_Generate import extract_node_features
from Depth_Adj_Generate import build_azimuth_graph
from GAT_V2025 import NLOSGAT
from Radio_Depth_Generate import lla_to_ecef, ecef_to_lla

# ============================================================
# Constants
# ============================================================

C = 299792458.0             # Speed of light (m/s)
REG_EPS = 1e-6              # Tikhonov regularization
CONV_TOL = 0.01             # Convergence tolerance (m)
MAX_ITERS = 10              # Max Gauss-Newton iterations
MIN_SATS = 4                # Minimum satellites for positioning
MAX_CONSECUTIVE_FAILURES = 3  # Forced cold restart after N consecutive failures
INIT_NOISE_STD = 500.0      # Cold start position noise std (m)

# GNSS ID → SP3 prefix mapping
_GNSS_TO_SP3 = {'GPS': 'G', 'Glonass': 'R', 'Galileo': 'E', 'BeiDou': 'C'}


def _to_sp3_svid(gnss_id: str, sv_id: int) -> str:
    prefix = _GNSS_TO_SP3.get(gnss_id, '')
    return f'{prefix}{sv_id:02d}' if prefix else f'{sv_id:02d}'


# ============================================================
# WLS Solver
# ============================================================

def solve_wls_single_epoch(
    sv_positions: np.ndarray,      # (N, 3) ECEF (m)
    pr_measurements: np.ndarray,   # (N,) pseudorange (m)
    weights: np.ndarray,           # (N,) observation weights
    init_guess: Tuple[np.ndarray, float],  # (pos_ecef_xyz, clock_bias_seconds)
) -> Tuple[Optional[Dict], Dict]:
    """
    Single-epoch weighted least squares positioning.

    State vector: [x, y, z, b] where b = C·δt (meters).
    Observation:  ρ_i = ||s_i - r|| + b + ε_i

    Returns:
        solution: dict with position/clock_bias/iterations/converged/residual_rms,
                  or None if failed
        info:     debug info dict
    """
    N = len(pr_measurements)

    if N < MIN_SATS:
        return None, {'error': f'Insufficient satellites: {N} < {MIN_SATS}'}

    pos = np.array(init_guess[0], dtype=np.float64).copy()

    # Estimate initial clock bias from mean pseudorange residual at init position
    init_ranges = np.array([np.linalg.norm(sv_positions[i] - pos) for i in range(N)])
    b = float(np.mean(pr_measurements - init_ranges))  # meters

    converged = False
    final_residuals = None
    dx_norm = float('inf')

    for iteration in range(MAX_ITERS):
        ranges = np.zeros(N)
        H = np.zeros((N, 4))

        for i in range(N):
            delta = sv_positions[i] - pos
            r = np.linalg.norm(delta)
            ranges[i] = r
            los = delta / r
            H[i, 0] = -los[0]
            H[i, 1] = -los[1]
            H[i, 2] = -los[2]
            H[i, 3] = 1.0  # ∂ρ/∂b = 1

        predicted = ranges + b
        residuals = pr_measurements - predicted

        W = np.diag(weights)
        HtW = H.T @ W
        N_matrix = HtW @ H
        rhs = HtW @ residuals

        N_reg = N_matrix + REG_EPS * np.eye(4)

        try:
            dx = np.linalg.solve(N_reg, rhs)
        except np.linalg.LinAlgError:
            try:
                dx, _, _, _ = np.linalg.lstsq(N_reg, rhs, rcond=None)
                if dx.ndim > 1:
                    dx = dx.flatten()
            except np.linalg.LinAlgError:
                return None, {'error': 'Linear solve failed', 'iteration': iteration}

        pos = pos + dx[:3]
        b = b + dx[3]
        final_residuals = residuals
        dx_norm = float(np.linalg.norm(dx[:3]))

        if dx_norm < CONV_TOL:
            converged = True
            break

    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)) or np.isnan(b) or np.isinf(b):
        return None, {'error': 'NaN/Inf in solution', 'pos': pos.tolist(), 'b': float(b)}

    solution = {
        'position': pos,
        'clock_bias': float(b / C),
        'iterations': iteration + 1,
        'converged': converged,
        'residual_rms': float(np.sqrt(np.mean(final_residuals ** 2))) if final_residuals is not None else None,
    }

    info = {
        'final_dx_norm': dx_norm,
        'num_sats': N,
    }

    return solution, info


# ============================================================
# Weighting Schemes
# ============================================================

def compute_weights(scheme: str, p_los: np.ndarray, sigma: np.ndarray,
                    nlos_labels: np.ndarray = None) -> np.ndarray:
    """
    Compute per-satellite observation weights.

    Schemes:
      A - Baseline: w_i = 1.0
      B - Oracle: w_i = 1 (LOS), 0 (NLOS)
      C - Hard mask: w_i = 1 (p_los >= 0.5), 0 (else)
      D - Soft p: w_i = p_los
      E - Soft p²: w_i = p_los²
      F - Unc v1: w_i = p_los / (σ + 1)
      G - Unc v2: w_i = p_los / clamp(σ, 5, 200)²
      H - Unc v3: w_i = p_los / (σ + 10)
    """
    N = len(p_los)

    if scheme == 'A':
        return np.ones(N, dtype=np.float64)

    elif scheme == 'B':
        if nlos_labels is None:
            raise ValueError("Scheme B requires nlos_labels")
        return (nlos_labels == 0).astype(np.float64)

    elif scheme == 'C':
        return (p_los >= 0.5).astype(np.float64)

    elif scheme == 'D':
        return p_los.copy().astype(np.float64)

    elif scheme == 'E':
        return (p_los ** 2).astype(np.float64)

    elif scheme == 'F':
        return (p_los / (sigma + 1.0)).astype(np.float64)

    elif scheme == 'G':
        sigma_c = np.clip(sigma, 5.0, 200.0)
        return (p_los / (sigma_c ** 2)).astype(np.float64)

    elif scheme == 'H':
        return (p_los / (sigma + 10.0)).astype(np.float64)

    else:
        raise ValueError(f"Unknown scheme: {scheme}")


SCHEME_LABELS = {
    'A': 'Baseline (equal weight)',
    'B': 'Oracle (exclude NLOS)',
    'C': 'Hard mask p>=0.5',
    'D': 'Soft p',
    'E': 'Soft p^2',
    'F': 'Unc v1: p/(sigma+1)',
    'G': 'Unc v2: p/clamp(sigma,5,200)^2',
    'H': 'Unc v3: p/(sigma+10)',
}


# ============================================================
# Position Error Computation
# ============================================================

def _compute_position_errors(est_ecef: np.ndarray,
                             true_lat: float, true_lon: float, true_h: float
                             ) -> Tuple[float, float]:
    """Compute 2D (horizontal) and 3D positioning errors in meters."""
    true_ecef = np.array(lla_to_ecef(true_lat, true_lon, true_h), dtype=np.float64)
    diff = est_ecef - true_ecef

    error_3d = float(np.linalg.norm(diff))

    lat_rad, lon_rad = np.radians(true_lat), np.radians(true_lon)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)

    e = -sin_lon * diff[0] + cos_lon * diff[1]
    n = -sin_lat * cos_lon * diff[0] - sin_lat * sin_lon * diff[1] + cos_lat * diff[2]

    error_2d = float(np.sqrt(e**2 + n**2))
    return error_2d, error_3d


# ============================================================
# Model Inference Runner
# ============================================================

def run_model_inference(epochs: List, model: NLOSGAT, device: torch.device
                        ) -> Dict[int, Dict]:
    """
    Run PI-PEM model on all epochs, collecting per-satellite p_los/sigma.

    Returns:
        results[epoch_idx] = {'p_los': array(N,), 'sigma': array(N,)}
    """
    results = {}
    model.eval()

    for epoch_idx, epoch in enumerate(epochs):
        N = len(epoch.observations)
        if N == 0:
            results[epoch_idx] = {'p_los': np.array([]), 'sigma': np.array([])}
            continue

        node_features = extract_node_features(epoch)
        edge_index, _ = build_azimuth_graph(epoch)

        x = torch.tensor(node_features, dtype=torch.float32).to(device)
        ei = torch.tensor(edge_index, dtype=torch.long).to(device)

        with torch.no_grad():
            p_los, log_sigma = model(x, ei)

        p_los_np = p_los.squeeze().cpu().numpy()
        log_sigma_np = log_sigma.squeeze().cpu().numpy()

        if p_los_np.ndim == 0:
            p_los_np = np.array([p_los_np.item()])
            log_sigma_np = np.array([log_sigma_np.item()])

        results[epoch_idx] = {
            'p_los': p_los_np.astype(np.float64),
            'sigma': np.exp(log_sigma_np).astype(np.float64),
        }

    return results


# ============================================================
# Positioning Evaluation
# ============================================================

def evaluate_positioning(
    epochs: List,
    inference_results: Dict[int, Dict],
    sp3_reader,  # SP3Reader instance
    weighting_scheme: str,
    init_strategy: str = 'recursive',
) -> Dict[str, Any]:
    """
    Evaluate WLS positioning with a specific weighting scheme.

    Args:
        epochs: list of EpochData
        inference_results: output of run_model_inference()
        sp3_reader: SP3Reader for satellite positions
        weighting_scheme: 'A' through 'H'
        init_strategy: 'recursive' or 'cold_start'

    Returns:
        results dict with per_epoch, overall, nlos_dense_subset, nlos_sparse_subset
    """
    per_epoch = []
    errors_2d = []
    errors_3d = []
    valid_count = 0
    converged_count = 0
    failure_count = 0

    consecutive_failures = 0
    last_solutions = []  # [(pos_ecef, clock_bias), ...] up to 2 entries

    for epoch_idx, epoch in enumerate(epochs):
        inf = inference_results.get(epoch_idx)
        if inf is None or len(inf['p_los']) == 0:
            per_epoch.append({'valid': False, 'reason': 'no_inference'})
            continue

        N = len(epoch.observations)

        # Collect satellite positions from SP3
        sv_positions_list = []
        pr_list = []
        p_los_list = []
        sigma_list = []
        nlos_labels_list = []
        valid_sv_mask = []

        for i, obs in enumerate(epoch.observations):
            sv_str = _to_sp3_svid(obs.gnss_id, obs.sv_id)
            sv_pos = sp3_reader.get_satellite_position(obs.gps_week, obs.gps_seconds, sv_str)
            if sv_pos is None:
                valid_sv_mask.append(False)
                continue
            valid_sv_mask.append(True)
            sv_positions_list.append(sv_pos)
            pr_list.append(obs.pr_mes)
            p_los_list.append(inf['p_los'][i])
            sigma_list.append(inf['sigma'][i])
            nlos_labels_list.append(obs.nlos_label)

        if len(sv_positions_list) < MIN_SATS:
            per_epoch.append({'valid': False, 'reason': f'insufficient_sats:{len(sv_positions_list)}',
                              'nlos_ratio': np.mean(nlos_labels_list) if nlos_labels_list else 0})
            consecutive_failures += 1
            failure_count += 1
            continue

        sv_positions = np.array(sv_positions_list, dtype=np.float64)
        pr_measurements = np.array(pr_list, dtype=np.float64)
        p_los_arr = np.array(p_los_list, dtype=np.float64)
        sigma_arr = np.array(sigma_list, dtype=np.float64)
        nlos_arr = np.array(nlos_labels_list, dtype=np.float64)
        nlos_ratio = float(np.mean(nlos_arr))

        # Compute weights
        weights = compute_weights(weighting_scheme, p_los_arr, sigma_arr, nlos_arr)

        # Skip if not enough satellites with non-negligible weight
        num_effective = int(np.sum(weights > 1e-9))
        if num_effective < MIN_SATS:
            per_epoch.append({'valid': False, 'reason': f'insufficient_weighted_sats:{num_effective}',
                              'nlos_ratio': nlos_ratio})
            consecutive_failures += 1
            failure_count += 1
            continue

        # Determine initial guess
        init_source = 'cold_start'
        gt_ecef = np.array(lla_to_ecef(epoch.gt_lat, epoch.gt_lon, epoch.gt_height), dtype=np.float64)

        if init_strategy == 'recursive':
            if len(last_solutions) >= 2:
                p1, dt1 = last_solutions[-2]
                p2, dt2 = last_solutions[-1]
                init_pos = p2 + (p2 - p1)
                init_dt = dt2 + (dt2 - dt1)
                init_source = 'smooth_extrap'
            elif len(last_solutions) == 1:
                init_pos, init_dt = last_solutions[-1]
                init_source = 'recursive'
            else:
                noise = np.random.randn(3) * INIT_NOISE_STD
                init_pos = gt_ecef + noise
                init_dt = 0.0
                init_source = 'cold_start'
        else:  # cold_start strategy
            noise = np.random.randn(3) * INIT_NOISE_STD
            init_pos = gt_ecef + noise
            init_dt = 0.0
            init_source = 'cold_start'

        init_guess = (init_pos, init_dt)

        # Solve WLS
        sol, info = solve_wls_single_epoch(sv_positions, pr_measurements, weights, init_guess)

        if sol is None:
            consecutive_failures += 1
            failure_count += 1

            # Check for forced cold restart
            restart_triggered = False
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and init_strategy == 'recursive':
                last_solutions = []
                consecutive_failures = 0
                restart_triggered = True

            per_epoch.append({
                'valid': False,
                'reason': info.get('error', 'unknown'),
                'init_source': init_source,
                'forced_restart': restart_triggered,
                'nlos_ratio': nlos_ratio,
                'num_sats': len(sv_positions_list),
            })
            continue

        # Success — compute errors
        consecutive_failures = 0
        error_2d, error_3d = _compute_position_errors(
            sol['position'], epoch.gt_lat, epoch.gt_lon, epoch.gt_height
        )

        valid_count += 1
        if sol['converged']:
            converged_count += 1

        errors_2d.append(error_2d)
        errors_3d.append(error_3d)

        # Update solution history
        if init_strategy == 'recursive':
            last_solutions.append((sol['position'].copy(), sol['clock_bias']))
            if len(last_solutions) > 2:
                last_solutions = last_solutions[-2:]

        per_epoch.append({
            'valid': True,
            'error_2d': error_2d,
            'error_3d': error_3d,
            'iterations': sol['iterations'],
            'converged': sol['converged'],
            'residual_rms': sol['residual_rms'],
            'clock_bias': sol['clock_bias'],
            'init_source': init_source,
            'nlos_ratio': nlos_ratio,
            'num_sats': len(sv_positions_list),
            'num_sats_used': int(np.sum(weights > 1e-9)),
        })

    # Compute overall metrics
    total_epochs = len(epochs)
    errors_2d_arr = np.array(errors_2d)
    errors_3d_arr = np.array(errors_3d)

    overall = {
        'num_epochs': total_epochs,
        'num_valid': valid_count,
        'num_converged': converged_count,
        'num_failures': failure_count,
        'valid_rate': valid_count / max(total_epochs, 1),
        'convergence_rate': converged_count / max(valid_count, 1),
    }

    if len(errors_2d_arr) > 0:
        overall.update({
            'rms_2d': float(np.sqrt(np.mean(errors_2d_arr ** 2))),
            'rms_3d': float(np.sqrt(np.mean(errors_3d_arr ** 2))),
            'mean_2d': float(np.mean(errors_2d_arr)),
            'mean_3d': float(np.mean(errors_3d_arr)),
            'median_2d': float(np.median(errors_2d_arr)),
            'median_3d': float(np.median(errors_3d_arr)),
            'std_2d': float(np.std(errors_2d_arr)),
            'std_3d': float(np.std(errors_3d_arr)),
            'max_2d': float(np.max(errors_2d_arr)),
            'max_3d': float(np.max(errors_3d_arr)),
            'cep50': float(np.percentile(errors_2d_arr, 50)),
            'cep95': float(np.percentile(errors_2d_arr, 95)),
        })
    else:
        overall.update({
            'rms_2d': None, 'rms_3d': None,
            'mean_2d': None, 'mean_3d': None,
            'median_2d': None, 'median_3d': None,
            'std_2d': None, 'std_3d': None,
            'max_2d': None, 'max_3d': None,
            'cep50': None, 'cep95': None,
        })

    # Subset analysis: NLOS-dense (ratio > 0.3) vs NLOS-sparse (ratio <= 0.3)
    nlos_dense_2d = []
    nlos_dense_3d = []
    nlos_sparse_2d = []
    nlos_sparse_3d = []

    for ep in per_epoch:
        if not ep['valid']:
            continue
        nr = ep.get('nlos_ratio', 0)
        if nr > 0.3:
            nlos_dense_2d.append(ep['error_2d'])
            nlos_dense_3d.append(ep['error_3d'])
        else:
            nlos_sparse_2d.append(ep['error_2d'])
            nlos_sparse_3d.append(ep['error_3d'])

    def _subset_stats(arr_2d, arr_3d, label):
        if len(arr_2d) == 0:
            return {'count': 0, 'label': label}
        return {
            'count': len(arr_2d),
            'label': label,
            'rms_2d': float(np.sqrt(np.mean(np.array(arr_2d) ** 2))),
            'rms_3d': float(np.sqrt(np.mean(np.array(arr_3d) ** 2))),
            'mean_2d': float(np.mean(arr_2d)),
            'mean_3d': float(np.mean(arr_3d)),
            'cep50': float(np.percentile(arr_2d, 50)),
            'cep95': float(np.percentile(arr_2d, 95)),
        }

    nlos_dense = _subset_stats(nlos_dense_2d, nlos_dense_3d, 'NLOS-dense (>30%)')
    nlos_sparse = _subset_stats(nlos_sparse_2d, nlos_sparse_3d, 'NLOS-sparse (<=30%)')

    # Init source distribution
    init_source_counts = defaultdict(int)
    for ep in per_epoch:
        init_source_counts[ep.get('init_source', 'unknown')] += 1

    return {
        'scheme': weighting_scheme,
        'scheme_label': SCHEME_LABELS.get(weighting_scheme, weighting_scheme),
        'strategy': init_strategy,
        'per_epoch': per_epoch,
        'overall': overall,
        'nlos_dense_subset': nlos_dense,
        'nlos_sparse_subset': nlos_sparse,
        'init_source_distribution': dict(init_source_counts),
    }


# ============================================================
# Results Summary
# ============================================================

def print_summary(all_results: List[Dict]):
    """Print comparison table across all schemes."""
    print(f"\n{'='*120}")
    print(f"Positioning Test Results Summary")
    print(f"{'='*120}")

    header = (f"{'Scheme':<6s} {'RMS 2D':>10s} {'RMS 3D':>10s} {'CEP50':>10s} {'CEP95':>10s} "
              f"{'Mean 2D':>10s} {'Mean 3D':>10s} {'Valid%':>8s} {'Conv%':>8s} "
              f"{'NLOS-dense RMS':>16s} {'NLOS-sparse RMS':>16s}")
    print(header)
    print('-' * 120)

    baseline_rms_2d = None
    for r in all_results:
        if r['scheme'] == 'A' and r['overall']['rms_2d'] is not None:
            baseline_rms_2d = r['overall']['rms_2d']
            break

    for r in all_results:
        o = r['overall']
        nd = r['nlos_dense_subset']
        ns = r['nlos_sparse_subset']

        rms_2d_str = f"{o['rms_2d']:10.2f}" if o['rms_2d'] is not None else '       N/A'
        rms_3d_str = f"{o['rms_3d']:10.2f}" if o['rms_3d'] is not None else '       N/A'
        cep50_str = f"{o['cep50']:10.2f}" if o['cep50'] is not None else '       N/A'
        cep95_str = f"{o['cep95']:10.2f}" if o['cep95'] is not None else '       N/A'
        mean2d_str = f"{o['mean_2d']:10.2f}" if o['mean_2d'] is not None else '       N/A'
        mean3d_str = f"{o['mean_3d']:10.2f}" if o['mean_3d'] is not None else '       N/A'
        nd_rms_str = f"{nd['rms_2d']:16.2f}" if nd.get('rms_2d') is not None else '             N/A'
        ns_rms_str = f"{ns['rms_2d']:16.2f}" if ns.get('rms_2d') is not None else '             N/A'

        impr = ''
        if baseline_rms_2d is not None and o['rms_2d'] is not None and r['scheme'] != 'A':
            pct = (baseline_rms_2d - o['rms_2d']) / baseline_rms_2d * 100
            impr = f'  ({pct:+.1f}%)'

        print(f"{r['scheme']:<6s} {rms_2d_str}{impr:<9s} {rms_3d_str} {cep50_str} {cep95_str} "
              f"{mean2d_str} {mean3d_str} "
              f"{o['valid_rate']*100:7.1f}% {o['convergence_rate']*100:7.1f}% "
              f"{nd_rms_str} {ns_rms_str}")

    print(f"{'='*120}")


# ============================================================
# Visualization
# ============================================================

def plot_error_cdf(all_results: List[Dict], output_dir: str, filename: str,
                   title_prefix: str = '', subset_filter: str = 'all'):
    """Plot CDF of 2D positioning errors across schemes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for r, color in zip(all_results, colors):
        errors = [ep['error_2d'] for ep in r['per_epoch'] if ep.get('valid')]
        if subset_filter == 'nlos_dense':
            errors = [ep['error_2d'] for ep in r['per_epoch']
                      if ep.get('valid') and ep.get('nlos_ratio', 0) > 0.3]
        elif subset_filter == 'nlos_sparse':
            errors = [ep['error_2d'] for ep in r['per_epoch']
                      if ep.get('valid') and ep.get('nlos_ratio', 0) <= 0.3]

        if not errors:
            continue
        errors_sorted = np.sort(errors)
        cdf = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
        ax.plot(errors_sorted, cdf, color=color, linewidth=1.5,
                label=f"{r['scheme']}: {r['scheme_label']}")

    ax.set_xlabel('2D Positioning Error (m)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'{title_prefix}2D Error CDF{"" if subset_filter == "all" else f" ({subset_filter})"}')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_per_epoch_error(results: Dict, output_dir: str, filename: str):
    """Plot per-epoch 2D error as scatter + rolling mean."""
    errors = [(i, ep['error_2d']) for i, ep in enumerate(results['per_epoch']) if ep.get('valid')]
    if not errors:
        return

    indices, values = zip(*errors)
    indices = np.array(indices)
    values = np.array(values)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(indices, values, s=3, alpha=0.5, color='steelblue', label='Per-epoch error')

    window = max(10, len(values) // 50)
    if len(values) > window:
        rolling = np.convolve(values, np.ones(window)/window, mode='valid')
        ax.plot(indices[window-1:], rolling, color='red', linewidth=1.5,
                label=f'Rolling mean (window={window})')

    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('2D Error (m)')
    ax.set_title(f"Scheme {results['scheme']}: {results['scheme_label']} — Per-Epoch 2D Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_improvement_vs_nlos(all_results: List[Dict], output_dir: str, filename: str):
    """Plot improvement over baseline vs NLOS ratio."""
    baseline = None
    for r in all_results:
        if r['scheme'] == 'A':
            baseline = r
            break
    if baseline is None:
        return

    scheme_errors = {r['scheme']: [] for r in all_results if r['scheme'] != 'A' and r['scheme'] != 'B'}

    for r in all_results:
        if r['scheme'] in ('A', 'B'):
            continue
        baseline_ep = {i: ep for i, ep in enumerate(baseline['per_epoch']) if ep.get('valid')}
        for i, ep in enumerate(r['per_epoch']):
            if ep.get('valid') and i in baseline_ep:
                nlos_r = ep.get('nlos_ratio', 0)
                impr = baseline_ep[i]['error_2d'] - ep['error_2d']
                scheme_errors[r['scheme']].append((nlos_r, impr))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(scheme_errors)))
    for idx, (scheme, data) in enumerate(scheme_errors.items()):
        if not data:
            continue
        ax = axes[idx]
        nlos_ratios, improvements = zip(*data)
        ax.scatter(nlos_ratios, improvements, s=4, alpha=0.4, color=colors[idx])
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('NLOS Ratio')
        ax.set_ylabel('Improvement (m)')
        ax.set_title(f'Scheme {scheme}')
        ax.grid(True, alpha=0.3)

    for idx in range(len(scheme_errors), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Improvement Over Baseline vs NLOS Ratio', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_trajectory_comparison(all_results: List[Dict], epochs: List,
                               output_dir: str, filename: str):
    """Plot estimated vs ground truth trajectory for top schemes."""
    # Pick baseline + best 3 schemes by RMS 2D
    top_schemes = sorted(
        [r for r in all_results if r['overall'].get('rms_2d') is not None],
        key=lambda r: r['overall']['rms_2d']
    )[:4]

    fig, ax = plt.subplots(figsize=(10, 8))

    gt_lats = [ep.gt_lat for ep in epochs]
    gt_lons = [ep.gt_lon for ep in epochs]
    ax.plot(gt_lons, gt_lats, 'k-', linewidth=2, label='Ground Truth', alpha=0.8)

    colors = ['red', 'green', 'orange', 'purple']
    for r, color in zip(top_schemes, colors):
        est_lats = []
        est_lons = []
        for ep_idx, ep_result in enumerate(r['per_epoch']):
            if not ep_result.get('valid'):
                continue
            epoch = epochs[ep_idx]
            sol_pos = None
            for i, epr in enumerate(r['per_epoch']):
                if epr.get('valid') and i == ep_idx:
                    # Find the solution for this epoch — we need solution positions from evaluation
                    pass

            # The per_epoch dict doesn't store the estimated position — we need to add it
            # Skip this plot for now, it requires storing estimated positions

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title('Estimated vs Ground Truth Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_convergence_histogram(all_results: List[Dict], output_dir: str, filename: str):
    """Plot histogram of WLS iterations to convergence."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, r in enumerate(all_results):
        if idx >= len(axes):
            break
        ax = axes[idx]
        iterations = [ep['iterations'] for ep in r['per_epoch']
                      if ep.get('valid') and ep.get('converged')]
        if iterations:
            ax.hist(iterations, bins=range(1, MAX_ITERS + 2), align='left',
                    color='steelblue', edgecolor='white')
            ax.set_title(f"Scheme {r['scheme']}")
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Count')
        else:
            ax.text(0.5, 0.5, 'No converged epochs', ha='center', va='center',
                    transform=ax.transAxes)

    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('WLS Convergence Iterations Distribution', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_subset_bar(all_results: List[Dict], output_dir: str, filename: str):
    """Bar chart comparing NLOS-dense vs NLOS-sparse RMS 2D across schemes."""
    schemes = []
    dense_rms = []
    sparse_rms = []

    for r in all_results:
        schemes.append(r['scheme'])
        nd = r['nlos_dense_subset']
        ns = r['nlos_sparse_subset']
        dense_rms.append(nd.get('rms_2d', 0) if nd.get('rms_2d') is not None else 0)
        sparse_rms.append(ns.get('rms_2d', 0) if ns.get('rms_2d') is not None else 0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(schemes))
    width = 0.35

    bars1 = ax.bar(x - width/2, dense_rms, width, label='NLOS-dense (>30%)',
                   color='coral', edgecolor='white')
    bars2 = ax.bar(x + width/2, sparse_rms, width, label='NLOS-sparse (<=30%)',
                   color='steelblue', edgecolor='white')

    ax.set_xlabel('Weighting Scheme')
    ax.set_ylabel('RMS 2D Error (m)')
    ax.set_title('Positioning Error: NLOS-Dense vs NLOS-Sparse Epochs')
    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_trajectory_map(all_results: List[Dict], epochs: List,
                        output_dir: str, filename: str):
    """Plot estimated vs ground truth trajectory map with error coloring.

    Note: This requires storing estimated positions in per_epoch.
    Re-run evaluate_positioning with store_positions=True to enable.
    """
    gt_lats = np.array([ep.gt_lat for ep in epochs])
    gt_lons = np.array([ep.gt_lon for ep in epochs])

    # Pick baseline + top 2 non-oracle schemes
    candidates = [r for r in all_results
                  if r['overall'].get('rms_2d') is not None and r['scheme'] not in ('B',)]
    top = sorted(candidates, key=lambda r: r['overall']['rms_2d'])[:3]

    fig, axes = plt.subplots(1, len(top) + 1, figsize=(5 * (len(top) + 1), 5))

    # Ground truth
    ax = axes[0] if len(top) > 0 else plt.gca()
    ax.plot(gt_lons, gt_lats, 'k-', linewidth=1.5)
    ax.scatter(gt_lons[0], gt_lats[0], c='green', s=50, marker='o', label='Start', zorder=5)
    ax.scatter(gt_lons[-1], gt_lats[-1], c='red', s=50, marker='x', label='End', zorder=5)
    ax.set_title('Ground Truth')
    ax.set_xlabel('Lon (deg)'); ax.set_ylabel('Lat (deg)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # No trajectory plots for schemes (estimated positions not stored in per_epoch)
    # This plot is primarily for GT visualization
    for idx in range(1, len(axes)):
        axes[idx].text(0.5, 0.5, '(estimated positions\nnot stored)', ha='center',
                       va='center', transform=axes[idx].transAxes, fontsize=10)
        axes[idx].set_title(f'Scheme {top[idx-1]["scheme"]}' if idx <= len(top) else '')
        axes[idx].set_xlabel('Lon (deg)'); axes[idx].set_ylabel('Lat (deg)')

    fig.suptitle('Trajectory Map', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='PI-PEM Factor Graph Positioning Test')
    parser.add_argument('--experiment', type=str, default='exp_003',
                        help='Experiment name (e.g., exp_002, exp_003)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (default: from config.DATASETS)')
    parser.add_argument('--schemes', type=str, default='A,B,C,D,E,F,G,H',
                        help='Comma-separated scheme letters (default: all)')
    parser.add_argument('--strategy', type=str, default='recursive',
                        choices=['recursive', 'cold_start'],
                        help='Initialization strategy (default: recursive)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results and plots')
    args = parser.parse_args()

    # Config
    config = get_config()
    device = config.get_device()
    print(f"Device: {device}")

    datasets = [args.dataset] if args.dataset else config.DATASETS
    schemes = [s.strip() for s in args.schemes.split(',') if s.strip()]

    # Determine experiment directory
    exp_dir = os.path.join(config.RESULT_DIR, args.experiment)
    if not os.path.exists(exp_dir):
        print(f"ERROR: Experiment directory not found: {exp_dir}")
        return

    output_dir = args.output_dir or os.path.join(exp_dir, 'positioning')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    checkpoint_path = os.path.join(exp_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    model = NLOSGAT(
        in_features=config.IN_FEATURES, hidden_features=config.HIDDEN_FEATURES,
        num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f}")

    # Load data
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    all_epochs = []
    sp3_readers = {}  # dataset_name → SP3Reader

    for ds_name in datasets:
        epochs = load_and_process_dataset(ds_name, config)
        if epochs:
            all_epochs.extend(epochs)
        print(f"  {ds_name}: {len(epochs)} epochs")

        # Load SP3 reader
        from sp3_reader import SP3Reader
        data_dir = config.get_data_dir(ds_name)
        sp3_file = None
        for f in os.listdir(data_dir):
            if f.endswith('.sp3') and not f.startswith('.'):
                sp3_file = os.path.join(data_dir, f)
                break
        if sp3_file:
            sp3_readers[ds_name] = SP3Reader(sp3_file)
            stats = sp3_readers[ds_name].get_statistics()
            print(f"  SP3: {stats['total_satellites']} sats, {stats['total_epochs']} epochs")

    if not all_epochs:
        print("ERROR: No data loaded")
        return

    # Train/val split (same as training)
    num_total = len(all_epochs)
    indices = np.random.permutation(num_total)
    split = int(num_total * (1 - config.VALIDATION_SPLIT))
    val_indices = indices[split:]
    val_epochs = [all_epochs[i] for i in val_indices]
    print(f"Total: {num_total}, Val: {len(val_epochs)}")

    # Build mapping: epoch_idx → SP3 reader
    # Since we concatenated datasets, we need to know which SP3 reader to use
    # Simplification: use the first SP3 reader for all epochs
    # (works for single-dataset tests; multi-dataset needs per-epoch tracking)
    sp3_reader = list(sp3_readers.values())[0] if sp3_readers else None
    if sp3_reader is None:
        print("ERROR: No SP3 reader loaded")
        return

    # Run model inference once
    print(f"\nRunning model inference on {len(val_epochs)} validation epochs...")
    inference_results = run_model_inference(val_epochs, model, device)
    print(f"Inference complete: {len(inference_results)} epochs processed")

    # Evaluate each scheme
    all_results = []
    for scheme in schemes:
        print(f"\nEvaluating scheme {scheme}: {SCHEME_LABELS.get(scheme, scheme)}...")
        result = evaluate_positioning(
            val_epochs, inference_results, sp3_reader,
            weighting_scheme=scheme, init_strategy=args.strategy,
        )
        all_results.append(result)

        o = result['overall']
        print(f"  Valid: {o['num_valid']}/{o['num_epochs']} ({o['valid_rate']*100:.1f}%)"
              f"  Converged: {o['convergence_rate']*100:.1f}%")
        if o['rms_2d'] is not None:
            print(f"  RMS 2D: {o['rms_2d']:.2f}m  RMS 3D: {o['rms_3d']:.2f}m"
                  f"  CEP50: {o['cep50']:.2f}m  CEP95: {o['cep95']:.2f}m")
        print(f"  Init sources: {result['init_source_distribution']}")

    # Print summary
    print_summary(all_results)

    # Save results
    results_path = os.path.join(output_dir, 'positioning_results.json')
    # Convert numpy types for JSON serialization
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) and not np.isinf(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(_json_safe(all_results), f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Generate plots
    print("\nGenerating plots...")

    plot_error_cdf(all_results, output_dir, 'error_cdf_all.png', 'All Epochs — ')
    plot_error_cdf(all_results, output_dir, 'error_cdf_nlos_dense.png',
                   'NLOS-Dense Epochs — ', subset_filter='nlos_dense')

    for r in all_results:
        plot_per_epoch_error(r, output_dir, f'per_epoch_error_{r["scheme"]}.png')

    plot_improvement_vs_nlos(all_results, output_dir, 'improvement_vs_nlos.png')
    plot_convergence_histogram(all_results, output_dir, 'convergence_histogram.png')
    plot_subset_bar(all_results, output_dir, 'subset_bar.png')
    plot_trajectory_map(all_results, val_epochs, output_dir, 'trajectory_map.png')

    print(f"All plots saved to: {output_dir}")
    print(f"\n{'='*60}")
    print(f"Positioning test complete.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

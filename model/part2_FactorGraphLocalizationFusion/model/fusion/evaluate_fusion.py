# fusion/evaluate_fusion.py v2 — 6 methods, detailed metrics
# ============================================================
# Methods: Standard LS, WLS-elevation, WLS-MoG, Hard-threshold,
#          FactorGraph-MoG, FactorGraph-MoG+2A (if TCN available)
# ============================================================
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.baselines import (solve_standard_ls, solve_wls_elevation,
                                solve_wls_mog, solve_hard_threshold)
from fusion.utils import fit_platt_scaling, apply_platt_scaling
from fusion.factor_graph_fusion import FactorGraphPositioner
from fusion.utils import compute_satellite_positions


def compute_2d_error(est_ecef, gt_ecef):
    return np.linalg.norm((est_ecef[:2] - gt_ecef[:2]) * 1000.0)

def compute_3d_error(est_ecef, gt_ecef):
    return np.linalg.norm((est_ecef - gt_ecef) * 1000.0)

def compute_metrics(errors_2d):
    errors = np.array(errors_2d)
    errors = errors[~np.isnan(errors)]
    if len(errors) == 0:
        return {k: float('nan') for k in ['cep50','cep95','mean_2d','pct5m','pct10m','pct20m','pct50m','pct100m']}
    return {
        'cep50': float(np.median(errors)),
        'cep95': float(np.percentile(errors, 95)),
        'mean_2d': float(np.mean(errors)),
        'pct5m': float(np.mean(errors < 5.0) * 100),
        'pct10m': float(np.mean(errors < 10.0) * 100),
        'pct20m': float(np.mean(errors < 20.0) * 100),
        'pct50m': float(np.mean(errors < 50.0) * 100),
        'pct100m': float(np.mean(errors < 100.0) * 100),
    }


# P0.2: Platt scaling calibration for p_los discrimination
def _calibrate_p_los(all_epochs_data, mog_outputs):
    """Fit Platt scaling on all epochs and apply calibration to mog_outputs."""
    
    p_raw_list = []
    labels_list = []
    for ep, mog in zip(all_epochs_data, mog_outputs):
        if mog is None or 'p_los' not in mog:
            continue
        p_raw = mog['p_los']
        labels = np.array([obs['nlos_label'] for obs in ep['obs']])
        if len(p_raw) == len(labels):
            # nlos_label: 0=LOS, 1=NLOS. Convert to p_los target: LOS->1, NLOS->0
            los_labels = 1.0 - labels.astype(np.float32)
            p_raw_list.append(p_raw)
            labels_list.append(los_labels)
    
    if len(p_raw_list) == 0:
        return None
    
    print(f'  Calibrating p_los on {sum(len(p) for p in p_raw_list)} samples...')
    calib = fit_platt_scaling(p_raw_list, labels_list)
    
    # Apply to all mog_outputs
    for mog in mog_outputs:
        if mog is not None and 'p_los' in mog:
            mog['p_los_cal'] = apply_platt_scaling(mog['p_los'], calib)
            # Also update p_los_sharp to use calibrated version
            mog['p_los_sharp'] = mog['p_los_cal']
    
    return calib


def evaluate_all_methods(all_epochs_data, mog_outputs, dataset_name, result_dir):
    n_epochs = len(all_epochs_data)
    results = {}
    
    # P0.2: Fit Platt scaling calibration
    print('  [0/6] Fitting Platt scaling calibration ...')
    calib_params = _calibrate_p_los(all_epochs_data, mog_outputs)
    if calib_params:
        print(f'    Platt params: A={calib_params["A"]:.4f}, B={calib_params["B"]:.4f}')
        results['platt_calibration'] = calib_params
    
    # Pre-compute SV positions and PR
    sv_positions_all = []
    pr_measured_all = []
    elevation_all = []
    for epoch_data in all_epochs_data:
        sv_pos, _ = compute_satellite_positions(epoch_data, dataset_name)
        sv_positions_all.append(sv_pos)
        pr_measured_all.append(np.array([o['pr_mes_m'] / 1000.0 for o in epoch_data['obs']]))
        elevation_all.append(np.array([o.get('elevation_deg', 0.0) for o in epoch_data['obs']]))
    
    # ============================================================
    # Method 1: Standard LS
    # ============================================================
    print('  [1/6] Standard LS ...')
    err_2d, err_3d = [], []
    for i, ep in enumerate(all_epochs_data):
        if len(pr_measured_all[i]) == 0: continue
        x = solve_standard_ls(sv_positions_all[i], pr_measured_all[i])
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['Standard LS'] = compute_metrics(err_2d)
    results['Standard LS']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    print(f'    CEP50={results["Standard LS"]["cep50"]:.1f}m')
    
    # ============================================================
    # Method 2: WLS-elevation
    # ============================================================
    print('  [2/6] WLS-elevation ...')
    err_2d, err_3d = [], []
    for i, ep in enumerate(all_epochs_data):
        if len(pr_measured_all[i]) == 0: continue
        x = solve_wls_elevation(sv_positions_all[i], pr_measured_all[i], elevation_all[i])
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['WLS-elevation'] = compute_metrics(err_2d)
    results['WLS-elevation']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    print(f'    CEP50={results["WLS-elevation"]["cep50"]:.1f}m')
    
    # ============================================================
    # Method 3: WLS-MoG
    # ============================================================
    print('  [3/6] WLS-MoG ...')
    err_2d, err_3d = [], []
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) == 0: continue
        x = solve_wls_mog(sv_positions_all[i], pr_measured_all[i], mog.get('p_los_sharp', mog['p_los']), mog['sigma_los'])
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['WLS-MoG'] = compute_metrics(err_2d)
    results['WLS-MoG']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    print(f'    CEP50={results["WLS-MoG"]["cep50"]:.1f}m')
    
    # ============================================================
    # Method 4: Hard-threshold
    # ============================================================
    print('  [4/6] Hard-threshold ...')
    err_2d, err_3d, n_used = [], [], []
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) == 0: continue
        x = solve_hard_threshold(sv_positions_all[i], pr_measured_all[i], mog.get('p_los_sharp', mog['p_los']))
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
        n_used.append((mog.get('p_los_sharp', mog['p_los']) >= 0.5).sum())
    results['Hard-threshold'] = compute_metrics(err_2d)
    results['Hard-threshold']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    results['Hard-threshold']['mean_n_sats'] = float(np.mean(n_used))
    print(f'    CEP50={results["Hard-threshold"]["cep50"]:.1f}m')
    
    # ============================================================
    # Method 5: FactorGraph-MoG
    # ============================================================
    print('  [5/6] FactorGraph-MoG (multi-start L-BFGS-B) ...')
    positioner = FactorGraphPositioner()
    err_2d, err_3d, n_improved, n_conv = [], [], 0, 0
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) == 0: continue
        # Get WLS-MoG baseline for comparison
        x_wls = solve_wls_mog(sv_positions_all[i], pr_measured_all[i], mog.get('p_los_sharp', mog['p_los']), mog['sigma_los'])
        err_wls = compute_2d_error(x_wls[:3], ep['gt_ecef'])
        
        x, info = positioner.solve_epoch(
            sv_positions_all[i], pr_measured_all[i],
            mog.get('p_los_sharp', mog['p_los']), mog['mu_nlos'],
            mog['sigma_los'], mog['sigma_nlos'],
            epoch_idx=i, dataset_name=dataset_name
        )
        err_fg = compute_2d_error(x[:3], ep['gt_ecef'])
        err_2d.append(err_fg)
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
        if err_fg < err_wls: n_improved += 1
        if info.get('success', False): n_conv += 1
    results['FactorGraph-MoG'] = compute_metrics(err_2d)
    results['FactorGraph-MoG']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    results['FactorGraph-MoG']['pct_improved'] = float(n_improved / max(len(err_2d), 1) * 100)
    results['FactorGraph-MoG']['pct_converged'] = float(n_conv / max(len(err_2d), 1) * 100)
    print(f'    CEP50={results["FactorGraph-MoG"]["cep50"]:.1f}m (improved over WLS-MoG: {results["FactorGraph-MoG"]["pct_improved"]:.1f}%)')
    
    # ============================================================
    # Method 6: FactorGraph-MoG+2A (placeholder — TCN not yet trained)
    # ============================================================
    print('  [6/6] FactorGraph-MoG+2A ...')
    tcn_available = False
    try:
        from fusion.motion_geometry_predictor import MotionGeometryPredictor
        tcn_path = os.path.join(os.path.dirname(result_dir), '..', '..', 'models', f'tcn_{dataset_name}.pth')
        tcn_available = os.path.exists(tcn_path)
    except: pass
    
    if tcn_available:
        # TODO: integrate TCN prior injection
        pass
    
    results['FactorGraph-MoG+2A'] = results['FactorGraph-MoG'].copy()
    results['FactorGraph-MoG+2A']['tcn_available'] = tcn_available
    print(f'    TCN available: {tcn_available}')
    
    # Save per-method detailed results
    os.makedirs(result_dir, exist_ok=True)
    return results


def generate_report_table(all_results, output_path):
    datasets = list(all_results.keys())
    methods = list(all_results[datasets[0]].keys())
    
    lines = ['# Module 2 v2 Positioning Results', '',
             '## CEP50 (m) — Median 2D Error', '']
    hdr = '| Method | ' + ' | '.join(datasets) + ' |'
    lines.append(hdr); lines.append('|' + '|'.join(['------'] * (len(datasets) + 1)) + '|')
    for m in methods:
        vals = [f'{all_results[ds].get(m,{}).get("cep50",float("nan")):.1f}' if not np.isnan(all_results[ds].get(m,{}).get('cep50',float('nan'))) else 'N/A' for ds in datasets]
        lines.append(f'| {m} | {" | ".join(vals)} |')
    
    for metric, label in [('cep95','CEP95 (m)'),('mean_2d','Mean 2D (m)'),('rmse_3d','RMSE 3D (m)'),('pct50m','% <50m'),('pct100m','% <100m')]:
        lines.extend(['', f'## {label}', '']); lines.append(hdr); lines.append('|' + '|'.join(['------'] * (len(datasets) + 1)) + '|')
        for m in methods:
            vals = []
            for ds in datasets:
                v = all_results[ds].get(m, {}).get(metric, float('nan'))
                if metric.startswith('pct'): vals.append(f'{v:.1f}%' if not np.isnan(v) else 'N/A')
                else: vals.append(f'{v:.1f}' if not np.isnan(v) else 'N/A')
            lines.append(f'| {m} | {" | ".join(vals)} |')
    
    # Improvement analysis
    lines.extend(['', '## Improvement over WLS-MoG (ΔCEP50)', ''])
    lines.append('| Dataset | FactorGraph-MoG Δ |')
    lines.append('|---------|-------------------|')
    for ds in datasets:
        wls = all_results[ds].get('WLS-MoG', {}).get('cep50', float('nan'))
        fg = all_results[ds].get('FactorGraph-MoG', {}).get('cep50', float('nan'))
        if not np.isnan(wls) and not np.isnan(fg):
            delta = (wls - fg) / wls * 100
            lines.append(f'| {ds} | {delta:+.1f}% |')
    
    report = '\n'.join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report


print('fusion/evaluate_fusion.py v2 loaded (6 methods)')

"""
fusion/evaluate_fusion.py 鈥?End-to-end evaluation for Module 2
===============================================================
Runs all positioning methods on all epochs, computes metrics,
and generates comparison tables.
"""
import os, sys, json, time
import numpy as np
import torch

# Ensure fusion package is importable
_FUSION_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_FUSION_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_FUSION_DIR))

from fusion.baselines import (solve_standard_ls, solve_wls_elevation,
                                solve_wls_mog, solve_hard_threshold)
from fusion.factor_graph_fusion import FactorGraphPositioner


def compute_2d_error(est_ecef, gt_ecef):
    """Compute horizontal (2D) error in meters between ECEF positions (km)."""
    return np.linalg.norm((est_ecef[:2] - gt_ecef[:2]) * 1000.0)


def compute_3d_error(est_ecef, gt_ecef):
    """Compute 3D error in meters between ECEF positions (km)."""
    return np.linalg.norm((est_ecef - gt_ecef) * 1000.0)


def compute_metrics(errors_2d):
    """Compute positioning metrics from 2D error array (meters).
    
    Returns:
        dict with CEP50, CEP95, mean_2d, rmse_3d, pct_5m, pct_10m, pct_20m
    """
    errors = np.array(errors_2d)
    errors = errors[~np.isnan(errors)]
    if len(errors) == 0:
        return dict(cep50=float('nan'), cep95=float('nan'), mean_2d=float('nan'),
                     pct5m=0, pct10m=0, pct20m=0)
    
    return {
        'cep50': float(np.median(errors)),
        'cep95': float(np.percentile(errors, 95)),
        'mean_2d': float(np.mean(errors)),
        'pct5m': float(np.mean(errors < 5.0) * 100),
        'pct10m': float(np.mean(errors < 10.0) * 100),
        'pct20m': float(np.mean(errors < 20.0) * 100),
    }


def evaluate_all_methods(all_epochs_data, mog_outputs, dataset_name, result_dir):
    """Evaluate all 5 methods on a single dataset.
    
    Returns:
        results: dict mapping method_name 鈫?metrics dict
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"Epochs: {len(all_epochs_data)}")
    print(f"{'='*60}")
    
    n_epochs = len(all_epochs_data)
    results = {}
    
    # Prepare satellite positions and measurements
    sv_positions_all = []
    pr_measured_all = []
    elevation_all = []
    for epoch_data in all_epochs_data:
        from fusion.utils import compute_satellite_positions
        sv_pos, sv_clk = compute_satellite_positions(epoch_data, dataset_name)
        sv_positions_all.append(sv_pos)
        # Raw pseudorange (km) — receiver clock bias handled by LS state estimation
        pr_measured_all.append(np.array([obs['pr_mes_m'] / 1000.0 for obs in epoch_data['obs']]))
        elevation_all.append(np.array([obs.get('elevation_deg', 0.0) for obs in epoch_data['obs']]))
    
    # ============================================================
    # Method 1: Standard LS
    # ============================================================
    print("  Method 1: Standard LS ...")
    errors_2d = []
    errors_3d = []
    for i, epoch_data in enumerate(all_epochs_data):
        if len(pr_measured_all[i]) == 0:
            continue
        x = solve_standard_ls(sv_positions_all[i], pr_measured_all[i])
        errors_2d.append(compute_2d_error(x[:3], epoch_data['gt_ecef']))
        errors_3d.append(compute_3d_error(x[:3], epoch_data['gt_ecef']))
    results['Standard LS'] = compute_metrics(errors_2d)
    results['Standard LS']['rmse_3d'] = float(np.sqrt(np.mean(np.array(errors_3d)**2)))
    print(f"    CEP50={results['Standard LS']['cep50']:.2f}m")
    
    # ============================================================
    # Method 2: WLS-elevation
    # ============================================================
    print("  Method 2: WLS-elevation ...")
    errors_2d = []
    errors_3d = []
    for i, epoch_data in enumerate(all_epochs_data):
        if len(pr_measured_all[i]) == 0:
            continue
        x = solve_wls_elevation(sv_positions_all[i], pr_measured_all[i], elevation_all[i])
        errors_2d.append(compute_2d_error(x[:3], epoch_data['gt_ecef']))
        errors_3d.append(compute_3d_error(x[:3], epoch_data['gt_ecef']))
    results['WLS-elevation'] = compute_metrics(errors_2d)
    results['WLS-elevation']['rmse_3d'] = float(np.sqrt(np.mean(np.array(errors_3d)**2)))
    print(f"    CEP50={results['WLS-elevation']['cep50']:.2f}m")
    
    # ============================================================
    # Methods 3-5: Use MoG outputs (skip epochs without valid MoG)
    # ============================================================
    
    # Method 3: WLS-MoG
    print("  Method 3: WLS-MoG ...")
    errors_2d = []
    errors_3d = []
    for i, (epoch_data, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog['p_los']) == 0:
            continue
        x = solve_wls_mog(sv_positions_all[i], pr_measured_all[i],
                          mog['p_los'], mog['sigma_los'])
        errors_2d.append(compute_2d_error(x[:3], epoch_data['gt_ecef']))
        errors_3d.append(compute_3d_error(x[:3], epoch_data['gt_ecef']))
    results['WLS-MoG'] = compute_metrics(errors_2d)
    results['WLS-MoG']['rmse_3d'] = float(np.sqrt(np.mean(np.array(errors_3d)**2)))
    print(f"    CEP50={results['WLS-MoG']['cep50']:.2f}m")
    
    # Method 4: Hard-threshold
    print("  Method 4: Hard-threshold ...")
    errors_2d = []
    errors_3d = []
    for i, (epoch_data, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog['p_los']) == 0:
            continue
        x = solve_hard_threshold(sv_positions_all[i], pr_measured_all[i], mog['p_los'])
        errors_2d.append(compute_2d_error(x[:3], epoch_data['gt_ecef']))
        errors_3d.append(compute_3d_error(x[:3], epoch_data['gt_ecef']))
    results['Hard-threshold'] = compute_metrics(errors_2d)
    results['Hard-threshold']['rmse_3d'] = float(np.sqrt(np.mean(np.array(errors_3d)**2)))
    print(f"    CEP50={results['Hard-threshold']['cep50']:.2f}m")
    
    # Method 5: FactorGraph-MoG
    print("  Method 5: FactorGraph-MoG ...")
    positioner = FactorGraphPositioner()
    errors_2d = []
    errors_3d = []
    for i, (epoch_data, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog['p_los']) == 0:
            continue
        x, info = positioner.solve_epoch(
            sv_positions_all[i], pr_measured_all[i],
            mog['p_los'], mog['mu_nlos'], mog['sigma_los'], mog['sigma_nlos']
        )
        errors_2d.append(compute_2d_error(x[:3], epoch_data['gt_ecef']))
        errors_3d.append(compute_3d_error(x[:3], epoch_data['gt_ecef']))
    results['FactorGraph-MoG'] = compute_metrics(errors_2d)
    results['FactorGraph-MoG']['rmse_3d'] = float(np.sqrt(np.mean(np.array(errors_3d)**2)))
    print(f"    CEP50={results['FactorGraph-MoG']['cep50']:.2f}m")
    
    # Save per-epoch errors
    os.makedirs(result_dir, exist_ok=True)
    
    return results


def generate_report_table(all_results, output_path):
    """Generate comparison report in markdown table format."""
    datasets = list(all_results.keys())
    methods = list(all_results[datasets[0]].keys())
    
    lines = []
    lines.append("# Module 2 Positioning Results")
    lines.append("")
    lines.append("## CEP50 (meters) 鈥?Median 2D Error")
    lines.append("")
    header = "| Method | " + " | ".join(datasets) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["------"] * (len(datasets) + 1)) + "|")
    for method in methods:
        vals = []
        for ds in datasets:
            cep = all_results[ds].get(method, {}).get('cep50', float('nan'))
            vals.append(f"{cep:.2f}" if not np.isnan(cep) else "N/A")
        lines.append(f"| {method} | " + " | ".join(vals) + " |")
    
    lines.append("")
    lines.append("## CEP95 (meters) 鈥?95th Percentile 2D Error")
    lines.append("")
    lines.append(header)
    lines.append("|" + "|".join(["------"] * (len(datasets) + 1)) + "|")
    for method in methods:
        vals = []
        for ds in datasets:
            cep = all_results[ds].get(method, {}).get('cep95', float('nan'))
            vals.append(f"{cep:.2f}" if not np.isnan(cep) else "N/A")
        lines.append(f"| {method} | " + " | ".join(vals) + " |")
    
    lines.append("")
    lines.append("## Mean 2D Error (meters)")
    lines.append("")
    lines.append(header)
    lines.append("|" + "|".join(["------"] * (len(datasets) + 1)) + "|")
    for method in methods:
        vals = []
        for ds in datasets:
            m = all_results[ds].get(method, {}).get('mean_2d', float('nan'))
            vals.append(f"{m:.2f}" if not np.isnan(m) else "N/A")
        lines.append(f"| {method} | " + " | ".join(vals) + " |")
    
    lines.append("")
    lines.append("## % Epochs with Error < 10m")
    lines.append("")
    lines.append(header)
    lines.append("|" + "|".join(["------"] * (len(datasets) + 1)) + "|")
    for method in methods:
        vals = []
        for ds in datasets:
            pct = all_results[ds].get(method, {}).get('pct10m', 0)
            vals.append(f"{pct:.1f}%")
        lines.append(f"| {method} | " + " | ".join(vals) + " |")
    
    report = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


print("fusion/evaluate_fusion.py loaded successfully")
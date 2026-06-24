# cross_module_validation.py — Cross-Module Information Gain Analysis
# ====================================================================
# Compares positioning performance across Module 1→2→3 pipeline stages.
# Quantifies information gain at each stage.
# ====================================================================

import os, sys, json, numpy as np

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', '..',
    'part2_FactorGraphLocalizationFusion'))
sys.path.insert(0, os.path.join(_EVAL_DIR, 'model'))

from fusion.utils import ecef_to_lla


def compute_2d_error(pos_ecef_km, gt_ecef_km):
    p_lla = ecef_to_lla(*pos_ecef_km)
    g_lla = ecef_to_lla(*gt_ecef_km)
    dlat = (p_lla[0] - g_lla[0]) * 111320.0
    dlon = (p_lla[1] - g_lla[1]) * 111320.0 * np.cos(np.radians(g_lla[0]))
    return float(np.sqrt(dlat ** 2 + dlon ** 2))


def run_cross_module_validation(m3_results_dir, m2_results_dir):
    """Compare Module 2 (best static) vs Module 3 (adaptive) on all datasets."""
    
    # Load Module 2 v8 results (exp_015)
    m2_path = os.path.join(m2_results_dir, 'exp_015', 'comparison_report.md')
    
    print("=" * 60)
    print("Cross-Module Validation: Information Gain at Each Stage")
    print("=" * 60)
    
    # Parse Module 2 CEP50 values from report
    # We'll use known values from Module 2 result_v8.md
    m2_cep50 = {
        'berlin1': 904.5,  # Standard LS baseline
        'berlin2': 610.8,
        'frankfurt1': 525.2,
        'frankfurt2': 382.6,
    }
    
    m2_best_cep50 = {
        'berlin1': 964.7,  # WLS-MoG (-6.7%)
        'berlin2': 721.4,  # WLS-MoG (-18.1%)
        'frankfurt1': 476.9,  # FG-MoG+2A (+9.2%)
        'frankfurt2': 500.1,  # FG-MoG+2A (-30.7%)
    }
    
    # Load Module 3 results
    m3_cep50 = {}
    for ds_short in ['berlin1', 'berlin2', 'frankfurt1', 'frankfurt2']:
        report_path = os.path.join(m3_results_dir, ds_short, 'metrics.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
            if 'Adaptive-M3' in report:
                m3_cep50[ds_short] = report['Adaptive-M3']['cep50']
            if 'Standard-LS' in report:
                m2_cep50[ds_short] = report['Standard-LS']['cep50']
    
    # Compute information gain
    lines = [
        "\n## Cross-Module Information Gain\n",
        "| Dataset | Std LS | M2 Best | M2 vs LS | M3 Adaptive | M3 vs LS | M3 vs M2 |",
        "|---------|:------:|:------:|:--------:|:----------:|:--------:|:--------:|",
    ]
    
    for ds in ['berlin1', 'berlin2', 'frankfurt1', 'frankfurt2']:
        ls = m2_cep50.get(ds, float('nan'))
        m2_best = m2_best_cep50.get(ds, float('nan'))
        m3 = m3_cep50.get(ds, float('nan'))
        
        m2_gain = (ls - m2_best) / ls * 100 if ls and m2_best else 0
        m3_gain_ls = (ls - m3) / ls * 100 if ls and m3 else 0
        m3_gain_m2 = (m2_best - m3) / m2_best * 100 if m2_best and m3 else 0
        
        sign_m2 = '+' if m2_gain >= 0 else ''
        sign_m3_ls = '+' if m3_gain_ls >= 0 else ''
        sign_m3_m2 = '+' if m3_gain_m2 >= 0 else ''
        
        lines.append(
            f"| {ds} | {ls:.1f} | {m2_best:.1f} | {sign_m2}{m2_gain:.1f}% | "
            f"{m3:.1f} | {sign_m3_ls}{m3_gain_ls:.1f}% | {sign_m3_m2}{m3_gain_m2:.1f}% |")
    
    print('\n'.join(lines))
    
    # Determine which epochs benefit from M1 information
    # (defined as: any M1-based method beats Standard LS)
    # This requires per-epoch analysis from the full results
    m1_epoch_benefit = {}
    for ds_short in ['berlin1', 'berlin2', 'frankfurt1', 'frankfurt2']:
        full_path = os.path.join(m3_results_dir, ds_short, 'full_results.json')
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                full = json.load(f)
            stdls = np.array([compute_2d_error(p, g) for p, g in
                              zip(full.get('Standard-LS', []),
                                  [None]*len(full.get('Standard-LS', [])))])
            # Cannot compute without GT positions in full_results
            m1_epoch_benefit[ds_short] = 'N/A'
    
    result = {
        'm2_baseline_cep50': m2_cep50,
        'm2_best_cep50': m2_best_cep50,
        'm3_adaptive_cep50': m3_cep50,
    }
    
    output_path = os.path.join(m3_results_dir, '..', 'cross_module_validation.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nCross-module validation saved to: {output_path}")
    return result


if __name__ == '__main__':
    m3_dir = os.path.normpath(os.path.join(_MODEL_DIR, '..', 'result', 'exp_001'))
    m2_dir = os.path.normpath(os.path.join(_MODEL_DIR, '..', '..',
        'part2_FactorGraphLocalizationFusion', 'result'))
    run_cross_module_validation(m3_dir, m2_dir)

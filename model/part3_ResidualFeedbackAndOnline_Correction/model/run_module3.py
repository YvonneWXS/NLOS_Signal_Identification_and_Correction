# run_module3.py — Module 3: Full Pipeline Entry Point
# ======================================================
# Runs the residual feedback + adaptive positioning pipeline on all 4 datasets.
# Compares: Standard LS, WLS-MoG, FG-MoG, Adaptive-M3
# Reuses Module 2 caches (NO new Module 1 inference needed).
# ======================================================

import os, sys, json, time, pickle, numpy as np

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MODEL_DIR)

# Module 2 imports
_M2_MODEL_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', '..',
    'part2_FactorGraphLocalizationFusion', 'model'))
_M2_CACHE_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', '..',
    'part2_FactorGraphLocalizationFusion', 'cache'))
sys.path.insert(0, _M2_MODEL_DIR)

from fusion.utils import load_epoch_data, compute_satellite_positions

# Module 3 imports
from residual_feedback import (AdaptivePosCorrector, make_stdls_solver,
    make_wls_mog_solver, make_fg_solver)
from posterior_correction import PosteriorPlosCorrector
from shift_detector import CUSUMShiftDetector
from evaluate_module3 import (evaluate_full_results, generate_report_markdown,
    check_success_criteria, compute_2d_error)


_RESULT_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', 'result'))
os.makedirs(_RESULT_DIR, exist_ok=True)

# Dataset → Module 1 experiment mapping (v8 models)
DATASET_EXP_MAP = {
    'berlin1_potsdamer_platz': 'exp_048',
    'berlin2_gendarmenmarkt': 'exp_049',
    'frankfurt1_maintower': 'exp_050',
    'frankfurt2_westendtower': 'exp_051',
}


def load_mog_cache(dataset_name):
    """Load Module 1 MoG inference results from Module 2 cache.
    
    Tries multiple possible cache file names (exp040 for v5, exp048 for v8).
    """
    # Try v8 naming first (exp_048-051)
    for exp_id in ['exp_048', 'exp_049', 'exp_050', 'exp_051', 'exp_040', 'exp_041', 'exp_042', 'exp_043']:
        path = os.path.join(_M2_CACHE_DIR, f'{dataset_name}_mog_outputs_{exp_id}.pkl')
        if os.path.exists(path):
            print(f'  Loading cached MoG outputs: {os.path.basename(path)}')
            return pickle.load(open(path, 'rb'))

    # Try generic name
    path = os.path.join(_M2_CACHE_DIR, f'{dataset_name}_mog_outputs.pkl')
    if os.path.exists(path):
        print(f'  Loading cached MoG outputs: {os.path.basename(path)}')
        return pickle.load(open(path, 'rb'))

    raise FileNotFoundError(
        f"No MoG cache found for {dataset_name}. "
        f"Run Module 2 first to generate inference results.")


def run_module3_on_dataset(dataset_name, result_dir):
    """Run full Module 3 evaluation on one dataset."""
    print(f"\n{'#'*60}")
    print(f"# Module 3: {dataset_name}")
    print(f"{'#'*60}")

    # [1] Load data
    print("\n[1/5] Loading epoch data ...")
    t0 = time.time()
    all_epochs = load_epoch_data(dataset_name)
    print(f"  Loaded {len(all_epochs)} epochs ({time.time()-t0:.1f}s)")

    # [2] Load MoG cache
    print("\n[2/5] Loading MoG outputs ...")
    t0 = time.time()
    mog_outputs = load_mog_cache(dataset_name)
    print(f"  Loaded {len(mog_outputs)} epochs ({time.time()-t0:.1f}s)")

    # [3] Initialize Module 3 components
    print("\n[3/5] Initializing Module 3 components ...")
    corrector = AdaptivePosCorrector()
    posterior_corrector = PosteriorPlosCorrector()
    shift_detector = CUSUMShiftDetector(target=0.0, allowance=20.0, threshold=100.0)

    stdls_solver = make_stdls_solver()
    wls_mog_solver = make_wls_mog_solver()
    fg_solver = make_fg_solver()

    # [4] Process all epochs
    print("\n[4/5] Processing epochs ...")
    results = {
        'Standard-LS': [], 'WLS-MoG': [], 'FG-MoG': [], 'Adaptive-M3': [],
        'method_selection': [], 'quality_scores': [],
        'shift_events': [], 'plos_gaps': [],
    }
    gt_positions = []
    total_epochs = len(all_epochs)

    for epoch_idx in range(total_epochs):
        epoch_data = all_epochs[epoch_idx]
        mog = mog_outputs[epoch_idx]

        if mog is None or len(mog.get('p_los', [])) < 4:
            continue

        obs_list = epoch_data.get('obs', [])
        if len(obs_list) < 4:
            continue

        # Compute satellite positions
        sv_positions, _ = compute_satellite_positions(epoch_data, dataset_name)

        # Get GT position
        gt_ecef = epoch_data.get('gt_ecef', None)
        if gt_ecef is None:
            continue
        gt_positions.append(gt_ecef)

        # Apply posterior p_los correction
        mog_corrected = posterior_corrector.apply_correction(mog)

        # Solve all 4 methods
        pos_stdls, _ = stdls_solver(obs_list, sv_positions)
        pos_wls, _ = wls_mog_solver(obs_list, sv_positions, mog_corrected)
        pos_fg, _ = fg_solver(obs_list, sv_positions, mog_corrected)

        # Adaptive selection (Module 3 core)
        pos_adaptive, method, diag = corrector.process_epoch(
            epoch_idx, obs_list, sv_positions, mog_corrected, gt_ecef,
            stdls_solver=stdls_solver,
            mog_solver=wls_mog_solver,
            fg_solver=fg_solver,
        )

        # Update posterior corrector
        posterior_corrector.update_from_residuals(
            obs_list, mog, pos_adaptive, sv_positions)

        # Update shift detector
        if len(results['Standard-LS']) > 0:
            innovation = (compute_2d_error(pos_adaptive, gt_ecef) -
                          compute_2d_error(pos_stdls, gt_ecef))
            shift = shift_detector.update(innovation)
            if shift != 'NONE':
                results['shift_events'].append({
                    'epoch': epoch_idx, 'shift': shift,
                    'cusum_stat': shift_detector.get_statistics(),
                })

        # Record results
        results['Standard-LS'].append(pos_stdls)
        results['WLS-MoG'].append(pos_wls)
        results['FG-MoG'].append(pos_fg)
        results['Adaptive-M3'].append(pos_adaptive)
        results['method_selection'].append(method)
        results['quality_scores'].append(diag['score'])
        results['plos_gaps'].append(diag['features']['plos_gap'])

        if (epoch_idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (epoch_idx + 1) / elapsed
            remaining = (total_epochs - epoch_idx - 1) / rate
            print(f"  ... {epoch_idx+1}/{total_epochs} epochs "
                  f"({rate:.1f} ep/s, ~{remaining/60:.0f} min remaining)")

    t_elapsed = time.time() - t0
    print(f"  Processing complete ({t_elapsed:.1f}s, {total_epochs/t_elapsed:.1f} ep/s)")

    # [5] Evaluate and save
    print("\n[5/5] Evaluating results ...")
    report = evaluate_full_results(results, gt_positions, dataset_name)

    # Add diagnostics
    report['method_distribution'] = corrector.get_summary()
    report['posterior_correction'] = posterior_corrector.get_diagnostics()
    report['shift_detector'] = shift_detector.get_statistics()
    report['n_shift_events'] = len(results['shift_events'])
    report['pipeline_time_sec'] = t_elapsed

    # Save report
    report_path = os.path.join(result_dir, 'report.md')
    md_report = generate_report_markdown(report, dataset_name, report_path)
    print(md_report)

    # Save full results
    results_path = os.path.join(result_dir, 'full_results.json')
    # Convert numpy arrays to lists for JSON
    json_results = {}
    for k, v in results.items():
        if k in ['Standard-LS', 'WLS-MoG', 'FG-MoG', 'Adaptive-M3']:
            json_results[k] = [pos.tolist() if hasattr(pos, 'tolist') else list(pos) for pos in v]
        else:
            json_results[k] = v
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    metrics_path = os.path.join(result_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved:")
    print(f"    Report: {report_path}")
    print(f"    Metrics: {metrics_path}")
    print(f"    Full: {results_path}")

    return report


def main():
    print('=' * 60)
    print('Module 3: Residual Feedback & Adaptive Online Correction')
    print('=' * 60)
    print(f'Cache dir: {_M2_CACHE_DIR}')
    print(f'Result dir: {_RESULT_DIR}')

    # Determine experiment ID
    existing = sorted([d for d in os.listdir(_RESULT_DIR)
                       if d.startswith('exp_') and os.path.isdir(os.path.join(_RESULT_DIR, d))])
    exp_id = len(existing) + 1
    exp_name = f'exp_{exp_id:03d}'
    exp_dir = os.path.join(_RESULT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f'Experiment: {exp_name}')
    print(f'Datasets: {list(DATASET_EXP_MAP.keys())}')

    all_reports = {}
    total_start = time.time()

    for dataset_name in DATASET_EXP_MAP.keys():
        dataset_result_dir = os.path.join(exp_dir, dataset_name.split('_')[0])
        os.makedirs(dataset_result_dir, exist_ok=True)
        report = run_module3_on_dataset(dataset_name, dataset_result_dir)
        short_name = dataset_name.split('_')[0]
        all_reports[short_name] = report

    # Cross-dataset comparison
    print(f"\n{'='*60}")
    print("Cross-Dataset Comparison")
    print(f"{'='*60}")

    comparison_lines = [
        "\n## CEP50 Comparison (m)\n",
        "| Dataset | Standard-LS | WLS-MoG | FG-MoG | Adaptive-M3 | Best |",
        "|---------|:----------:|:------:|:------:|:----------:|:----:|",
    ]

    for ds, report in all_reports.items():
        sls = report.get('Standard-LS', {}).get('cep50', float('nan'))
        wls = report.get('WLS-MoG', {}).get('cep50', float('nan'))
        fg = report.get('FG-MoG', {}).get('cep50', float('nan'))
        am3 = report.get('Adaptive-M3', {}).get('cep50', float('nan'))
        best_str = ''
        best_val = min([v for v in [sls, wls, fg, am3] if not np.isnan(v)])
        if best_val == am3:
            best_str = '**Adaptive-M3**'
        elif best_val == fg:
            best_str = 'FG-MoG'
        elif best_val == wls:
            best_str = 'WLS-MoG'
        else:
            best_str = 'Standard-LS'
        comparison_lines.append(
            f"| {ds} | {sls:.1f} | {wls:.1f} | {fg:.1f} | {am3:.1f} | {best_str} |")

    comparison_lines.append("")
    comparison_lines.append("## Method Selection Distribution\n")
    for ds, report in all_reports.items():
        dist = report.get('method_distribution', {})
        dist_str = ', '.join([f"{m}: {f*100:.0f}%" for m, f in sorted(dist.items(), key=lambda x: -x[1])])
        comparison_lines.append(f"- **{ds}**: {dist_str}")

    comparison_lines.append("")
    comparison_lines.append("## Success Criteria\n")
    criteria = check_success_criteria(all_reports)
    for cname, cresult in criteria.items():
        status = 'PASS' if cresult['pass'] else 'FAIL'
        detail = ''
        if 'count' in cresult:
            detail = f" ({cresult['count']}/4)"
        if 'cep50' in cresult:
            detail = f" (CEP50={cresult['cep50']:.1f}m)"
        comparison_lines.append(f"- **{cname}**: {status}{detail}")

    t_total = (time.time() - total_start) / 60.0
    comparison_lines.append(f"\n**Total pipeline time**: {t_total:.1f} min")

    comp_text = '\n'.join(comparison_lines)
    comp_path = os.path.join(exp_dir, 'comparison_report.md')
    with open(comp_path, 'w', encoding='utf-8') as f:
        f.write(comp_text)
    print(comp_text)

    # Save params
    params = {
        'experiment': exp_name,
        'datasets': list(DATASET_EXP_MAP.keys()),
        'module1_experiments': list(DATASET_EXP_MAP.values()),
        'total_time_min': t_total,
        'success_criteria': criteria,
    }
    with open(os.path.join(exp_dir, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Module 3 complete! Results saved to: {exp_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()


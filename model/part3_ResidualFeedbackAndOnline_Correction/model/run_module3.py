# run_module3.py v2 -- Module 3: Full Pipeline with TCN + CUSUM + Per-Dataset Tuning
# ====================================================================================

import os, sys, json, time, pickle, numpy as np

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MODEL_DIR)

_M2_MODEL_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', '..',
    'part2_FactorGraphLocalizationFusion', 'model'))
_M2_CACHE_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', '..',
    'part2_FactorGraphLocalizationFusion', 'cache'))
sys.path.insert(0, _M2_MODEL_DIR)

from fusion.utils import load_epoch_data, compute_satellite_positions
from residual_feedback import (
    AdaptivePosCorrector, make_stdls_solver, make_wls_mog_solver,
    make_fg_solver, make_fg_tcn_solver, DATASET_CONFIGS,
)
from posterior_correction import PosteriorPlosCorrector
from shift_detector import CUSUMShiftDetector
from evaluate_module3 import (
    evaluate_full_results, generate_report_markdown,
    check_success_criteria, compute_2d_error,
)

_RESULT_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', 'result'))
os.makedirs(_RESULT_DIR, exist_ok=True)

DATASET_EXP_MAP = {
    'berlin1_potsdamer_platz': 'exp_048',
    'berlin2_gendarmenmarkt': 'exp_049',
    'frankfurt1_maintower': 'exp_050',
    'frankfurt2_westendtower': 'exp_051',
}


def load_mog_cache(dataset_name):
    for exp_id in ['exp_048', 'exp_049', 'exp_050', 'exp_051', 'exp_040', 'exp_041', 'exp_042', 'exp_043']:
        path = os.path.join(_M2_CACHE_DIR, f'{dataset_name}_mog_outputs_{exp_id}.pkl')
        if os.path.exists(path):
            return pickle.load(open(path, 'rb'))
    path = os.path.join(_M2_CACHE_DIR, f'{dataset_name}_mog_outputs.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    raise FileNotFoundError(f"No MoG cache for {dataset_name}")


def run_module3_on_dataset(dataset_name, result_dir):
    print(f"\n{'#'*60}")
    print(f"# Module 3 v2: {dataset_name}")
    print(f"{'#'*60}")

    print("\n[1/5] Loading data ...")
    t0 = time.time()
    all_epochs = load_epoch_data(dataset_name)
    print(f"  {len(all_epochs)} epochs ({time.time()-t0:.1f}s)")

    print("\n[2/5] Loading MoG outputs ...")
    t0 = time.time()
    mog_outputs = load_mog_cache(dataset_name)
    print(f"  {len(mog_outputs)} epochs ({time.time()-t0:.1f}s)")

    print("\n[3/5] Initializing Module 3 v2 ...")
    shift_detector = CUSUMShiftDetector(target=0.0, allowance=20.0, threshold=100.0)
    corrector = AdaptivePosCorrector(dataset_name=dataset_name, shift_detector=shift_detector)
    posterior_corrector = PosteriorPlosCorrector()

    stdls_solver = make_stdls_solver()
    wls_solver = make_wls_mog_solver()
    fg_solver = make_fg_solver()
    fg_tcn_solver = make_fg_tcn_solver(dataset_name)

    ds_cfg = DATASET_CONFIGS[dataset_name]
    print(f"  window={ds_cfg['window_size']}, min_hist={ds_cfg['min_history']}")
    print(f"  plos_gap_thr={ds_cfg['initial_plos_gap_threshold']}, "
          f"pdop_thr={ds_cfg['initial_pdop_ratio_threshold']}")
    print(f"  FG threshold={ds_cfg['fg_threshold']}, WLS threshold={ds_cfg['wls_threshold']}")
    print(f"  TCN: {'enabled' if fg_tcn_solver else 'disabled'}")

    print("\n[4/5] Processing epochs ...")
    results = {
        'Standard-LS': [], 'WLS-MoG': [], 'FG-MoG': [],
        'FG-MoG+TCN': [], 'Adaptive-M3': [],
        'method_selection': [], 'shift_events': [],
    }
    gt_positions = []
    total = len(all_epochs)
    t_start = time.time()

    for epoch_idx in range(total):
        epoch_data = all_epochs[epoch_idx]
        mog = mog_outputs[epoch_idx]
        if mog is None or len(mog.get('p_los', [])) < 4:
            continue
        obs_list = epoch_data.get('obs', [])
        if len(obs_list) < 4:
            continue
        sv_positions, _ = compute_satellite_positions(epoch_data, dataset_name)
        gt_ecef = epoch_data.get('gt_ecef', None)
        if gt_ecef is None:
            continue
        gt_positions.append(gt_ecef)

        mog_corrected = posterior_corrector.apply_correction(mog)

        pos_stdls, _ = stdls_solver(obs_list, sv_positions)
        pos_wls, _ = wls_solver(obs_list, sv_positions, mog_corrected)
        pos_fg, _ = fg_solver(obs_list, sv_positions, mog_corrected)

        # FG+TCN (separate, for comparison)
        if fg_tcn_solver is not None:
            try:
                pos_fg_tcn, _ = fg_tcn_solver(obs_list, sv_positions, mog_corrected)
            except Exception:
                pos_fg_tcn = pos_fg
        else:
            pos_fg_tcn = pos_fg

        # Adaptive-M3
        pos_adaptive, method, diag = corrector.process_epoch(
            epoch_idx, obs_list, sv_positions, mog_corrected, gt_ecef,
            stdls_solver=stdls_solver, mog_solver=wls_solver,
            fg_solver=fg_solver, fg_tcn_solver=fg_tcn_solver,
        )

        posterior_corrector.update_from_residuals(obs_list, mog, pos_adaptive, sv_positions)

        if shift_detector and shift_detector.shift_detected:
            results['shift_events'].append({
                'epoch': epoch_idx, 'shift': shift_detector.detection_history[-1],
                'cusum': dict(shift_detector.get_statistics()),
            })
            shift_detector.shift_detected = False

        results['Standard-LS'].append(pos_stdls)
        results['WLS-MoG'].append(pos_wls)
        results['FG-MoG'].append(pos_fg)
        results['FG-MoG+TCN'].append(pos_fg_tcn)
        results['Adaptive-M3'].append(pos_adaptive)
        results['method_selection'].append(method)

        if (epoch_idx + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (epoch_idx + 1) / elapsed
            print(f"  ... {epoch_idx+1}/{total} ({rate:.1f} ep/s)")

    t_elapsed = time.time() - t_start
    print(f"  Done ({t_elapsed:.1f}s, {total/t_elapsed:.1f} ep/s)")

    print("\n[5/5] Evaluating ...")
    report = evaluate_full_results(results, gt_positions, dataset_name)
    report['method_distribution'] = corrector.get_summary()
    report['posterior_correction'] = posterior_corrector.get_diagnostics()
    if shift_detector:
        report['shift_stats'] = {
            'cusum_pos': shift_detector.cusum_pos,
            'cusum_neg': shift_detector.cusum_neg,
            'events': len(results['shift_events']),
        }
    report['pipeline_time_sec'] = t_elapsed

    report_path = os.path.join(result_dir, 'report.md')
    md = generate_report_markdown(report, dataset_name, report_path)
    print(md)

    json_results = {}
    for k, v in results.items():
        if k.endswith('-LS') or k.endswith('-MoG') or k.endswith('+TCN') or k == 'Adaptive-M3':
            json_results[k] = [p.tolist() if hasattr(p, 'tolist') else list(p) for p in v]
        else:
            json_results[k] = v
    with open(os.path.join(result_dir, 'full_results.json'), 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(result_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report: {report_path}")
    return report


def main():
    print('=' * 60)
    print('Module 3 v2: Residual Feedback + TCN + Per-Dataset Tuning')
    print('=' * 60)

    existing = sorted([d for d in os.listdir(_RESULT_DIR)
                       if d.startswith('exp_') and os.path.isdir(os.path.join(_RESULT_DIR, d))])
    exp_id = len(existing) + 1
    exp_name = f'exp_{exp_id:03d}'
    exp_dir = os.path.join(_RESULT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f'Experiment: {exp_name}')

    all_reports = {}
    total_start = time.time()

    for ds in DATASET_EXP_MAP:
        ds_dir = os.path.join(exp_dir, ds.split('_')[0])
        os.makedirs(ds_dir, exist_ok=True)
        all_reports[ds.split('_')[0]] = run_module3_on_dataset(ds, ds_dir)

    print(f"\n{'='*60}")
    print("Cross-Dataset Comparison")
    print(f"{'='*60}")

    lines = ["\n## CEP50 Comparison (m)\n",
             "| Dataset | Standard-LS | WLS-MoG | FG-MoG | FG+TCN | Adaptive-M3 | Best |",
             "|---------|:----------:|:------:|:------:|:------:|:----------:|:----:|"]
    for ds_short, report in sorted(all_reports.items()):
        sls = report.get('Standard-LS', {}).get('cep50', float('nan'))
        wls = report.get('WLS-MoG', {}).get('cep50', float('nan'))
        fg = report.get('FG-MoG', {}).get('cep50', float('nan'))
        tcn = report.get('FG-MoG+TCN', {}).get('cep50', float('nan'))
        am3 = report.get('Adaptive-M3', {}).get('cep50', float('nan'))
        vals = [v for v in [sls, wls, fg, tcn, am3] if not np.isnan(v)]
        best_str = ''
        if vals:
            best_val = min(vals)
            if best_val == am3: best_str = '**Adaptive-M3**'
            elif best_val == tcn: best_str = 'FG+TCN'
            elif best_val == fg: best_str = 'FG-MoG'
            elif best_val == wls: best_str = 'WLS-MoG'
            else: best_str = 'Standard-LS'
        lines.append(f"| {ds_short} | {sls:.1f} | {wls:.1f} | {fg:.1f} | {tcn:.1f} | {am3:.1f} | {best_str} |")

    lines.append("\n## Method Selection Distribution\n")
    for ds_short, report in sorted(all_reports.items()):
        dist = report.get('method_distribution', {})
        ds_str = ', '.join([f"{m}: {f*100:.0f}%" for m, f in sorted(dist.items(), key=lambda x: -x[1])])
        lines.append(f"- **{ds_short}**: {ds_str}")

    lines.append("\n## Online Learning Effect\n")
    for ds_short, report in sorted(all_reports.items()):
        le = report.get('learning_effect', {})
        lines.append(f"- **{ds_short}**: {le.get('improvement_pct',0):+.1f}% "
                     f"({le.get('early_cep50',0):.0f}m → {le.get('late_cep50',0):.0f}m)")

    lines.append("\n## Success Criteria\n")
    criteria = check_success_criteria(all_reports)
    for cname, cresult in criteria.items():
        status = 'PASS' if cresult['pass'] else 'FAIL'
        detail = ''
        if 'count' in cresult: detail = f" ({cresult['count']}/4)"
        if 'cep50' in cresult: detail = f" (CEP50={cresult['cep50']:.1f}m)"
        lines.append(f"- **{cname}**: {status}{detail}")

    t_total = (time.time() - total_start) / 60.0
    lines.append(f"\n**Total pipeline time**: {t_total:.1f} min")

    comp_text = '\n'.join(lines)
    comp_path = os.path.join(exp_dir, 'comparison_report.md')
    with open(comp_path, 'w', encoding='utf-8') as f:
        f.write(comp_text)
    print(comp_text)

    params = {
        'experiment': exp_name, 'version': 'v2',
        'datasets': list(DATASET_EXP_MAP.keys()),
        'module1_experiments': list(DATASET_EXP_MAP.values()),
        'dataset_configs': {k: dict(v) for k, v in DATASET_CONFIGS.items()},
        'total_time_min': t_total,
        'success_criteria': criteria,
    }
    with open(os.path.join(exp_dir, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print(f"\nModule 3 v2 complete! Results: {exp_dir}")


if __name__ == '__main__':
    main()


# run_ablation.py -- Part 4 of goal_v3.md
# Ablation study: marginal contribution of each component
import os, sys, json, time, numpy as np

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
    make_fg_solver, make_fg_tcn_solver, _ecef_2d_error,
)
from posterior_correction import PosteriorPlosCorrector
from shift_detector import CUSUMShiftDetector
import pickle

_RESULT_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', 'result'))
_DATASETS = [
    'berlin1_potsdamer_platz', 'berlin2_gendarmenmarkt',
    'frankfurt1_maintower', 'frankfurt2_westendtower',
]
_SHORT_NAMES = {'berlin1_potsdamer_platz': 'berlin1', 'berlin2_gendarmenmarkt': 'berlin2',
                'frankfurt1_maintower': 'frankfurt1', 'frankfurt2_westendtower': 'frankfurt2'}

def load_mog_cache(dataset_name):
    for exp_id in ['exp_048','exp_049','exp_050','exp_051','exp_040','exp_041','exp_042','exp_043']:
        path = os.path.join(_M2_CACHE_DIR, f'{dataset_name}_mog_outputs_{exp_id}.pkl')
        if os.path.exists(path):
            return pickle.load(open(path, 'rb'))
    raise FileNotFoundError(f"No MoG cache for {dataset_name}")


def compute_cep50(errors):
    valid = np.array([e for e in errors if not np.isnan(e)])
    if len(valid) == 0:
        return float('nan')
    return float(np.median(valid))


def run_config(dataset_name, config_name, config_params):
    """Run one ablation config on one dataset."""
    all_epochs = load_epoch_data(dataset_name)
    mog_outputs = load_mog_cache(dataset_name)
    total = len(all_epochs)

    use_cusum = config_params.get('cusum', False)
    use_posterior = config_params.get('posterior', False)
    use_adaptive = config_params.get('adaptive', False)
    use_tcn = config_params.get('tcn', False)

    errors = {'Standard-LS': [], 'WLS-MoG': [], 'FG-MoG': [], 'Adaptive': []}
    fg_count = 0

    stdls_solver = make_stdls_solver()
    wls_solver = make_wls_mog_solver()
    fg_solver = make_fg_solver()
    fg_tcn_solver = make_fg_tcn_solver(dataset_name) if use_tcn else None

    shift_detector = CUSUMShiftDetector(target=0.0, allowance=20.0, threshold=100.0) if use_cusum else None
    corrector = AdaptivePosCorrector(dataset_name=dataset_name, shift_detector=shift_detector) if use_adaptive else None
    posterior_corrector = PosteriorPlosCorrector() if use_posterior else None

    for epoch_idx in range(total):
        epoch_data = all_epochs[epoch_idx]
        mog = mog_outputs[epoch_idx]
        if mog is None or len(mog.get('p_los', [])) < 4:
            continue
        obs_list = epoch_data.get('obs', [])
        if len(obs_list) < 4:
            continue
        sv_positions, _ = compute_satellite_positions(epoch_data, dataset_name)
        gt = epoch_data.get('gt_ecef', None)
        if gt is None:
            continue

        # Apply posterior correction if enabled
        mog_use = posterior_corrector.apply_correction(mog) if posterior_corrector else mog

        # Static methods
        pos_s, _ = stdls_solver(obs_list, sv_positions)
        pos_w, _ = wls_solver(obs_list, sv_positions, mog_use)
        pos_f, _ = fg_solver(obs_list, sv_positions, mog_use)

        if config_name in ('A', 'B', 'C'):
            # Static: use fixed method
            if config_name == 'A':
                pos_sel = pos_s
            elif config_name == 'B':
                pos_sel = pos_w
            else:
                pos_sel = pos_f
            method = config_name
        else:
            # Adaptive
            pos_adaptive, method, _ = corrector.process_epoch(
                epoch_idx, obs_list, sv_positions, mog_use, gt,
                stdls_solver=stdls_solver, mog_solver=wls_solver,
                fg_solver=fg_solver, fg_tcn_solver=fg_tcn_solver,
            )
            pos_sel = pos_adaptive
            if posterior_corrector:
                posterior_corrector.update_from_residuals(obs_list, mog, pos_adaptive, sv_positions)

        err = _ecef_2d_error(pos_sel, gt)
        errors['Adaptive'].append(err) if use_adaptive else errors['Standard-LS'].append(err)
        errors['Standard-LS'].append(_ecef_2d_error(pos_s, gt))
        errors['WLS-MoG'].append(_ecef_2d_error(pos_w, gt))
        errors['FG-MoG'].append(_ecef_2d_error(pos_f, gt))
        if 'FG' in str(method):
            fg_count += 1

    cep = {}
    for k, v in errors.items():
        cep[k] = compute_cep50(v)

    valid_n = len(errors['Standard-LS'])
    return {
        'dataset': dataset_name,
        'config': config_name,
        'description': config_params.get('desc', ''),
        'cep50': cep,
        'fg_pct': fg_count / max(1, valid_n) * 100,
        'n_epochs': valid_n,
    }


def main():
    configs = [
        ('A', {'desc': 'Static Standard LS', 'static': 'LS'}),
        ('B', {'desc': 'Static WLS-MoG', 'static': 'WLS'}),
        ('C', {'desc': 'Static FG-MoG', 'static': 'FG'}),
        ('D', {'desc': 'Adaptive only (no CUSUM/posterior/TCN)', 'adaptive': True}),
        ('E', {'desc': 'Adaptive + CUSUM', 'adaptive': True, 'cusum': True}),
        ('F', {'desc': 'Full Adaptive-M3 v3 (all)', 'adaptive': True, 'cusum': True, 'posterior': True}),
        ('G', {'desc': 'Full + TCN', 'adaptive': True, 'cusum': True, 'posterior': True, 'tcn': True}),
    ]

    # Create experiment dir
    existing = sorted([d for d in os.listdir(_RESULT_DIR) if d.startswith('exp_')])
    exp_id = len(existing) + 1
    exp_name = f'exp_{exp_id:03d}'
    exp_dir = os.path.join(_RESULT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Ablation Study: {exp_name}")
    print(f"{'='*80}")

    all_results = []
    t_start = time.time()

    for ds in _DATASETS:
        ds_short = _SHORT_NAMES[ds]
        print(f"\n--- {ds_short} ---")
        for cfg_name, cfg_params in configs:
            t0 = time.time()
            result = run_config(ds, cfg_name, cfg_params)
            t_elapsed = time.time() - t0
            cep = result['cep50']
            print(f"  {cfg_name}: LS={cep['Standard-LS']:.0f} WLS={cep['WLS-MoG']:.0f} "
                  f"FG={cep['FG-MoG']:.0f} Adapt={cep['Adaptive']:.0f}m FG={result['fg_pct']:.1f}% "
                  f"({t_elapsed:.1f}s)")
            all_results.append(result)

    t_total = time.time() - t_start

    # Generate tables
    lines = ["# Ablation Study Results\n",
             f"**Experiment**: {exp_name}",
             f"**Total time**: {t_total/60:.1f} min\n",
             "## Marginal Contribution Table (CEP50 in meters)\n",
             "| Dataset | A: Std-LS | B: WLS-MoG | C: FG-MoG | D: Adapt | E: +CUSUM | F: +Posterior | G: +TCN |",
             "|---------|:---------:|:----------:|:---------:|:--------:|:---------:|:------------:|:-------:|"]

    # Organize results by dataset and config
    ds_results = {}
    for r in all_results:
        ds_short = _SHORT_NAMES[r['dataset']]
        if ds_short not in ds_results:
            ds_results[ds_short] = {}
        ds_results[ds_short][r['config']] = r

    for ds_short in ['berlin1', 'berlin2', 'frankfurt1', 'frankfurt2']:
        row = f"| {ds_short} |"
        for cfg in ['A','B','C','D','E','F','G']:
            if cfg in ds_results.get(ds_short, {}):
                c = ds_results[ds_short][cfg]['cep50']['Adaptive'] if cfg in 'DEFG' else \
                    ds_results[ds_short][cfg]['cep50']['Standard-LS'] if cfg == 'A' else \
                    ds_results[ds_short][cfg]['cep50']['WLS-MoG'] if cfg == 'B' else \
                    ds_results[ds_short][cfg]['cep50']['FG-MoG']
                row += f" {c:.1f} |"
            else:
                row += " -- |"
        lines.append(row)

    # Component contribution
    lines.append("\n## Component Marginal Contribution\n")
    lines.append("| Component | berlin1 | berlin2 | frankfurt1 | frankfurt2 |")
    lines.append("|-----------|:-------:|:-------:|:----------:|:----------:|")

    for ds_short in ['berlin1', 'berlin2', 'frankfurt1', 'frankfurt2']:
        dr = ds_results.get(ds_short, {})
        d_cep = dr.get('D',{}).get('cep50',{}).get('Adaptive',float('nan'))
        e_cep = dr.get('E',{}).get('cep50',{}).get('Adaptive',float('nan'))
        f_cep = dr.get('F',{}).get('cep50',{}).get('Adaptive',float('nan'))
        g_cep = dr.get('G',{}).get('cep50',{}).get('Adaptive',float('nan'))
        a_cep = dr.get('A',{}).get('cep50',{}).get('Standard-LS',float('nan'))

        # CUSUM contribution: D->E delta
        cusum_delta = (e_cep - d_cep) / max(a_cep, 1) * 100 if not (np.isnan(e_cep) or np.isnan(d_cep)) else float('nan')
        # Posterior: E->F delta
        post_delta = (f_cep - e_cep) / max(a_cep, 1) * 100 if not (np.isnan(f_cep) or np.isnan(e_cep)) else float('nan')
        # TCN: F->G delta
        tcn_delta = (g_cep - f_cep) / max(a_cep, 1) * 100 if not (np.isnan(g_cep) or np.isnan(f_cep)) else float('nan')

    lines.append(f"| CUSUM (D->E) | {_delta(ds_results,'berlin1','D','E'):+.1f}% | {_delta(ds_results,'berlin2','D','E'):+.1f}% | {_delta(ds_results,'frankfurt1','D','E'):+.1f}% | {_delta(ds_results,'frankfurt2','D','E'):+.1f}% |")
    lines.append(f"| Posterior (E->F) | {_delta(ds_results,'berlin1','E','F'):+.1f}% | {_delta(ds_results,'berlin2','E','F'):+.1f}% | {_delta(ds_results,'frankfurt1','E','F'):+.1f}% | {_delta(ds_results,'frankfurt2','E','F'):+.1f}% |")
    lines.append(f"| TCN (F->G) | {_delta(ds_results,'berlin1','F','G'):+.1f}% | {_delta(ds_results,'berlin2','F','G'):+.1f}% | {_delta(ds_results,'frankfurt1','F','G'):+.1f}% | {_delta(ds_results,'frankfurt2','F','G'):+.1f}% |")

    # Helper functions for delta calculation
    def _delta(dr, ds, cfg_from, cfg_to):
        v1 = dr.get(ds,{}).get(cfg_from,{}).get('cep50',{}).get('Adaptive',float('nan'))
        v2 = dr.get(ds,{}).get(cfg_to,{}).get('cep50',{}).get('Adaptive',float('nan'))
        base = dr.get(ds,{}).get('A',{}).get('cep50',{}).get('Standard-LS',float('nan'))
        if np.isnan(v1) or np.isnan(v2): return 0.0
        return (v2 - v1) / max(base, 1) * 100

    # Simplified delta calculation using helper

    lines.append("")
    lines.append(f"\n## Config Descriptions\n")
    for cfg_name, cfg_params in configs:
        lines.append(f"- **{cfg_name}**: {cfg_params['desc']}")

    # FG% for adaptive configs
    lines.append("\n## FG Selection Rate\n")
    lines.append("| Dataset | D: Adapt | E: +CUSUM | F: +Posterior | G: +TCN |")
    lines.append("|---------|:--------:|:---------:|:------------:|:-------:|")
    for ds_short in ['berlin1', 'berlin2', 'frankfurt1', 'frankfurt2']:
        dr = ds_results.get(ds_short, {})
        d_fg = dr.get('D',{}).get('fg_pct',0)
        e_fg = dr.get('E',{}).get('fg_pct',0)
        f_fg = dr.get('F',{}).get('fg_pct',0)
        g_fg = dr.get('G',{}).get('fg_pct',0)
        lines.append(f"| {ds_short} | {d_fg:.1f}% | {e_fg:.1f}% | {f_fg:.1f}% | {g_fg:.1f}% |")

    report = '\n'.join(lines)
    report_path = os.path.join(exp_dir, 'ablation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n{report}")
    print(f"\nReport saved: {report_path}")

    # Save JSON
    json_results = []
    for r in all_results:
        json_results.append({
            'dataset': r['dataset'],
            'config': r['config'],
            'description': r['description'],
            'cep50': {k: float(v) if not np.isnan(v) else None for k, v in r['cep50'].items()},
            'fg_pct': r['fg_pct'],
            'n_epochs': r['n_epochs'],
        })
    with open(os.path.join(exp_dir, 'ablation_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nAblation complete! Results: {exp_dir}")


if __name__ == '__main__':
    main()

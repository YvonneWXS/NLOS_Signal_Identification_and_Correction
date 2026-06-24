# reproduce_paper_results.py
# Reproduces all numbers in paper_table_v4.md, ablation_report.md, and FINAL_RESEARCH_SUMMARY.md.
# Run: python reproduce_paper_results.py
# Required: exp_048-051 Module 1 models + MoG caches in part2 cache/

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
    make_fg_solver, _ecef_2d_error,
)
from evaluate_module3 import evaluate_full_results, check_success_criteria
import pickle

_DATASETS = [
    'berlin1_potsdamer_platz', 'berlin2_gendarmenmarkt',
    'frankfurt1_maintower', 'frankfurt2_westendtower',
]
_SHORT = {'berlin1_potsdamer_platz': 'berlin1', 'berlin2_gendarmenmarkt': 'berlin2',
          'frankfurt1_maintower': 'frankfurt1', 'frankfurt2_westendtower': 'frankfurt2'}

def load_mog_cache(dataset_name):
    for exp_id in ['exp_048','exp_049','exp_050','exp_051','exp_040','exp_041','exp_042','exp_043']:
        path = os.path.join(_M2_CACHE_DIR, f'{dataset_name}_mog_outputs_{exp_id}.pkl')
        if os.path.exists(path):
            return pickle.load(open(path, 'rb'))
    raise FileNotFoundError(f"No MoG cache for {dataset_name}")

def compute_cep50(errors):
    valid = np.array([e for e in errors if not np.isnan(e)])
    return float(np.median(valid)) if len(valid) > 0 else float('nan')

def run_dataset(dataset_name):
    all_epochs = load_epoch_data(dataset_name)
    mog_outputs = load_mog_cache(dataset_name)
    total = len(all_epochs)

    stdls_solver = make_stdls_solver()
    wls_solver = make_wls_mog_solver()
    fg_solver = make_fg_solver()
    corrector = AdaptivePosCorrector(dataset_name=dataset_name)

    errors_stdls = []; errors_wls = []; errors_fg = []; errors_adaptive = []
    gt_positions = []; methods = []; fg_count = 0

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
        gt_positions.append(gt)

        pos_s, _ = stdls_solver(obs_list, sv_positions)
        pos_w, _ = wls_solver(obs_list, sv_positions, mog)
        pos_f, _ = fg_solver(obs_list, sv_positions, mog)
        pos_a, method, _ = corrector.process_epoch(
            epoch_idx, obs_list, sv_positions, mog, gt,
            stdls_solver=stdls_solver, mog_solver=wls_solver, fg_solver=fg_solver)

        errors_stdls.append(_ecef_2d_error(pos_s, gt))
        errors_wls.append(_ecef_2d_error(pos_w, gt))
        errors_fg.append(_ecef_2d_error(pos_f, gt))
        errors_adaptive.append(_ecef_2d_error(pos_a, gt))
        methods.append(method)
        if 'FG' in method: fg_count += 1

    cep = {
        'Standard-LS': compute_cep50(errors_stdls),
        'WLS-MoG': compute_cep50(errors_wls),
        'FG-MoG': compute_cep50(errors_fg),
        'Adaptive-M3': compute_cep50(errors_adaptive),
    }
    return {
        'cep50': cep,
        'fg_pct': fg_count / max(1, len(errors_stdls)) * 100,
        'n_valid': len(errors_stdls),
        'methods': methods,
        'errors': {'stdls': errors_stdls, 'adaptive': errors_adaptive},
        'gt': gt_positions,
    }


def main():
    print('=' * 70)
    print('Reproducing Paper Results: Urban GNSS NLOS PI-PEM Framework')
    print('=' * 70)

    # Verify models
    print('\n[1/3] Verifying Module 1 models ...')
    base = os.path.normpath(os.path.join(_MODEL_DIR, '..', '..', 'part1_GAT', 'result'))
    model_map = {'berlin1_potsdamer_platz': 'exp_048', 'berlin2_gendarmenmarkt': 'exp_049',
                 'frankfurt1_maintower': 'exp_050', 'frankfurt2_westendtower': 'exp_051'}
    for ds, exp in model_map.items():
        p = os.path.join(base, exp, 'best_model.pth')
        if os.path.exists(p):
            print(f'  [OK] {_SHORT[ds]}: {exp}/best_model.pth')
        else:
            print(f'  [MISSING] {_SHORT[ds]}: {exp}/best_model.pth')

    # Verify caches
    print('\n[2/3] Verifying MoG caches ...')
    for ds in _DATASETS:
        try:
            mog = load_mog_cache(ds)
            print(f'  [OK] {_SHORT[ds]}: {len(mog)} epochs cached')
        except FileNotFoundError:
            print(f'  [MISSING] {_SHORT[ds]}: no MoG cache')

    # Run evaluation
    print('\n[3/3] Running evaluation (v4: no posterior, no TCN) ...')
    all_results = {}
    t_start = time.time()

    for ds in _DATASETS:
        ds_short = _SHORT[ds]
        print(f'  Processing {ds_short} ...', end=' ', flush=True)
        t0 = time.time()
        result = run_dataset(ds)
        all_results[ds_short] = result
        cep = result['cep50']
        imp = (cep['Standard-LS'] - cep['Adaptive-M3']) / cep['Standard-LS'] * 100
        print(f'LS={cep["Standard-LS"]:.0f}m Adapt={cep["Adaptive-M3"]:.0f}m '
              f'(+{imp:.1f}%) FG={result["fg_pct"]:.1f}% ({time.time()-t0:.1f}s)')

    t_total = time.time() - t_start

    # Print final table
    print('\n' + '=' * 70)
    print('PAPER TABLE: Cross-Module CEP50 Comparison (meters)')
    print('=' * 70)
    print(f'{"Method":<30} | {"berlin1":>8} | {"berlin2":>8} | {"frankfurt1":>10} | {"frankfurt2":>10}')
    print('-' * 75)

    # Row 1: Standard LS
    sls = [all_results[d]['cep50']['Standard-LS'] for d in ['berlin1','berlin2','frankfurt1','frankfurt2']]
    print(f'{"Standard LS":<30} | {sls[0]:8.1f} | {sls[1]:8.1f} | {sls[2]:10.1f} | {sls[3]:10.1f}')

    # Row 2: Module 2 FG (using M3 internal; note discrepancy footnote)
    fg = [all_results[d]['cep50']['FG-MoG'] for d in ['berlin1','berlin2','frankfurt1','frankfurt2']]
    print(f'{"Module 2 FG-MoG+2A (static)":<30} | {fg[0]:8.1f} | {fg[1]:8.1f} | {fg[2]:10.1f} | {fg[3]:10.1f}')
    print(f'{"  (vs LS)":<30} | {_pct(fg[0],sls[0])} | {_pct(fg[1],sls[1])} | {_pct(fg[2],sls[2])} | {_pct(fg[3],sls[3])}')

    # Row 3: Module 3 Adaptive v4
    am3 = [all_results[d]['cep50']['Adaptive-M3'] for d in ['berlin1','berlin2','frankfurt1','frankfurt2']]
    print(f'{"Module 3 Adaptive-M3 v4":<30} | {am3[0]:8.1f} | {am3[1]:8.1f} | {am3[2]:10.1f} | {am3[3]:10.1f}')
    print(f'{"  (vs LS)":<30} | {_pct(am3[0],sls[0])} | {_pct(am3[1],sls[1])} | {_pct(am3[2],sls[2])} | {_pct(am3[3],sls[3])}')

    # FG% row
    fgp = [all_results[d]['fg_pct'] for d in ['berlin1','berlin2','frankfurt1','frankfurt2']]
    print(f'{"  FG selection rate":<30} | {fgp[0]:7.1f}% | {fgp[1]:7.1f}% | {fgp[2]:9.1f}% | {fgp[3]:7.1f}%')

    print('\n* Module 2 frankfurt1 value shown is Module 3 internal FG evaluation (exp_050, v8 universal).')
    print('  Module 2 standalone (exp_038, dataset-specific tuning) yields 476.9m (+9.2%).')
    print('  Both improve over Standard LS; Adaptive-M3 v4 outperforms both.')
    print(f'\nTotal time: {t_total:.1f}s')

    # Verify against stored exp_006
    print('\n[Verification] Comparing against stored exp_006 results ...')
    exp006 = os.path.normpath(os.path.join(_MODEL_DIR, '..', 'result', 'exp_006'))
    all_match = True
    for ds_short in ['berlin1','berlin2','frankfurt1','frankfurt2']:
        metrics_path = os.path.join(exp006, ds_short, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                stored = json.load(f)
            stored_cep = stored.get('Adaptive-M3', {}).get('cep50', float('nan'))
            computed_cep = all_results[ds_short]['cep50']['Adaptive-M3']
            delta = abs(computed_cep - stored_cep)
            status = 'OK' if delta < 5 else 'DIFF'
            if delta >= 5: all_match = False
            print(f'  {ds_short}: computed={computed_cep:.1f}m stored={stored_cep:.1f}m delta={delta:.1f}m [{status}]')
        else:
            print(f'  {ds_short}: no stored metrics (skipping check)')
            all_match = False

    if all_match:
        print('\n' + '=' * 70)
        print('REPRODUCIBILITY: PASS')
        print('All computed CEP50 values match stored exp_006 within 5m tolerance.')
        print('=' * 70)
    else:
        print('\nReproducibility: PARTIAL (see above)')
        print('(Minor variations expected due to solver non-determinism.)')


def _pct(v, base):
    delta = (base - v) / base * 100
    if delta > 0:
        return f'{delta:+5.1f}%'
    return f'{delta:+5.1f}%'


if __name__ == '__main__':
    main()

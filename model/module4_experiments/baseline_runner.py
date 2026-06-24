# module4_experiments/baseline_runner.py -- Run all methods on all datasets, collect metrics
# v2: Fixed SP3 integration (no clock correction) + MoG cache integration
import sys, os, json, time, pickle
import numpy as np
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here / 'common'))

# Import all methods
import module2_localization.standard_ls
import module2_localization.wls
import module2_localization.hard_threshold
import module2_localization.factor_graph
import module2_localization.cno_weighted
import module2_localization.snr_weighted
import module2_localization.raim
import module2_localization.irls
import module2_localization.kalman
import module2_localization.dnn
import module2_localization.gat_e2e
import module2_localization.ins_gnss
from module2_localization.factory import LocalizationFactory
from common.metrics import cep50, cep95, rmse, all_metrics
from common.coordinate import lla_to_ecef, ecef_to_lla
from common.sp3_reader import SP3Reader

# GNSS ID mapping
_GNSS_TO_SP3 = {'GPS': 'G', 'Glonass': 'R', 'Galileo': 'E', 'BeiDou': 'C'}
_SP3_CACHE = {}
_MOG_CACHE = {}

# MoG cache paths
_MOG_CACHE_DIR = r'D:\3_document\4_research\NLOS Signal Identification and Correction\model_2\part2_FactorGraphLocalizationFusion\cache'
_MOG_CACHE_MAP = {
    'berlin1_potsdamer_platz': 'berlin1_potsdamer_platz_mog_outputs_exp040.pkl',
    'berlin2_gendarmenmarkt': 'berlin2_gendarmenmarkt_mog_outputs_exp_049.pkl',
    'frankfurt1_maintower': 'frankfurt1_maintower_mog_outputs_exp_050.pkl',
    'frankfurt2_westendtower': 'frankfurt2_westendtower_mog_outputs_exp_051.pkl',
}


def _load_mog_cache(dataset_name):
    if dataset_name in _MOG_CACHE:
        return _MOG_CACHE[dataset_name]
    fn = _MOG_CACHE_MAP.get(dataset_name)
    if fn:
        path = os.path.join(_MOG_CACHE_DIR, fn)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                _MOG_CACHE[dataset_name] = pickle.load(f)
            return _MOG_CACHE[dataset_name]
    print(f'  [WARN] No MoG cache for {dataset_name}')
    _MOG_CACHE[dataset_name] = None
    return None


def _load_sp3(dataset_name):
    if dataset_name in _SP3_CACHE:
        return _SP3_CACHE[dataset_name]
    data_root = r'D:\3_document\4_research\NLOS Signal Identification and Correction\data\dataset'
    ds_dir = os.path.join(data_root, dataset_name)
    sp3_files = [f for f in os.listdir(ds_dir) if f.endswith('.sp3') and not f.endswith('.Z')]
    if not sp3_files:
        _SP3_CACHE[dataset_name] = None
        return None
    reader = SP3Reader(os.path.join(ds_dir, sp3_files[0]))
    _SP3_CACHE[dataset_name] = reader
    return reader


def _to_sp3_svid(gnss_name, svid):
    prefix = _GNSS_TO_SP3.get(str(gnss_name), 'G')
    return f'{prefix}{int(svid):02d}'


def compute_satellite_positions(ep, dataset_name, gt_ecef_km):
    reader = _load_sp3(dataset_name)
    gps_week = int(ep.gps_week) if hasattr(ep, 'gps_week') else 0
    gps_sec = float(ep.gps_seconds) if hasattr(ep, 'gps_seconds') else 0.0
    N = len(ep.observations)
    sv_positions = np.zeros((N, 3))
    for i, obs in enumerate(ep.observations):
        gnss = str(obs.gnss_id) if hasattr(obs, 'gnss_id') else 'GPS'
        svid = int(obs.sv_id) if hasattr(obs, 'sv_id') else 0
        sp3_svid = _to_sp3_svid(gnss, svid)
        pos_found = False
        if reader is not None and reader.has_satellite(sp3_svid):
            pos = reader.get_satellite_position(gps_week, gps_sec, sp3_svid)
            if pos is not None:
                sv_positions[i] = np.array(pos) / 1000.0
                pos_found = True
        if not pos_found:
            el, az = np.radians(float(obs.elevation)), np.radians(float(obs.azimuth))
            e_unit = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
            pr_km = float(obs.pr_mes) / 1000.0
            lat, lon = ecef_to_lla(gt_ecef_km[0], gt_ecef_km[1], gt_ecef_km[2])[:2]
            lat_r, lon_r = np.radians(lat), np.radians(lon)
            R = np.array([
                [-np.sin(lon_r), -np.sin(lat_r)*np.cos(lon_r), np.cos(lat_r)*np.cos(lon_r)],
                [np.cos(lon_r), -np.sin(lat_r)*np.sin(lon_r), np.cos(lat_r)*np.sin(lon_r)],
                [0, np.cos(lat_r), np.sin(lat_r)],
            ])
            sv_positions[i] = gt_ecef_km + R @ e_unit * pr_km
    return sv_positions


def load_epoch_data(dataset_name):
    processed_dir = r'D:\3_document\4_research\NLOS Signal Identification and Correction\data\processedData'
    pkl_path = os.path.join(processed_dir, f'{dataset_name}_processed.pkl')
    with open(pkl_path, 'rb') as f:
        ep_list = pickle.load(f)
    return ep_list


def run_baseline(dataset_name, methods=None, n_epochs=None, use_sp3=True):
    ep_list = load_epoch_data(dataset_name)
    mog_cache = _load_mog_cache(dataset_name)
    if n_epochs:
        ep_list = ep_list[:n_epochs]
    n_epochs_actual = len(ep_list)
    if mog_cache and len(mog_cache) < n_epochs_actual:
        n_epochs_actual = len(mog_cache)
        ep_list = ep_list[:n_epochs_actual]

    all_methods = LocalizationFactory.list_methods()
    if methods == 'all' or methods is None:
        methods = all_methods
    else:
        methods = [m for m in methods if m in all_methods]

    if use_sp3:
        _load_sp3(dataset_name)

    results = {}
    for method_name in methods:
        errors_3d = []
        errors_h = []
        method = LocalizationFactory.create(method_name)
        for ei, ep in enumerate(ep_list):
            obs = np.array([float(o.pr_mes) / 1000.0 for o in ep.observations])
            gt = lla_to_ecef(float(ep.gt_lat), float(ep.gt_lon), float(ep.gt_height))
            sv_positions = compute_satellite_positions(ep, dataset_name, gt)
            additional_info = {}
            if hasattr(ep.observations[0], 'elevation'):
                additional_info['elevation_deg'] = np.array([float(o.elevation) for o in ep.observations])
            if hasattr(ep.observations[0], 'cno'):
                additional_info['cno'] = np.array([float(o.cno) for o in ep.observations])
            if hasattr(ep.observations[0], 'nlos_label'):
                additional_info['nlos_label'] = np.array([int(o.nlos_label) for o in ep.observations])
            # MoG outputs from cache
            if mog_cache and ei < len(mog_cache):
                mog = mog_cache[ei]
                additional_info['p_los'] = np.asarray(mog['p_los_sharp'], dtype=np.float64)
                additional_info['sigma_los'] = np.asarray(mog['sigma_los'], dtype=np.float64)
                additional_info['sigma_nlos'] = np.asarray(mog['sigma_nlos'], dtype=np.float64)
                additional_info['mu_nlos'] = np.asarray(mog['mu_nlos'], dtype=np.float64)
            try:
                pos, clk, details = method.solve(obs, sv_positions, additional_info=additional_info)
                err_3d = float(np.linalg.norm(pos - gt))
                errors_3d.append(err_3d)
                err_h = float(np.sqrt((pos[0]-gt[0])**2 + (pos[1]-gt[1])**2))
                errors_h.append(err_h)
            except Exception as e:
                errors_3d.append(np.nan)
                errors_h.append(np.nan)

        errors_3d = np.array(errors_3d)
        errors_h = np.array(errors_h)
        valid_3d = errors_3d[~np.isnan(errors_3d)]
        valid_h = errors_h[~np.isnan(errors_h)]
        metrics_3d = all_metrics(valid_3d) if len(valid_3d) > 0 else {}
        metrics_3d['n_valid'] = int(len(valid_3d))
        metrics_3d['n_total'] = int(len(errors_3d))
        metrics_3d['failure_rate'] = float(np.sum(np.isnan(errors_3d)) / max(len(errors_3d), 1))
        if len(valid_h) > 0:
            metrics_3d['cep50_h'] = float(np.percentile(valid_h, 50))
            metrics_3d['cep95_h'] = float(np.percentile(valid_h, 95))
        results[method_name] = metrics_3d
    return results


def run_all_datasets(datasets=None, methods=None, output_dir='results/baseline', n_epochs=None):
    if datasets is None:
        datasets = [
            'berlin1_potsdamer_platz', 'berlin2_gendarmenmarkt',
            'frankfurt1_maintower', 'frankfurt2_westendtower'
        ]
    all_methods = LocalizationFactory.list_methods()
    if methods == 'all' or methods is None:
        methods = all_methods
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    for ds in datasets:
        print('\n' + '=' * 60)
        print(f'Running baseline on {ds}...')
        sep = "=" * 60
        print(sep)
        t0 = time.time()
        results = run_baseline(ds, methods=methods, n_epochs=n_epochs, use_sp3=True)
        elapsed = time.time() - t0
        all_results[ds] = results
        print(f'  Completed in {elapsed:.1f}s ({len(results)} methods)')
        sorted_methods = sorted(results.items(), key=lambda x: x[1].get('cep50', 999))
        for name, m in sorted_methods[:6]:
            cep = m.get('cep50', 999)
            cep95_val = m.get("cep95", 999)
            print(f'    {name:20s}: CEP50={cep:.4f} km, CEP95={cep95_val:.4f} km')
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    generate_comparison_table(all_results, output_dir)
    save_condition(output_dir, datasets, methods, n_epochs)
    return all_results


def generate_comparison_table(all_results, output_dir):
    lines = ['# Baseline Comparison -- CEP50 (km)', '']
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    lines.append(f'**Generated**: {ts}')
    lines.append('')
    datasets = list(all_results.keys())
    methods = sorted(set().union(*[set(r.keys()) for r in all_results.values()]))
    avg_cep = {}
    for m in methods:
        vals = [all_results[ds].get(m, {}).get('cep50', np.nan) for ds in datasets]
        avg_cep[m] = np.nanmean(vals)
    methods = sorted(methods, key=lambda m: avg_cep.get(m, 999))
    short_names = [d.replace('_', ' ').split()[0][:8] for d in datasets]
    header = '| Method | ' + ' | '.join(short_names) + ' | Avg |'
    lines.append(header)
    sep = '|' + '|'.join([' --- ' for _ in range(len(datasets) + 2)]) + '|'
    lines.append(sep)
    for method in methods:
        vals = []
        for ds in datasets:
            cep = all_results[ds].get(method, {}).get('cep50', None)
            vals.append(f'{cep:.4f}' if cep is not None else 'N/A')
        avg = avg_cep.get(method, 999)
        vals.append(f'{avg:.4f}')
        lines.append('| ' + method.ljust(20) + ' | ' + ' | '.join(vals) + ' |')
    table = '\n'.join(lines)
    table_path = os.path.join(output_dir, 'comparison_table.md')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(table)
    print(f'\nComparison table saved to {table_path}')
    csv_lines = ['method,' + ','.join(datasets)]
    for method in methods:
        vals = [str(all_results[ds].get(method, {}).get('cep50', '')) for ds in datasets]
        csv_lines.append(method + ',' + ','.join(vals))
    csv_path = os.path.join(output_dir, 'comparison_table.csv')
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f'CSV saved to {csv_path}')


def save_condition(output_dir, datasets, methods, n_epochs):
    lines = ['# Experiment Condition', '']
    ts2 = time.strftime('%Y-%m-%d %H:%M:%S')
    lines.append(f'- **Timestamp**: {ts2}')
    lines.append(f'- **Datasets**: {datasets}')
    lines.append(f'- **Methods**: {methods}')
    epoch_limit_str = n_epochs if n_epochs else 'all'
    lines.append(f'- **Epoch limit**: {epoch_limit_str}')
    lines.append(f'- **SP3**: enabled')
    with open(os.path.join(output_dir, 'condition.md'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='berlin1_potsdamer_platz')
    parser.add_argument('--methods', type=str, default='all')
    parser.add_argument('--output', type=str, default='results/baseline')
    parser.add_argument('--n_epochs', type=int, default=None)
    args = parser.parse_args()
    datasets = args.datasets.split(',')
    methods = args.methods.split(',') if args.methods != 'all' else 'all'
    run_all_datasets(datasets=datasets, methods=methods, output_dir=args.output, n_epochs=args.n_epochs)

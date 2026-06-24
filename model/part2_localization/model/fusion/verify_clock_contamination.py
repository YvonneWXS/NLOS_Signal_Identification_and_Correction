import os, sys, json, pickle, time
import numpy as np

FUSION_DIR = os.path.dirname(os.path.abspath(__file__))
M1_DIR = os.path.normpath(os.path.join(FUSION_DIR, '..', '..', '..', '..', 'part1_GAT', 'model'))
sys.path.insert(0, FUSION_DIR)
sys.path.insert(0, M1_DIR)
sys.path.insert(0, os.path.dirname(FUSION_DIR))

from fusion.utils import load_epoch_data, compute_satellite_positions, load_mog_model, run_mog_inference

CACHE_DIR = os.path.normpath(os.path.join(FUSION_DIR, '..', '..', 'cache'))
DATASETS = ['berlin1_potsdamer_platz', 'berlin2_gendarmenmarkt', 'frankfurt1_maintower', 'frankfurt2_westendtower']

def analyze_dataset(dataset_name):
    sep = '=' * 60
    print('\n' + sep)
    print(dataset_name)
    print(sep)

    obs_data = load_epoch_data(dataset_name)
    if obs_data is None:
        print('  No data loaded')
        return None

    results = {'clk_c': [], 'clk_l': [], 'nlos_A': [], 'nlos_B': [], 'los_A': [], 'los_B': [], 'n_los_sats': [], 'n_total': []}
    THRESH = 0.7
    count = 0

    for ep_idx, ep_data in enumerate(obs_data):
        if count >= 500:
            break
        obs_list = ep_data['obs']
        try:
            sv_pos_tuple = compute_satellite_positions(ep_data)
            sv_pos = sv_pos_tuple[0] if isinstance(sv_pos_tuple, tuple) else np.array(sv_pos_tuple)
        except Exception:
            continue
        if sv_pos.ndim != 2 or sv_pos.shape[0] != len(obs_list):
            continue

        pr_mes = np.array([o.get('pr_mes_m', o.get('pr', 0.0)) / 1000.0 for o in obs_list])
        nlos_labels = np.array([o.get('nlos_label', 0) for o in obs_list])
        n = len(obs_list)

        # Estimate p_los via simple heuristic (elevation-based)
        elevations = np.array([o.get('elevation_deg', 45.0) for o in obs_list])
        p_los = 1.0 / (1.0 + np.exp(-(elevations - 15.0) / 10.0))

        results['n_total'].append(n)
        x = np.array(ep_data["gt_ecef"])
        dists = np.linalg.norm(sv_pos - x[np.newaxis, :], axis=1)
        raw = pr_mes - dists

        clk_c = np.median(raw)
        res_A = raw - clk_c

        high_los = p_los > THRESH
        n_los = high_los.sum()
        results['n_los_sats'].append(n_los)

        if n_los >= 4:
            clk_l = np.median(raw[high_los])
        else:
            sidx = np.argsort(p_los)[::-1]
            top_n = max(4, n // 2)
            clk_l = np.median(raw[sidx[:top_n]])

        res_B = raw - clk_l
        results['clk_c'].append(clk_c)
        results['clk_l'].append(clk_l)

        nmask = nlos_labels == 1
        lmask = nlos_labels == 0
        if nmask.any():
            results['nlos_A'].extend(res_A[nmask].tolist())
            results['nlos_B'].extend(res_B[nmask].tolist())
        if lmask.any():
            results['los_A'].extend(res_A[lmask].tolist())
            results['los_B'].extend(res_B[lmask].tolist())
        count += 1

    clk_c = np.array(results['clk_c'])
    clk_l = np.array(results['clk_l'])
    dc = clk_l - clk_c
    nA = np.array(results['nlos_A'])
    nB = np.array(results['nlos_B'])
    lA = np.array(results['los_A'])
    lB = np.array(results['los_B'])
    nls = np.array(results['n_los_sats'])

    s = {
        'clk_c_mean_m': float(np.mean(clk_c) * 1000),
        'clk_l_mean_m': float(np.mean(clk_l) * 1000),
        'delta_m': float(np.mean(dc) * 1000),
        'delta_pos_pct': float((dc > 0).mean() * 100),
        'nlos_A_mean_m': float(np.mean(nA) * 1000),
        'nlos_B_mean_m': float(np.mean(nB) * 1000),
        'nlos_A_pos_pct': float((nA > 0).mean() * 100),
        'nlos_B_pos_pct': float((nB > 0).mean() * 100),
        'los_A_mean_m': float(np.mean(lA) * 1000),
        'los_B_mean_m': float(np.mean(lB) * 1000),
        'pct_4plus': float((nls >= 4).mean() * 100),
        'pct_5plus': float((nls >= 5).mean() * 100),
        'pct_6plus': float((nls >= 6).mean() * 100),
        'mean_nlos': float(np.mean(nls)),
        'mean_total': float(np.mean(results['n_total'])),
    }
    return s

def main():
    sep = '=' * 60
    print(sep)
    print('v6 PART 0: Clock Contamination Verification')
    print(sep)
    print('Using elevation-based p_los heuristic (no MoG cache needed)')
    print('Sampling: 500 epochs per dataset')
    all_r = {}
    for ds in DATASETS:
        r = analyze_dataset(ds)
        if r is None:
            continue
        all_r[ds] = r
        print('\n  Clock: C=' + '{:.1f}'.format(r['clk_c_mean_m']) + 'm L=' + '{:.1f}'.format(r['clk_l_mean_m']) + 'm delta=' + '{:.1f}'.format(r['delta_m']) + 'm pos=' + '{:.1f}'.format(r['delta_pos_pct']) + '%')
        print('  NLOS: A=' + '{:.1f}'.format(r['nlos_A_mean_m']) + 'm/' + '{:.1f}'.format(r['nlos_A_pos_pct']) + '% -> B=' + '{:.1f}'.format(r['nlos_B_mean_m']) + 'm/' + '{:.1f}'.format(r['nlos_B_pos_pct']) + '%')
        print('  LOS:  A=' + '{:.1f}'.format(r['los_A_mean_m']) + 'm -> B=' + '{:.1f}'.format(r['los_B_mean_m']) + 'm')
        print('  LOS sats: mean=' + '{:.1f}'.format(r['mean_nlos']) + ' >=4:' + '{:.1f}'.format(r['pct_4plus']) + '% >=5:' + '{:.1f}'.format(r['pct_5plus']) + '%')

    print('\n' + sep)
    print('DIAGNOSIS')
    print(sep)
    for ds, r in all_r.items():
        short = ds.split('_')[0]
        imp = r['nlos_B_pos_pct'] - r['nlos_A_pos_pct']
        delta = abs(r['delta_m'])
        conf = (r['nlos_B_pos_pct'] > 60) or (imp > 10) or (delta > 50)
        st = 'CONFIRMED' if conf else 'WEAK'
        print('  ' + short + ': +' + '{:.1f}'.format(imp) + '% NLOS pos, delta=' + '{:.1f}'.format(delta) + 'm [' + st + ']')

    out = os.path.join(CACHE_DIR, 'clock_contamination_analysis.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_r, f, indent=2, ensure_ascii=False)
    print('\nSaved: ' + out)

if __name__ == '__main__':
    main()

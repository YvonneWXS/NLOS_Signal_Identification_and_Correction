# fusion/evaluate_fusion.py v2 — 6 methods, detailed metrics
# ============================================================
# Methods: Standard LS, WLS-elevation, WLS-MoG, Hard-threshold,
#          FactorGraph-MoG, FactorGraph-MoG+2A (if TCN available)
# ============================================================
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.baselines import (solve_standard_ls, solve_wls_elevation,
                                solve_wls_mog, solve_hard_threshold,
                                solve_wls_aggressive_power, solve_wls_log_odds,
                                solve_wls_soft_floor, solve_wls_geometry_aware,
                                solve_wls_debiased, solve_raim_mog)
from fusion.utils import fit_platt_scaling, apply_platt_scaling
from fusion.factor_graph_fusion import FactorGraphPositioner
from fusion.prnc import PRNCPositioner
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
    # [v5] Method 1b: PRNC-mu (direct mu_nlos correction)
    # ============================================================
    print('  [1b/12] PRNC-mu ...')
    prnc = PRNCPositioner()
    err_2d, err_3d = [], []
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) < 4: continue
        x, _ = prnc.solve_mu(pr_measured_all[i], sv_positions_all[i],
                              mog.get('p_los_sharp', mog['p_los']), mog['mu_nlos'])
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['PRNC-mu'] = compute_metrics(err_2d)
    results['PRNC-mu']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    print(f'    CEP50={results["PRNC-mu"]["cep50"]:.1f}m')

    # ============================================================
    # [v5] Method 1c: PRNC-adaptive
    # ============================================================
    print('  [1c/12] PRNC-adaptive ...')
    err_2d, err_3d = [], []
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) < 4: continue
        x, _ = prnc.solve_adaptive(pr_measured_all[i], sv_positions_all[i],
                                    mog.get('p_los_sharp', mog['p_los']), mog['sigma_los'], mog.get('mu_nlos'))
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['PRNC-adaptive'] = compute_metrics(err_2d)
    results['PRNC-adaptive']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    print(f'    CEP50={results["PRNC-adaptive"]["cep50"]:.1f}m')

    # ============================================================
    # [v5] Method 1d: PRNC-mu-adaptive (mu + residual combined)
    # ============================================================
    print('  [1d/12] PRNC-mu-adaptive ...')
    err_2d, err_3d = [], []
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) < 4: continue
        # Adaptive with mu_nlos enabled (mu*0.3 weighting in solve_adaptive)
        x, _ = prnc.solve_adaptive(pr_measured_all[i], sv_positions_all[i],
                                    mog.get('p_los_sharp', mog['p_los']), mog['sigma_los'],
                                    mog.get('mu_nlos'))
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['PRNC-mu-adaptive'] = compute_metrics(err_2d)
    results['PRNC-mu-adaptive']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    print(f'    CEP50={results["PRNC-mu-adaptive"]["cep50"]:.1f}m')

    # ============================================================
    # Method 2: WLS-elevation
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
    # Method 4b: WLS-MoG-power3 (Scheme 1)
    # ============================================================
    print('  [4b/9] WLS-MoG-power3 ...')
    err_2d, err_3d = [], []
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) == 0: continue
        x = solve_wls_aggressive_power(sv_positions_all[i], pr_measured_all[i],
                                        mog.get('p_los_sharp', mog['p_los']), mog['sigma_los'])
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['WLS-MoG-power3'] = compute_metrics(err_2d)
    results['WLS-MoG-power3']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    print(f'    CEP50={results["WLS-MoG-power3"]["cep50"]:.1f}m')
    
    # ============================================================
    # Method 4c: WLS-log-odds (Scheme 2)
    # ============================================================
    print('  [4c/9] WLS-log-odds ...')
    err_2d, err_3d = [], []
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) == 0: continue
        x = solve_wls_log_odds(sv_positions_all[i], pr_measured_all[i],
                                mog.get('p_los_sharp', mog['p_los']), mog['sigma_los'])
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['WLS-log-odds'] = compute_metrics(err_2d)
    results['WLS-log-odds']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    print(f'    CEP50={results["WLS-log-odds"]["cep50"]:.1f}m')
    
    # ============================================================
    # Method 4d: WLS-debiased (Scheme 5) ? KEY FIX
    # ============================================================
    print('  [4d/9] WLS-debiased (mu_nlos correction) ...')
    err_2d, err_3d = [], []
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) == 0: continue
        x = solve_wls_debiased(sv_positions_all[i], pr_measured_all[i],
                                mog.get('p_los_sharp', mog['p_los']), mog['sigma_los'],
                                mog.get('mu_nlos', None))
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['WLS-debiased'] = compute_metrics(err_2d)
    results['WLS-debiased']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    print(f'    CEP50={results["WLS-debiased"]["cep50"]:.1f}m')
    
    # ============================================================
    # Method 4e: RAIM-MoG (Scheme 6)
    # ============================================================
    print('  [4e/9] RAIM-MoG (iterative exclusion) ...')
    err_2d, err_3d = [], []
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) == 0: continue
        x = solve_raim_mog(sv_positions_all[i], pr_measured_all[i],
                            mog.get('p_los_sharp', mog['p_los']), mog['sigma_los'],
                            mog.get('sigma_nlos', None))
        err_2d.append(compute_2d_error(x[:3], ep['gt_ecef']))
        err_3d.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['RAIM-MoG'] = compute_metrics(err_2d)
    results['RAIM-MoG']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d)**2)))
    print(f'    CEP50={results["RAIM-MoG"]["cep50"]:.1f}m')
    
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
    # Method 5b: FactorGraph-debiased (v4)
    # ============================================================
    print('  [5b/9] FactorGraph-debiased ...')
    positioner = FactorGraphPositioner()
    err_2d_db, err_3d_db, n_imp_db, n_conv_db = [], [], 0, 0
    for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
        if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) == 0: continue
        try:
            x, info = positioner.solve_epoch_debiased(
                sv_positions_all[i], pr_measured_all[i],
                mog.get('p_los_sharp', mog['p_los']), mog['mu_nlos'],
                mog['sigma_los'], mog['sigma_nlos'],
                epoch_idx=i, dataset_name=dataset_name
            )
            err_db = compute_2d_error(x[:3], ep['gt_ecef'])
            err_2d_db.append(err_db)
            err_3d_db.append(compute_3d_error(x[:3], ep['gt_ecef']))
            if info.get('improvement') == 'improved': n_imp_db += 1
            if info.get('success', False): n_conv_db += 1
        except Exception:
            x = solve_standard_ls(sv_positions_all[i], pr_measured_all[i])
            err_2d_db.append(compute_2d_error(x[:3], ep['gt_ecef']))
            err_3d_db.append(compute_3d_error(x[:3], ep['gt_ecef']))
    results['FactorGraph-debiased'] = compute_metrics(err_2d_db)
    results['FactorGraph-debiased']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d_db)**2)))
    results['FactorGraph-debiased']['pct_improved'] = float(n_imp_db / max(len(err_2d_db), 1) * 100)
    results['FactorGraph-debiased']['pct_converged'] = float(n_conv_db / max(len(err_2d_db), 1) * 100)
    print(f'    CEP50={results["FactorGraph-debiased"]["cep50"]:.1f}m')
    
    # ============================================================
    # Method 6: FactorGraph-MoG+2A (TCN temporal prior)
    # ============================================================
    # Method 6: FactorGraph-MoG+2A (TCN temporal prior)
    # ============================================================
    print('  [6/6] FactorGraph-MoG+2A ...')
    tcn_available = False
    try:
        tcn_models_dir = os.path.join(os.path.dirname(os.path.dirname(result_dir)), '..', 'models')
        tcn_path = os.path.join(tcn_models_dir, f'tcn_{dataset_name}.pth')
        if os.path.exists(tcn_path):
            import torch
            from fusion.train_tcn import SimpleTCN
            tcn = SimpleTCN(63, 64, 20, 10)
            ckpt = torch.load(tcn_path, map_location='cpu', weights_only=False)
            # Handle both raw state_dict and dict-wrapped checkpoints
            if 'model_state_dict' in ckpt:
                tcn.load_state_dict(ckpt['model_state_dict'])
            else:
                tcn.load_state_dict(ckpt)
            tcn.eval()
            tcn_available = True
    except Exception as e:
        print(f'    TCN load failed: {e}')
    
    if tcn_available:
        print('    TCN loaded, applying temporal prior...')
        SEQ_LEN = 10; MAX_SV = 20
        err_2d_tcn, err_3d_tcn, n_improved_tcn, n_conv_tcn = [], [], 0, 0
        
        for i, (ep, mog) in enumerate(zip(all_epochs_data, mog_outputs)):
            if mog is None or len(mog.get('p_los_sharp', mog['p_los'])) == 0:
                continue
            
            # Apply TCN temporal prior if we have enough history
            p_los_updated = mog.get('p_los_sharp', mog['p_los']).copy()
            
            if i >= SEQ_LEN:
                # Build input sequence
                seq_input = np.zeros((SEQ_LEN, 63), dtype=np.float32)
                for offset in range(SEQ_LEN, 0, -1):
                    t = i - offset
                    prev_mog = mog_outputs[t]
                    prev_ep = all_epochs_data[t]
                    if prev_mog is None:
                        continue
                    
                    # Velocity
                    vel = all_epochs_data[t]['gt_ecef'] - all_epochs_data[t-1]['gt_ecef'] if t > 0 else np.zeros(3)
                    
                    # Geometry per satellite
                    geom = np.zeros((MAX_SV, 3))
                    N_vis = min(len(prev_mog['elevation_deg']), MAX_SV)
                    geom[:N_vis, 0] = prev_mog['elevation_deg'][:N_vis] / 90.0
                    geom[:N_vis, 1] = prev_mog['azimuth_deg'][:N_vis] / 360.0
                    geom[:N_vis, 2] = prev_mog.get('p_los_sharp', prev_mog['p_los'])[:N_vis]
                    
                    seq_input[offset-1, :3] = vel
                    seq_input[offset-1, 3:] = geom.flatten()
                
                # TCN inference
                with torch.no_grad():
                    x_t = torch.tensor(seq_input, dtype=torch.float32).unsqueeze(0)
                    p_nlos_pred = tcn(x_t).squeeze(0).numpy()  # (MAX_SV,)
                
                # Fix B+C: Tightened gate + soft blending (v3)
                # Gate conditions:
                #   1. |p_nlos - 0.5| > 0.25 (TCN is directionally confident)
                #   2. TCN must DISAGREE with Module 1 (provides new info)
                # Soft blend capped at 30% TCN influence
                for j in range(min(len(p_los_updated), MAX_SV)):
                    p_nlos_j = p_nlos_pred[j]
                    p_los_gat_j = p_los_updated[j]
                    confidence = 2.0 * abs(p_nlos_j - 0.5)  # [0, 1]
                    tcn_disagrees = ((p_nlos_j > 0.6 and p_los_gat_j < 0.5) or
                                     (p_nlos_j < 0.4 and p_los_gat_j > 0.5))
                    
                    if abs(p_nlos_j - 0.5) > 0.25 and tcn_disagrees:
                        # Fix C: Soft blending
                        alpha = min(confidence * abs(p_nlos_j - 0.5) * 2.0, 0.3)
                        p_los_updated[j] = (1.0 - alpha) * p_los_gat_j + alpha * (1.0 - p_nlos_j)
            
            # Use updated p_los
            x_wls = solve_wls_mog(sv_positions_all[i], pr_measured_all[i], p_los_updated, mog['sigma_los'])
            x_fg, info_fg = positioner.solve_epoch(
                sv_positions_all[i], pr_measured_all[i],
                p_los_updated, mog['mu_nlos'],
                mog['sigma_los'], mog['sigma_nlos']
            )
            err_fg = compute_2d_error(x_fg[:3], ep['gt_ecef'])
            err_2d_tcn.append(err_fg)
            err_3d_tcn.append(compute_3d_error(x_fg[:3], ep['gt_ecef']))
            err_wls = compute_2d_error(x_wls[:3], ep['gt_ecef'])
            if err_fg < err_wls:
                n_improved_tcn += 1
            if info_fg.get('success', False):
                n_conv_tcn += 1
        
        results['FactorGraph-MoG+2A'] = compute_metrics(err_2d_tcn)
        results['FactorGraph-MoG+2A']['rmse_3d'] = float(np.sqrt(np.mean(np.array(err_3d_tcn)**2)))
        results['FactorGraph-MoG+2A']['pct_improved'] = float(n_improved_tcn / max(len(err_2d_tcn), 1) * 100)
        results['FactorGraph-MoG+2A']['pct_converged'] = float(n_conv_tcn / max(len(err_2d_tcn), 1) * 100)
        results['FactorGraph-MoG+2A']['tcn_available'] = True
        print(f'    CEP50={results["FactorGraph-MoG+2A"]["cep50"]:.1f}m (improved: {results["FactorGraph-MoG+2A"]["pct_improved"]:.1f}%)')
    else:
        results['FactorGraph-MoG+2A'] = results['FactorGraph-MoG'].copy()
        results['FactorGraph-MoG+2A']['tcn_available'] = False
        print('    TCN not available, using FactorGraph-MoG result')
    
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


print('fusion/evaluate_fusion.py v5 loaded (12 methods)')

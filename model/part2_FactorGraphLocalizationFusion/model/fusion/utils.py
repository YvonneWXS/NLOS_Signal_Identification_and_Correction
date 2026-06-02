# fusion/utils.py v4 — Coordinate transforms, data loading (processed pickle),
# Module 1 inference, and SP3-based satellite position computation.
# Fix: correct feature extraction matching training (11 features, GNSS one-hot)

import sys, os, pickle
import numpy as np

_MODULE1_PATH = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model"
if _MODULE1_PATH not in sys.path:
    sys.path.insert(0, _MODULE1_PATH)

# WGS84 constants
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2


def lla_to_ecef(lat_deg, lon_deg, height_m):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat); cos_lat = np.cos(lat)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    x = (N + height_m) * cos_lat * np.cos(lon)
    y = (N + height_m) * cos_lat * np.sin(lon)
    z = (N * (1.0 - _WGS84_E2) + height_m) * sin_lat
    return np.array([x, y, z]) / 1000.0


def ecef_to_lla(x_km, y_km, z_km):
    x, y, z = x_km * 1000.0, y_km * 1000.0, z_km * 1000.0
    lon = np.arctan2(y, x); p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(5):
        sin_lat = np.sin(lat); N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - _WGS84_E2 * N / (N + h)))
    sin_lat = np.sin(lat); N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    h = p / np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), h


_GNSS_ID_MAP = {0: 'GPS', 1: 'GPS', 2: 'Galileo', 3: 'Glonass', 4: 'BeiDou', 5: 'QZSS', 6: 'Galileo'}
_GNSS_TO_SP3 = {'GPS': 'G', 'Glonass': 'R', 'Galileo': 'E', 'BeiDou': 'C'}
_GNSS_TO_INDEX = {'GPS': 7, 'Glonass': 8, 'Galileo': 9, 'BeiDou': 10}
_SP3_CACHE = {}


def _to_sp3_svid(gnss_name, svid):
    prefix = _GNSS_TO_SP3.get(gnss_name, 'G')
    return f'{prefix}{svid:02d}'


# ============================================================
# Data loading
# ============================================================

def load_epoch_data(dataset_name):
    processed_dir = r"D:\3_document\4_research\NLOS Signal Identification and Correction\data\processedData"
    pkl_path = os.path.join(processed_dir, f'{dataset_name}_processed.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Processed data not found: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        ep_list = pickle.load(f)
    result = []
    for ep in ep_list:
        gt_ecef = lla_to_ecef(ep.gt_lat, ep.gt_lon, ep.gt_height)
        obs_list = []
        for obs in ep.observations:
            obs_list.append({
                'svid': obs.sv_id if hasattr(obs, 'sv_id') else 0,
                'gnss': str(obs.gnss_id) if hasattr(obs, 'gnss_id') else 'GPS',
                'pr_mes_m': obs.pr_mes,
                'cno': obs.cno if hasattr(obs, 'cno') else 0.0,
                'pr_stdev_m': obs.pr_stdev if hasattr(obs, 'pr_stdev') else 0.0,
                'nlos_label': obs.nlos_label,
                'elevation_deg': obs.elevation,
                'azimuth_deg': obs.azimuth,
            })
        result.append({
            'gps_week': ep.gps_week,
            'gps_seconds': ep.gps_seconds,
            'gt_ecef': gt_ecef,
            'obs': obs_list,
        })
    return result


# ============================================================
# Satellite position from SP3
# ============================================================

def _load_sp3(dataset_name):
    if dataset_name in _SP3_CACHE:
        return _SP3_CACHE[dataset_name]
    from sp3_reader import SP3Reader
    data_root = r"D:\3_document\4_research\NLOS Signal Identification and Correction\data\dataset"
    ds_dir = os.path.join(data_root, dataset_name)
    sp3_files = [f for f in os.listdir(ds_dir) if f.endswith('.sp3') and not f.endswith('.Z')]
    if not sp3_files:
        print(f"  [WARN] No .sp3 file in {ds_dir}")
        _SP3_CACHE[dataset_name] = None
        return None
    sp3_path = os.path.join(ds_dir, sp3_files[0])
    reader = SP3Reader(sp3_path)
    _SP3_CACHE[dataset_name] = reader
    print(f"  SP3: {sp3_files[0]} ({reader.get_statistics()['total_satellites']} sats)")
    return reader


def compute_satellite_positions(epoch_data, dataset_name=None):
    gps_week = epoch_data['gps_week']
    gps_sec = epoch_data['gps_seconds']
    obs_list = epoch_data['obs']
    N = len(obs_list)
    sv_positions = np.zeros((N, 3))
    sv_clock_m = np.zeros(N)
    reader = _load_sp3(dataset_name) if dataset_name else None
    for i, obs in enumerate(obs_list):
        sp3_svid = _to_sp3_svid(obs['gnss'], obs['svid'])
        if reader is not None and reader.has_satellite(sp3_svid):
            pos = reader.get_satellite_position(gps_week, gps_sec, sp3_svid)
            clk = reader.get_satellite_clock(gps_week, gps_sec, sp3_svid)
            if pos is not None:
                sv_positions[i] = np.array(pos) / 1000.0
                if clk is not None:
                    sv_clock_m[i] = clk
                continue
        # Fallback: geometric approximation
        el = np.deg2rad(obs['elevation_deg']); az = np.deg2rad(obs['azimuth_deg'])
        e_unit = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
        lat, lon = ecef_to_lla(epoch_data['gt_ecef'][0],
                               epoch_data['gt_ecef'][1],
                               epoch_data['gt_ecef'][2])[:2]
        lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
        R = np.array([
            [-np.sin(lon_r), -np.sin(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.cos(lon_r)],
            [np.cos(lon_r), -np.sin(lat_r) * np.sin(lon_r), np.cos(lat_r) * np.sin(lon_r)],
            [0, np.cos(lat_r), np.sin(lat_r)],
        ])
        sv_positions[i] = epoch_data['gt_ecef'] + R @ e_unit * (obs['pr_mes_m'] / 1000.0)
    return sv_positions, sv_clock_m


# ============================================================
# Module 1 inference — FIXED feature extraction
# ============================================================

def load_mog_model(exp_name):
    import torch
    from config import get_config
    from GAT_V2025 import NLOSGAT
    config = get_config()
    device = config.get_device()
    model = NLOSGAT(
        in_features=config.IN_FEATURES, hidden_features=config.HIDDEN_FEATURES,
        num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT,
    ).to(device)
    result_dir = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result"
    best_path = os.path.join(result_dir, exp_name, 'best_model.pth')
    if not os.path.exists(best_path):
        best_path = os.path.join(result_dir, exp_name, 'final_model.pth')
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config, device


def run_mog_inference(model, config, device, epoch_data, calib_params=None):
    import torch
    obs_list = epoch_data['obs']
    N = len(obs_list)
    if N == 0:
        return None
    # Match NodeFeature_Generate.py extract_node_features exactly
    features = np.zeros((N, 11), dtype=np.float32)
    for i, obs in enumerate(obs_list):
        features[i, 0] = obs['elevation_deg'] / 90.0        # elevation normalized
        features[i, 1] = obs['azimuth_deg'] / 360.0         # azimuth normalized
        features[i, 2] = obs.get('cno', 0.0) / 60.0         # CNO normalized (÷60 dBHz)
        features[i, 3] = obs.get('pr_stdev_m', 0.0) / 5.0   # prStdev normalized (÷5 m)
        features[i, 4] = obs['pr_mes_m'] / 3e7              # pseudorange scaled
        features[i, 5] = 0.0                                # pseudorange_error — unknown at inference, set neutral
        features[i, 6] = np.cos(np.radians(obs['elevation_deg']))  # cos(elevation)
        # GNSS one-hot (features 7-10)
        gnss_col = _GNSS_TO_INDEX.get(obs.get('gnss', 'GPS'), -1)
        if 7 <= gnss_col <= 10:
            features[i, gnss_col] = 1.0
    node_features = torch.tensor(features, device=device)
    # Edge index: fully connected graph (same as training)
    edge_list = [[i, j] for i in range(N) for j in range(N) if i != j]
    if edge_list:
        edge_index = torch.tensor(edge_list, device=device).t().contiguous()
    else:
        edge_index = torch.tensor([[0], [0]], device=device)
    with torch.no_grad():
        p_los, mu_nlos, log_sigma_los, log_sigma_nlos = model(node_features, edge_index)
    
    # P0.2: Platt scaling for p_los calibration (replaces fixed T=0.6)
    p_raw = p_los.squeeze().cpu().numpy()
    if calib_params is not None and 'A' in calib_params:
        p_sharp = apply_platt_scaling(p_raw, calib_params)
    else:
        # Fallback: mild temperature scaling
        eps = 1e-8
        logit = np.log(np.clip(p_raw, eps, 1 - eps) / np.clip(1 - p_raw, eps, 1 - eps))
        p_sharp = 1.0 / (1.0 + np.exp(-logit / 0.8))
    
    sigma_los_arr = np.exp(log_sigma_los.squeeze().cpu().numpy())
    sigma_nlos_arr = np.exp(log_sigma_nlos.squeeze().cpu().numpy())
    sigma_ratio = sigma_nlos_arr / np.maximum(sigma_los_arr, 0.01)  # >1 means NLOS more uncertain
    
    return {
                'p_los': p_raw,
        'p_los_sharp': p_sharp,  # P0.2: temperature-scaled
        'sigma_ratio': sigma_ratio,  # P0.2: sigma_nlos/sigma_los.cpu().numpy(),
        'mu_nlos': mu_nlos.squeeze().cpu().numpy(),
        'sigma_los': np.exp(log_sigma_los.squeeze().cpu().numpy()),
        'sigma_nlos': np.exp(log_sigma_nlos.squeeze().cpu().numpy()),
        'elevation_deg': np.array([obs['elevation_deg'] for obs in obs_list]),
        'azimuth_deg': np.array([obs['azimuth_deg'] for obs in obs_list]),
        'gnss': [obs['gnss'] for obs in obs_list],
        'svid': [obs['svid'] for obs in obs_list],
        'pr_mes_km': np.array([obs['pr_mes_m'] / 1000.0 for obs in obs_list]),
        'nlos_label': np.array([obs['nlos_label'] for obs in obs_list]),
    }



# ============================================================
# P0.2: Platt scaling calibration for p_los
# ============================================================
# Temperature scaling (T=0.6) made berlin2 WLS-MoG worse (716->831m).
# Platt scaling is more flexible: p_cal = sigmoid(A * logit(p_raw) + B)
# This allows both sharpening (A>1) and centering (B) corrections.

def fit_platt_scaling(p_raw_list, labels_list):
    """Fit Platt scaling parameters using logistic regression on raw logits.
    
    Args:
        p_raw_list: list of (N,) arrays of raw p_los values
        labels_list: list of (N,) arrays of true LOS/NLOS labels (0=LOS, 1=NLOS)
    
    Returns:
        dict with A, B (scaling parameters) and calibration metrics
    """
    from scipy.optimize import minimize
    import torch
    import torch.nn.functional as F
    
    all_p = np.concatenate([np.asarray(p).flatten() for p in p_raw_list])
    all_labels = np.concatenate([np.asarray(l).flatten() for l in labels_list])
    
    valid = np.isfinite(all_p) & np.isfinite(all_labels)
    all_p = all_p[valid]
    all_labels = all_labels[valid]
    
    if len(all_p) < 10:
        print("  Platt scaling: insufficient data, using identity")
        return {"A": 1.0, "B": 0.0, "bce_uncal": 0.0, "bce_cal": 0.0,
                "var_raw": 0.0, "var_cal": 0.0}
    
    eps = 1e-8
    p_clipped = np.clip(all_p, eps, 1 - eps)
    raw_logits = np.log(p_clipped / (1 - p_clipped))
    
    def bce_loss(params):
        A, B = params
        z = A * raw_logits + B
        z_t = torch.tensor(z, dtype=torch.float32)
        y_t = torch.tensor(all_labels, dtype=torch.float32)
        return float(F.binary_cross_entropy_with_logits(z_t, y_t))
    
    best_loss = float("inf")
    best_params = (1.0, 0.0)
    for A_init in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for B_init in [-0.5, 0.0, 0.5]:
            loss = bce_loss((A_init, B_init))
            if loss < best_loss:
                best_loss = loss
                best_params = (A_init, B_init)
    
    result = minimize(bce_loss, best_params, method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 1e-8})
    
    A, B = result.x
    cal_loss = result.fun
    
    z_cal = A * raw_logits + B
    p_cal = 1.0 / (1.0 + np.exp(-z_cal))
    
    z_t = torch.tensor(raw_logits, dtype=torch.float32)
    y_t = torch.tensor(all_labels, dtype=torch.float32)
    uncal_bce = float(F.binary_cross_entropy_with_logits(z_t, y_t))
    
    var_raw = np.var(all_p)
    var_cal = np.var(p_cal)
    
    print(f"  Platt scaling: A={A:.4f}, B={B:.4f}")
    print(f"    BCE: {uncal_bce:.4f} -> {cal_loss:.4f} (delta={uncal_bce-cal_loss:+.4f})")
    print(f"    p_los variance: {var_raw:.4f} -> {var_cal:.4f} (x{var_cal/max(var_raw,1e-8):.1f})")
    print(f"    p_los range: [{np.min(all_p):.3f}, {np.max(all_p):.3f}] -> [{np.min(p_cal):.3f}, {np.max(p_cal):.3f}]")
    
    return {"A": float(A), "B": float(B), "bce_uncal": float(uncal_bce), "bce_cal": float(cal_loss),
            "var_raw": float(var_raw), "var_cal": float(var_cal)}

def apply_platt_scaling(p_raw, calib_params):
    """Apply Platt scaling to raw p_los values."""
    eps = 1e-8
    p_clipped = np.clip(np.asarray(p_raw), eps, 1 - eps)
    logits = np.log(p_clipped / (1 - p_clipped))
    z = calib_params["A"] * logits + calib_params["B"]
    return 1.0 / (1.0 + np.exp(-z))


print("fusion/utils.py loaded (v4 — fixed 11-feature Module 1 inference)")

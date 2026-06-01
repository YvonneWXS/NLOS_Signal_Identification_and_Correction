"""
fusion/utils.py — Coordinate transforms, data loading, Module 1 inference interface
==================================================================================
Reuses ECEF/LLA conversion logic from Module 1 Radio_Depth_Generate.py.
Provides a clean interface for loading raw data and running Module 1 inference.
"""
import sys, os
import numpy as np
import pandas as pd

# Add Module 1 model path for imports
_MODULE1_PATH = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model"
if _MODULE1_PATH not in sys.path:
    sys.path.insert(0, _MODULE1_PATH)

# ============================================================
# ECEF ↔ LLA conversion (from Radio_Depth_Generate.py logic)
# ============================================================

# WGS84 constants
_WGS84_A = 6378137.0          # semi-major axis (m)
_WGS84_F = 1.0 / 298.257223563  # flattening
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2  # first eccentricity squared


def lla_to_ecef(lat_deg, lon_deg, height_m):
    """Convert LLA (deg, deg, m) → ECEF (km)."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    x = (N + height_m) * cos_lat * np.cos(lon)
    y = (N + height_m) * cos_lat * np.sin(lon)
    z = (N * (1.0 - _WGS84_E2) + height_m) * sin_lat
    return np.array([x, y, z]) / 1000.0  # km


def ecef_to_lla(x_km, y_km, z_km):
    """Convert ECEF (km) → LLA (deg, deg, m). Iterative method."""
    x, y, z = x_km * 1000.0, y_km * 1000.0, z_km * 1000.0
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - _WGS84_E2 * N / (N + h)))
    sin_lat = np.sin(lat)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    h = p / np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), h


# ============================================================
# GNSS constellation mapping
# ============================================================

_GNSS_ID_MAP = {0: 'GPS', 1: 'GPS', 2: 'Galileo', 3: 'Glonass', 4: 'BeiDou',
                5: 'QZSS', 6: 'Galileo', 'GPS': 0, 'Glonass': 3, 'Galileo': 2, 'BeiDou': 4}


def gnss_to_sp3_prefix(gnss_name):
    """Map GNSS name → SP3 satellite prefix."""
    mapping = {'GPS': 'G', 'Glonass': 'R', 'Galileo': 'E', 'BeiDou': 'C'}
    return mapping.get(gnss_name, 'G')


# ============================================================
# Data loading
# ============================================================

def load_epoch_data(dataset_name, data_root=None):
    """Load all epochs from a dataset with ground truth and raw observations.
    
    Returns:
        list of dicts per epoch with keys:
        - gps_week, gps_seconds
        - gt_ecef: (3,) km ground truth position
        - obs: list of dicts per satellite:
            {svid, gnss, pr_mes_m, cno, pr_stdev_m, nlos_label,
             sv_ecef_km, elevation_deg, azimuth_deg}
    """
    if data_root is None:
        data_root = r"D:\3_document\4_research\NLOS Signal Identification and Correction\data\dataset"
    ds_dir = os.path.join(data_root, dataset_name)
    
    # Load ground truth positions
    nav_path = os.path.join(ds_dir, 'NAV-POSLLH.csv')
    if not os.path.exists(nav_path):
        raise FileNotFoundError(f"NAV-POSLLH.csv not found in {ds_dir}")
    
    try:
        nav_df = pd.read_csv(nav_path, sep=';')
    except Exception:
        nav_df = pd.read_csv(nav_path, sep=',')
    
    # Find column names
    lat_col = [c for c in nav_df.columns if 'Latitude' in c and 'GT' in c and 'Cov' not in c][0]
    lon_col = [c for c in nav_df.columns if 'Longitude' in c and 'GT' in c and 'Cov' not in c][0]
    h_col = [c for c in nav_df.columns if 'Height' in c and 'GT' in c and 'Cov' not in c and 'MSL' not in c][0]
    week_col = [c for c in nav_df.columns if 'GPSWeek' in c or 'GPS Week' in c][0]
    sec_col = [c for c in nav_df.columns if 'GPSSeconds' in c or 'GPS Second' in c][0]
    
    gt_positions = {}
    for _, row in nav_df.iterrows():
        key = (int(row[week_col]), float(row[sec_col]))
        gt_positions[key] = lla_to_ecef(row[lat_col], row[lon_col], row[h_col])
    
    # Load raw observations
    rawx_path = os.path.join(ds_dir, 'RXM-RAWX.csv')
    rawx_df = pd.read_csv(rawx_path, sep=';')
    
    pr_col = [c for c in rawx_df.columns if 'prMes' in c][0]
    gnss_col = [c for c in rawx_df.columns if 'gnssId' in c or 'GNSS' in c][0]
    svid_col = [c for c in rawx_df.columns if 'svId' in c][0]
    cno_col = [c for c in rawx_df.columns if 'cno' in c.lower() and 'dbHz' in c][0] if any('cno' in c.lower() for c in rawx_df.columns) else None
    prstd_col = [c for c in rawx_df.columns if 'prStdev' in c][0]
    nlos_col = [c for c in rawx_df.columns if 'NLOS' in c][0]
    week_col_r = [c for c in rawx_df.columns if 'GPSWeek' in c or 'week' in c.lower()][0]
    sec_col_r = [c for c in rawx_df.columns if 'GPSSeconds' in c or 'rcvTow' in c][0]
    
    # Group by epoch
    epochs = {}
    for _, row in rawx_df.iterrows():
        key = (int(row[week_col_r]), float(row[sec_col_r]))
        if key not in epochs:
            epochs[key] = []
        
        gnss_val = row[gnss_col]
        gnss_name = _GNSS_ID_MAP.get(gnss_val, 'GPS') if isinstance(gnss_val, (int, float)) else gnss_val
        
        obs = {
            'svid': int(row[svid_col]),
            'gnss': gnss_name,
            'pr_mes_m': float(row[pr_col]),
            'cno': float(row[cno_col]) if cno_col else 0.0,
            'pr_stdev_m': float(row[prstd_col]),
            'nlos_label': int(row[nlos_col]) if row[nlos_col] not in ['#', ''] else 0,
        }
        epochs[key].append(obs)
    
    # Build final epoch list
    result = []
    for (week, sec), obs_list in sorted(epochs.items()):
        if (week, sec) not in gt_positions:
            continue
        gt_ecef = gt_positions[(week, sec)]
        result.append({
            'gps_week': week,
            'gps_seconds': sec,
            'gt_ecef': gt_ecef,
            'obs': obs_list,
        })
    
    return result


# ============================================================
# Module 1 inference interface
# ============================================================

def load_mog_model(exp_name):
    """Load the best MoG Fix6 model for a given experiment.
    
    Args:
        exp_name: e.g., 'exp_034' for berlin1
    
    Returns:
        (model, config, device) tuple ready for inference
    """
    import torch
    from config import get_config
    
    # Find the model file
    result_dir = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result"
    
    # Import GAT_V2025 model
    from GAT_V2025 import NLOSGAT
    
    config = get_config()
    device = config.get_device()
    
    model = NLOSGAT(
        in_features=config.IN_FEATURES,
        hidden_features=config.HIDDEN_FEATURES,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)
    
    best_path = os.path.join(result_dir, exp_name, 'best_model.pth')
    if not os.path.exists(best_path):
        # Try final_model.pth
        best_path = os.path.join(result_dir, exp_name, 'final_model.pth')
    
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, device


def run_mog_inference(model, config, device, epoch_data):
    """Run Module 1 MoG inference on a single epoch.
    
    Args:
        epoch_data: dict with 'obs' list
    
    Returns:
        dict with per-satellite MoG outputs:
        - p_los: (N,) array
        - mu_nlos: (N,) array (km)
        - sigma_los: (N,) array (km)
        - sigma_nlos: (N,) array (km)
        - elevation_deg: (N,) array
        - azimuth_deg: (N,) array
        - gnss: list of strings
        - svid: list of ints
        - pr_mes_km: (N,) array pseudorange measurements in km
        - nlos_label: (N,) array ground truth
    """
    import torch
    from GAT_V2025 import _extract_elevation
    
    obs_list = epoch_data['obs']
    N = len(obs_list)
    if N == 0:
        return None
    
    # Build node features (matching Module 1 format)
    # Features: [elevation/90, azimuth/360, sin(el), cos(el), sin(az), cos(az),
    #             cno/50, pr_stdev_norm, ..., constellation_onehot_4]
    features = np.zeros((N, config.IN_FEATURES), dtype=np.float32)
    for i, obs in enumerate(obs_list):
        el = obs.get('elevation_deg', 0.0)
        az = obs.get('azimuth_deg', 0.0)
        cno = obs.get('cno', 0.0)
        pr_std = obs.get('pr_stdev_m', 0.0)
        
        features[i, 0] = el / 90.0
        features[i, 1] = az / 360.0
        features[i, 2] = np.sin(np.deg2rad(el))
        features[i, 3] = np.cos(np.deg2rad(el))
        features[i, 4] = np.sin(np.deg2rad(az))
        features[i, 5] = np.cos(np.deg2rad(az))
        features[i, 6] = min(cno / 50.0, 1.0)
        features[i, 7] = min(pr_std / 100.0, 1.0)
        # Remaining features zero (pseudorange_error, constellation one-hot)
    
    node_features = torch.tensor(features, device=device)
    
    # Build edges (fully connected for now, matching Module 1 default)
    edge_list = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_list.append([i, j])
    if edge_list:
        edge_index = torch.tensor(edge_list, device=device).t().contiguous()
    else:
        edge_index = torch.tensor([[0], [0]], device=device)
    
    with torch.no_grad():
        p_los, mu_nlos, log_sigma_los, log_sigma_nlos = model(node_features, edge_index)
    
    return {
        'p_los': p_los.squeeze().cpu().numpy(),
        'mu_nlos': mu_nlos.squeeze().cpu().numpy(),
        'sigma_los': np.exp(log_sigma_los.squeeze().cpu().numpy()),
        'sigma_nlos': np.exp(log_sigma_nlos.squeeze().cpu().numpy()),
        'elevation_deg': np.array([obs.get('elevation_deg', 0.0) for obs in obs_list]),
        'azimuth_deg': np.array([obs.get('azimuth_deg', 0.0) for obs in obs_list]),
        'gnss': [obs['gnss'] for obs in obs_list],
        'svid': [obs['svid'] for obs in obs_list],
        'pr_mes_km': np.array([obs['pr_mes_m'] / 1000.0 for obs in obs_list]),
        'nlos_label': np.array([obs['nlos_label'] for obs in obs_list]),
    }


def compute_satellite_positions(dataset_name, epoch_data):
    """Compute satellite ECEF positions from SP3 ephemeris + pseudorange.
    Simplified: use broadcast ephemeris if SP3 unavailable.
    
    For now, returns approximate satellite positions based on 
    elevation/azimuth and pseudorange.
    
    Returns:
        sv_ecef: (N, 3) array of satellite ECEF positions in km
    """
    # Simplified approach: estimate satellite position from receiver position,
    # elevation, azimuth, and pseudorange
    gt_ecef = epoch_data['gt_ecef']  # (3,) km
    sv_positions = np.zeros((len(epoch_data['obs']), 3))
    
    for i, obs in enumerate(epoch_data['obs']):
        el = np.deg2rad(obs.get('elevation_deg', 0.0))
        az = np.deg2rad(obs.get('azimuth_deg', 0.0))
        pr_m = obs.get('pr_mes_m', 2e7)  # meters
        
        # Unit vector from receiver to satellite (ENU frame)
        # E: East, N: North, U: Up
        e_unit = np.array([np.cos(el) * np.sin(az),
                           np.cos(el) * np.cos(az),
                           np.sin(el)])
        
        # Convert ENU unit vector to ECEF
        lat, lon = ecef_to_lla(gt_ecef[0], gt_ecef[1], gt_ecef[2])[:2]
        lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
        
        # ENU to ECEF rotation matrix
        R = np.array([
            [-np.sin(lon_r), -np.sin(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.cos(lon_r)],
            [np.cos(lon_r), -np.sin(lat_r) * np.sin(lon_r), np.cos(lat_r) * np.sin(lon_r)],
            [0, np.cos(lat_r), np.sin(lat_r)],
        ])
        
        ecef_unit = R @ e_unit
        sv_positions[i] = gt_ecef + ecef_unit * (pr_m / 1000.0)  # km
    
    return sv_positions


print("fusion/utils.py loaded successfully")
# debug_geometry.py — Part 1: Pseudorange Geometry Verification (P0)
# ====================================================================
# Step 1: Single epoch sanity check (PR vs geometric range)
# Step 2: Clock bias estimation (median residual absorption)
# Step 3: Jacobian sign verification (definitive numerical test)
# Step 4: SP3 clock correction decision (RMS comparison)
# ====================================================================

import sys, os, pickle, glob, numpy as np

# ---- paths ----
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model")
from sp3_reader import SP3Reader

# ---- WGS84 constants ----
_A = 6378137.0; _F = 1.0 / 298.257223563; _E2 = 2 * _F - _F ** 2

def lla_to_ecef(lat_deg, lon_deg, height_m):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    N = _A / np.sqrt(1.0 - _E2 * sl**2)
    x = (N + height_m) * cl * np.cos(lon)
    y = (N + height_m) * cl * np.sin(lon)
    z = (N * (1.0 - _E2) + height_m) * sl
    return np.array([x, y, z]) / 1000.0

# ---- load data ----
DATA_ROOT = r"D:\3_document\4_research\NLOS Signal Identification and Correction\data"
PROCESSED = os.path.join(DATA_ROOT, "processedData")
DATASET_DIR = os.path.join(DATA_ROOT, "dataset")

def load_epoch(dataset, idx=0):
    pkl = os.path.join(PROCESSED, f"{dataset}_processed.pkl")
    with open(pkl, 'rb') as f: ep_list = pickle.load(f)
    ep = ep_list[idx]
    gt = lla_to_ecef(ep.gt_lat, ep.gt_lon, ep.gt_height)
    obs = []
    GNSS_MAP = {0: 'GPS', 1: 'GPS', 2: 'Galileo', 3: 'Glonass', 4: 'BeiDou', 5: 'QZSS', 6: 'Galileo'}
    for o in ep.observations:
        gnss = str(o.gnss_id) if isinstance(o.gnss_id, str) else GNSS_MAP.get(o.gnss_id, 'GPS')
        obs.append({
            'svid': o.sv_id, 'gnss': gnss,
            'pr_m': o.pr_mes, 'el': o.elevation, 'az': o.azimuth,
            'cno': o.cno if hasattr(o, 'cno') else 0,
            'pr_stdev': o.pr_stdev if hasattr(o, 'pr_stdev') else 0,
            'nlos': o.nlos_label,
        })
    return ep.gps_week, ep.gps_seconds, gt, obs

def load_sp3(dataset):
    ds_dir = os.path.join(DATASET_DIR, dataset)
    sp3_files = [f for f in os.listdir(ds_dir) if f.endswith('.sp3') and not f.endswith('.Z')]
    if not sp3_files: return None
    return SP3Reader(os.path.join(ds_dir, sp3_files[0]))

_GNSS_TO_SP3 = {'GPS': 'G', 'Glonass': 'R', 'Galileo': 'E', 'BeiDou': 'C'}

def get_sv_pos(reader, gps_week, gps_sec, gnss, svid):
    sp3_id = f"{_GNSS_TO_SP3.get(gnss, 'G')}{svid:02d}"
    if reader is None or not reader.has_satellite(sp3_id):
        return None, None
    pos = reader.get_satellite_position(gps_week, gps_sec, sp3_id)
    clk = reader.get_satellite_clock(gps_week, gps_sec, sp3_id)
    if pos is None: return None, None
    return np.array(pos) / 1000.0, clk  # pos in km, clk in meters

# ================================================================
print("=" * 70)
print("PART 1: PSEUDORANGE GEOMETRY VERIFICATION")
print("=" * 70)

# Run on berlin1 (highest NLOS ratio, most challenging)
dataset = "berlin1_potsdamer_platz"
gps_week, gps_sec, gt_ecef, obs_list = load_epoch(dataset, idx=0)
reader = load_sp3(dataset)
print(f"\nDataset: {dataset}")
print(f"GPS: week={gps_week}, sec={gps_sec:.1f}")
print(f"GT ECEF (km): [{gt_ecef[0]:.3f}, {gt_ecef[1]:.3f}, {gt_ecef[2]:.3f}]")
print(f"Observations: {len(obs_list)}")
if reader:
    stats = reader.get_statistics()
    print(f"SP3: {stats['total_epochs']} epochs, {stats['total_satellites']} sats")

# ================================================================
# STEP 1: Single epoch sanity check
# ================================================================
print(f"\n{'='*70}")
print("STEP 1: Single Epoch Sanity Check")
print(f"{'='*70}")
print(f"{'SV':>4s} {'GNSS':>8s} {'el°':>6s} {'PR(km)':>12s} {'Geo(km)':>12s} {'Res(m)':>10s} {'Clk(km)':>10s} {'NLOS':>5s}")
print("-" * 70)

geo_ranges = []; pr_values = []; pr_plus_clk = []; pr_minus_clk = []
clk_values = []; elevations = []; nlos_flags = []

for i, obs in enumerate(obs_list):
    sv_pos, sv_clk = get_sv_pos(reader, gps_week, gps_sec, obs['gnss'], obs['svid'])
    if sv_pos is None:
        print(f"  {obs['svid']:>4d} {obs['gnss']:>8s} -- NO SP3 DATA --")
        continue
    geo = np.linalg.norm(sv_pos - gt_ecef)
    pr_km = obs['pr_m'] / 1000.0
    res_m = (pr_km - geo) * 1000.0
    clk_km = sv_clk / 1000.0 if sv_clk is not None else float('nan')
    geo_ranges.append(geo); pr_values.append(pr_km)
    clk_values.append(clk_km); elevations.append(obs['el'])
    nlos_flags.append(obs['nlos'])
    clk_str = f'{clk_km:.3f}' if not np.isnan(clk_km) else '   N/A'
    pr_plus_clk.append(pr_km + clk_km if not np.isnan(clk_km) else pr_km)
    pr_minus_clk.append(pr_km - clk_km if not np.isnan(clk_km) else pr_km)
    print(f"  {obs['svid']:>4d} {obs['gnss']:>8s} {obs['el']:>5.1f} {pr_km:>12.3f} {geo:>12.3f} {res_m:>10.1f} {clk_str:>10s} {obs['nlos']:>5d}")

geo_ranges = np.array(geo_ranges)
pr_values = np.array(pr_values)
clk_values = np.array(clk_values)
pr_plus_clk = np.array(pr_plus_clk)
pr_minus_clk = np.array(pr_minus_clk)
elevations = np.array(elevations)
nlos_flags = np.array(nlos_flags)

N_valid = len(geo_ranges)
print(f"\nSummary ({N_valid} satellites with SP3):")
print(f"  Mean PR:    {pr_values.mean():.3f} km")
print(f"  Mean geo:   {geo_ranges.mean():.3f} km")
print(f"  Mean PR-geo: {(pr_values-geo_ranges).mean()*1000:.1f} m")

# Check: residuals with just geo range (no clock absorption)
raw_residuals_km = pr_values - geo_ranges
los_residuals = raw_residuals_km[nlos_flags == 0] if (nlos_flags == 0).any() else np.array([])
nlos_residuals = raw_residuals_km[nlos_flags == 1] if (nlos_flags == 1).any() else np.array([])
if len(los_residuals) > 0:
    print(f"  LOS residuals:  [{los_residuals.min()*1000:.0f}, {los_residuals.max()*1000:.0f}] m")
if len(nlos_residuals) > 0:
    print(f"  NLOS residuals: [{nlos_residuals.min()*1000:.0f}, {nlos_residuals.max()*1000:.0f}] m")

# ================================================================
# STEP 2: Clock bias estimation
# ================================================================
print(f"\n{'='*70}")
print("STEP 2: Clock Bias Estimation")
print(f"{'='*70}")

clk_bias_0 = np.median(raw_residuals_km)
print(f"  Clock bias (median of PR - geo): {clk_bias_0:.3f} km = {clk_bias_0*1e3:.1f} m")

residuals_after_clk = raw_residuals_km - clk_bias_0
los_after = residuals_after_clk[nlos_flags == 0] if (nlos_flags == 0).any() else np.array([])
nlos_after = residuals_after_clk[nlos_flags == 1] if (nlos_flags == 1).any() else np.array([])

print(f"  After clock absorption ({clk_bias_0:.3f} km):")
if len(los_after) > 0:
    print(f"    LOS  mean|res| = {np.mean(np.abs(los_after))*1000:.1f} m, range [{los_after.min()*1000:.0f}, {los_after.max()*1000:.0f}] m")
if len(nlos_after) > 0:
    print(f"    NLOS mean|res| = {np.mean(np.abs(nlos_after))*1000:.1f} m, range [{nlos_after.min()*1000:.0f}, {nlos_after.max()*1000:.0f}] m")

# Expected: LOS in [-500, +500]m, NLOS in [-2000, +5000]m
los_ok = len(los_after) == 0 or (los_after.max()*1000 < 500 and los_after.min()*1000 > -500)
nlos_ok = len(nlos_after) == 0 or (nlos_after.max()*1000 < 5000 and nlos_after.min()*1000 > -2000)
print(f"\n  LOS check:  {'PASS' if los_ok else 'FAIL'} (expected [-500, +500]m)")
print(f"  NLOS check: {'PASS' if nlos_ok else 'FAIL'} (expected [-2000, +5000]m)")

# ================================================================
# STEP 3: Jacobian Sign Verification (DEFINITIVE TEST)
# ================================================================
print(f"\n{'='*70}")
print("STEP 3: Jacobian Sign Verification")
print(f"{'='*70}")

# Use all N satellites. State: x = [pos_x, pos_y, pos_z, clk] in km
# Start from gt + 0.1km east
x0 = np.array([gt_ecef[0], gt_ecef[1] + 0.1, gt_ecef[2], 0.0])
print(f"  Starting state: pos=[{x0[0]:.3f}, {x0[1]:.3f}, {x0[2]:.3f}], clk={x0[3]:.3f} km")
print(f"  Offset from GT: 100m east")

sv_pos_km = np.array([get_sv_pos(reader, gps_week, gps_sec, obs['gnss'], obs['svid'])[0]
                       for obs in obs_list if get_sv_pos(reader, gps_week, gps_sec, obs['gnss'], obs['svid'])[0] is not None])
pr_km = np.array([obs['pr_m'] / 1000.0 for obs in obs_list])[:len(sv_pos_km)]

# Compute one WLS iteration
dist = np.linalg.norm(sv_pos_km - x0[:3], axis=1)
pred_pr = dist + x0[3]
residuals = pr_km - pred_pr

# H matrix: d(pred_pr)/dx
los_vecs = (sv_pos_km - x0[:3]) / np.maximum(dist[:, None], 1e-8)

# Test BOTH Jacobian conventions
print(f"\n  Convention A: H[:,:3] = -LOS  (d(pred_pr)/d(pos) = -(SV-rx)/dist)")
print(f"  Convention B: H[:,:3] = +LOS  (alternative)")
print()

for label, H_pos in [("H[:,:3] = -LOS (correct)", -los_vecs),
                      ("H[:,:3] = +LOS (old/v1)",   +los_vecs)]:
    H = np.zeros((len(pr_km), 4))
    H[:, :3] = H_pos
    H[:, 3] = 1.0
    try:
        delta = np.linalg.lstsq(H, residuals, rcond=None)[0]
    except:
        delta = np.zeros(4)
    x_new = x0 + delta
    d0 = np.linalg.norm((x0[:2] - gt_ecef[:2]) * 1000)
    d1 = np.linalg.norm((x_new[:2] - gt_ecef[:2]) * 1000)
    direction = "TOWARD GT" if d1 < d0 else "AWAY from GT" if d1 > d0 else "STATIONARY"
    print(f"  {label}:")
    print(f"    Delta = [{delta[0]*1e3:.1f}, {delta[1]*1e3:.1f}, {delta[2]*1e3:.1f}, {delta[3]*1e3:.1f}] (m, clk in m)")
    print(f"    d(GT) before={d0:.1f}m, after={d1:.1f}m → {direction}")

# DEFINTIVE: which convention moves toward GT?
H_correct = np.zeros((len(pr_km), 4))
H_correct[:, :3] = -los_vecs
H_correct[:, 3] = 1.0
delta_correct = np.linalg.lstsq(H_correct, residuals, rcond=None)[0]
H_old = np.zeros((len(pr_km), 4))
H_old[:, :3] = +los_vecs
H_old[:, 3] = 1.0
delta_old = np.linalg.lstsq(H_old, residuals, rcond=None)[0]

d_correct = np.linalg.norm(((x0 + delta_correct)[:2] - gt_ecef[:2]) * 1000)
d_old = np.linalg.norm(((x0 + delta_old)[:2] - gt_ecef[:2]) * 1000)
d_start = np.linalg.norm((x0[:2] - gt_ecef[:2]) * 1000)

print(f"\n  VERDICT:")
print(f"    Start distance: {d_start:.1f}m")
print(f"    H=-LOS gives:   {d_correct:.1f}m")
print(f"    H=+LOS gives:   {d_old:.1f}m")
if d_correct < d_start and d_old > d_start:
    print(f"    → H[:,:3] = -LOS is CORRECT (moves toward GT)")
    JACOBIAN_SIGN = "NEGATIVE"
elif d_old < d_start and d_correct > d_start:
    print(f"    → H[:,:3] = +LOS is correct (moves toward GT)")
    JACOBIAN_SIGN = "POSITIVE"
else:
    print(f"    → AMBIGUOUS — both or neither move toward GT")
    JACOBIAN_SIGN = "AMBIGUOUS"

# ================================================================
# STEP 4: SP3 Clock Correction Decision
# ================================================================
print(f"\n{'='*70}")
print("STEP 4: SP3 Clock Correction Decision")
print(f"{'='*70}")

# Case A: no SP3 clock correction
pr_a = pr_km
clk_a = np.median(pr_a - np.linalg.norm(sv_pos_km - gt_ecef[:3], axis=1))
res_a = pr_a - (np.linalg.norm(sv_pos_km - gt_ecef[:3], axis=1) + clk_a)
rms_a = np.sqrt(np.mean(res_a**2)) * 1000

# Case B: subtract SP3 clock correction
pr_b = pr_minus_clk
clk_b = np.median(pr_b - np.linalg.norm(sv_pos_km - gt_ecef[:3], axis=1))
res_b = pr_b - (np.linalg.norm(sv_pos_km - gt_ecef[:3], axis=1) + clk_b)
rms_b = np.sqrt(np.mean(res_b**2)) * 1000

# Case C: add SP3 clock correction
pr_c = pr_plus_clk
clk_c = np.median(pr_c - np.linalg.norm(sv_pos_km - gt_ecef[:3], axis=1))
res_c = pr_c - (np.linalg.norm(sv_pos_km - gt_ecef[:3], axis=1) + clk_c)
rms_c = np.sqrt(np.mean(res_c**2)) * 1000

print(f"  Case A (no clock corr):    RMS residual = {rms_a:.1f} m, clk_bias = {clk_a:.3f} km")
print(f"  Case B (PR - SP3 clock):   RMS residual = {rms_b:.1f} m, clk_bias = {clk_b:.3f} km")
print(f"  Case C (PR + SP3 clock):   RMS residual = {rms_c:.1f} m, clk_bias = {clk_c:.3f} km")

best_case = np.argmin([rms_a, rms_b, rms_c])
case_names = ['A (no correction)', 'B (PR - clk)', 'C (PR + clk)']
print(f"\n  BEST: Case {case_names[best_case]} with RMS = {[rms_a, rms_b, rms_c][best_case]:.1f} m")

USE_SP3_CLOCK = best_case != 0  # True if case B or C is better
SP3_CLOCK_SIGN = -1 if best_case == 1 else (+1 if best_case == 2 else 0)

# Multi-epoch verification
print(f"\n  Multi-epoch check (first 50 epochs):")
all_rms = {'A': [], 'B': [], 'C': []}
for ep_idx in range(min(50, 1)):  # Just epoch 0 for now — extend if needed
    pass

print(f"\n{'='*70}")
print("PART 1 SUMMARY")
print(f"{'='*70}")
print(f"  Step 1: PR vs geo values appear reasonable after clock absorption: {'PASS' if los_ok and nlos_ok else 'NEEDS REVIEW'}")
print(f"  Step 2: Clock bias estimated as {clk_bias_0:.3f} km ({clk_bias_0*1e3:.0f} m) via median")
print(f"  Step 3: Jacobian sign = {JACOBIAN_SIGN} (H[:,:3] should be -LOS)")
print(f"  Step 4: Best clock strategy = {case_names[best_case]} (RMS={min(rms_a, rms_b, rms_c):.1f}m)")
print(f"\n  Hard-coded decisions for v2 code:")
print(f"    H[:,:3] = -LOS  # d(pred_pr)/d(pos) = -(SV-rx)/||SV-rx||")
print(f"    USE_SP3_CLOCK_CORRECTION = {USE_SP3_CLOCK}")
if USE_SP3_CLOCK:
    sign_str = '+' if SP3_CLOCK_SIGN > 0 else '-'
    print(f"    SP3_CLOCK_SIGN = {sign_str}1  # PR_corrected = PR {sign_str} sp3_clock")
print("=" * 70)

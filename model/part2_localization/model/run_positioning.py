"""run_positioning.py — Simplified positioning baseline comparison for all datasets."""
import os, sys, json, time, pickle, numpy as np, torch
from scipy.optimize import minimize

_PART1_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model"
sys.path.insert(0, _PART1_DIR)
import GAT_V2025 as G
from run_urbannav import load_hk_data

RESULT_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\result\exp_hk_v2"
os.makedirs(RESULT_DIR, exist_ok=True)

# === Coordinate transforms ===
def lla_to_ecef(lat, lon, alt):
    a, f = 6378137.0, 1.0/298.257223563
    rlat, rlon = np.radians(lat), np.radians(lon)
    e2 = 2*f - f*f
    N = a / np.sqrt(1 - e2 * np.sin(rlat)**2)
    return np.array([(N+alt)*np.cos(rlat)*np.cos(rlon),
                     (N+alt)*np.cos(rlat)*np.sin(rlon),
                     (N*(1-e2)+alt)*np.sin(rlat)])

def ecef_to_lla(x, y, z):
    a, f = 6378137.0, 1.0/298.257223563
    e2 = 2*f - f*f
    lon = np.arctan2(y, x)
    p = np.sqrt(x*x + y*y)
    lat = np.arctan2(z, p*(1-e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2*np.sin(lat)**2)
        h = p/np.cos(lat) - N
        lat = np.arctan2(z, p*(1-e2*N/(N+h)))
    return np.degrees(lat), np.degrees(lon), h

# === Positioning Methods ===
def solve_standard_ls(sv_pos, pr_meas):
    """Standard LS: equal weights, solve position + clock."""
    N = len(pr_meas)
    state = np.zeros(4)
    for _ in range(10):
        H = np.zeros((N, 4))
        y = np.zeros(N)
        for i in range(N):
            r = np.linalg.norm(sv_pos[i] - state[:3])
            H[i, :3] = (state[:3] - sv_pos[i]) / r
            H[i, 3] = 1.0
            y[i] = pr_meas[i] - r - state[3]
        try:
            dx = np.linalg.lstsq(H, y, rcond=None)[0]
        except:
            break
        state += dx
        if np.linalg.norm(dx) < 1e-3: break
    return state

def solve_wls_mog(sv_pos, pr_meas, p_los, sigma):
    """WLS with p_los/sigma^2 weights."""
    N = len(pr_meas)
    state = np.zeros(4)
    for _ in range(10):
        H = np.zeros((N, 4))
        y = np.zeros(N)
        W = np.diag(np.clip(p_los, 0.05, 1.0) / np.clip(sigma, 0.1, 10.0)**2)
        for i in range(N):
            r = np.linalg.norm(sv_pos[i] - state[:3])
            H[i, :3] = (state[:3] - sv_pos[i]) / r
            H[i, 3] = 1.0
            y[i] = pr_meas[i] - r - state[3]
        try:
            dx = np.linalg.lstsq(H.T @ W @ H, H.T @ W @ y, rcond=None)[0]
        except:
            break
        state += dx
        if np.linalg.norm(dx) < 1e-3: break
    return state

def solve_fg_mog(sv_pos, pr_meas, p_los, mu_nlos, sigma_los, sigma_nlos):
    """Factor graph with MoG observation model (L-BFGS-B)."""
    N = len(pr_meas)
    p_nlos = 1.0 - p_los
    
    def nll(x):
        loss = 0.0
        for i in range(N):
            r = np.linalg.norm(sv_pos[i] - x[:3])
            residual = pr_meas[i] - r - x[3]
            ll_los = -0.5*(residual/sigma_los[i])**2 - np.log(sigma_los[i] + 1e-6)
            ll_nlos = -0.5*((residual-mu_nlos[i])/sigma_nlos[i])**2 - np.log(sigma_nlos[i] + 1e-6)
            max_ll = max(ll_los, ll_nlos)
            loss -= max_ll + np.log(p_los[i]*np.exp(ll_los-max_ll) + p_nlos[i]*np.exp(ll_nlos-max_ll) + 1e-10)
        return loss
    
    res = minimize(nll, np.zeros(4), method="L-BFGS-B", options={"maxiter": 50, "ftol": 1e-6})
    return res.x

# === Data Loading ===
def load_hk_for_positioning():
    train, val = load_hk_data()
    result = []
    for d in train + val:
        nf = d["node_features"]
        N = d["num_satellites"]
        # Derive obs from features
        obs = []
        for i in range(N):
            gnss_idx = {7:"GPS",8:"Glonass",9:"Galileo",10:"BeiDou"}
            gnss = "GPS"
            for j,n in gnss_idx.items():
                if nf[i,j] > 0.5: gnss = n; break
            obs.append({
                "elevation": float(nf[i,0]*90), "azimuth": float(nf[i,1]*360),
                "cno": float(nf[i,2]*60), "pr_mes": float(nf[i,4]*3e7),
                "nlos_label": int(d["nlos_labels"][i]),
                "gnss": gnss, "svid": i+1,
            })
        result.append({
            "gps_week": d["gps_week"], "gps_seconds": d["gps_seconds"],
            "gt_lat": d["gt_lat"], "gt_lon": d["gt_lon"], "gt_alt": d["gt_alt"],
            "obs": obs,
        })
    return result

def load_berlin_data(name):
    pkl_path = r"D:\3_document\4_research\NLOS Signal Identification and Correction\data\processedData\{0}_processed.pkl".format(name)
    with open(pkl_path, "rb") as f: data = pickle.load(f)
    result = []
    for ep in data:
        obs = []
        for o in ep.observations:
            obs.append({
                "elevation": o.elevation, "azimuth": o.azimuth,
                "cno": o.cno, "pr_mes": o.pr_mes,
                "nlos_label": o.nlos_label,
                "gnss": o.gnss_id, "svid": o.sv_id,
            })
        result.append({
            "gps_week": ep.gps_week, "gps_seconds": ep.gps_seconds,
            "gt_lat": ep.gt_lat, "gt_lon": ep.gt_lon, "gt_alt": ep.gt_height,
            "obs": obs,
        })
    return result

# === MoG Inference ===
def run_mog_inference(epochs, exp_name):
    exp_dir = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result" + "\\" + exp_name
    mp = os.path.join(exp_dir, "best_model.pth")
    device = torch.device("cuda")
    ckpt = torch.load(mp, map_location=device, weights_only=False)
    model = G.NLOSGAT(11, 128, 8, 2, 0.1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    gmap = {"GPS":7,"Glonass":8,"Galileo":9,"BeiDou":10}
    outputs = []
    for ep in epochs:
        obs = ep["obs"]; N = len(obs)
        nf = np.zeros((N,11), dtype=np.float32)
        for i,o in enumerate(obs):
            nf[i,0]=o["elevation"]/90; nf[i,1]=o["azimuth"]/360
            nf[i,2]=o["cno"]/60; nf[i,3]=1.0/5; nf[i,4]=o["pr_mes"]/3e7
            nf[i,6]=np.cos(np.radians(o["elevation"]))
            nf[i,gmap.get(o["gnss"],7)]=1.0
        az = np.array([o["azimuth"] for o in obs])
        edges = []
        for i in range(N):
            for j in range(i+1,N):
                d=abs(az[i]-az[j])
                if d>180: d=360-d
                if d<90: edges.extend([[i,j],[j,i]])
        if len(edges)==0: ei=torch.tensor([[i,i] for i in range(N)],dtype=torch.long,device=device)
        else: ei=torch.tensor(edges,dtype=torch.long).T.to(device)
        with torch.no_grad():
            p,mu,sl,sn = model(torch.tensor(nf,device=device), ei)
        outputs.append({"p_los":p.cpu().numpy().flatten(),"mu_nlos":mu.cpu().numpy().flatten(),
                        "sigma_los":torch.exp(sl).cpu().numpy().flatten(),
                        "sigma_nlos":torch.exp(sn).cpu().numpy().flatten()})
    return outputs

# === Main ===
datasets = {
    "berlin1": ("berlin1_potsdamer_platz", "exp_001"),
    "berlin2": ("berlin2_gendarmenmarkt", "exp_002"),
    "frankfurt1": ("frankfurt1_maintower", "exp_003"),
    "frankfurt2": ("frankfurt2_westendtower", "exp_004"),
    "hk": ("UrbanNav-HK_TST", "exp_hk"),
}

all_metrics = {}

for short, (ds_name, exp_name) in datasets.items():
    print("\n" + "="*60)
    print(short.upper(), "—", ds_name)
    print("="*60)
    
    if short == "hk":
        epochs = load_hk_for_positioning()
    else:
        epochs = load_berlin_data(ds_name)
    print("Epochs:", len(epochs))
    
    mog = run_mog_inference(epochs, exp_name)
    print("MoG inference done")
    
    # Use last 30% as test
    split = int(len(epochs) * 0.7)
    test_epochs = epochs[split:]
    test_mog = mog[split:]
    
    ls_errors, wls_errors, fg_errors = [], [], []
    for ep, m in zip(test_epochs, test_mog):
        obs = ep["obs"]; N = len(obs)
        gt_ecef = lla_to_ecef(ep["gt_lat"], ep["gt_lon"], ep["gt_alt"])
        
        # Simulate SV positions from elevation/azimuth (simple approximation)
        sv_pos = np.zeros((N, 3))
        for i, o in enumerate(obs):
            el_r, az_r = np.radians(o["elevation"]), np.radians(o["azimuth"])
            enu = np.array([np.cos(el_r)*np.sin(az_r), np.cos(el_r)*np.cos(az_r), np.sin(el_r)])
            # R from ENU to ECEF
            rlat, rlon = np.radians(ep["gt_lat"]), np.radians(ep["gt_lon"])
            R = np.array([[-np.sin(rlon), -np.sin(rlat)*np.cos(rlon), np.cos(rlat)*np.cos(rlon)],
                          [np.cos(rlon), -np.sin(rlat)*np.sin(rlon), np.cos(rlat)*np.sin(rlon)],
                          [0, np.cos(rlat), np.sin(rlat)]])
            sv_pos[i] = gt_ecef + R @ enu * 20000  # ~20,000 km
        
        pr_meas = np.array([o["pr_mes"] for o in obs])
        
        # LS
        try:
            ls_state = solve_standard_ls(sv_pos, pr_meas)
            ls_err = np.linalg.norm(ls_state[:3] - gt_ecef)
            ls_errors.append(ls_err)
        except: pass
        
        # WLS-MoG
        try:
            wls_state = solve_wls_mog(sv_pos, pr_meas, m["p_los"], m["sigma_los"])
            wls_err = np.linalg.norm(wls_state[:3] - gt_ecef)
            wls_errors.append(wls_err)
        except: pass
        
        # FG-MoG
        try:
            fg_state = solve_fg_mog(sv_pos, pr_meas, m["p_los"], m["mu_nlos"], m["sigma_los"], m["sigma_nlos"])
            fg_err = np.linalg.norm(fg_state[:3] - gt_ecef)
            fg_errors.append(fg_err)
        except: pass
    
    ls_e = np.array(ls_errors); wls_e = np.array(wls_errors); fg_e = np.array(fg_errors)
    
    def cep(errs, pct):
        if len(errs) == 0: return 0
        return np.percentile(errs, pct)
    
    metrics = {
        "ls": {"cep50": float(cep(ls_e,50)), "cep95": float(cep(ls_e,95)), "mean": float(ls_e.mean()), "n": len(ls_e)},
        "wls_mog": {"cep50": float(cep(wls_e,50)), "cep95": float(cep(wls_e,95)), "mean": float(wls_e.mean()), "n": len(wls_e)},
        "fg_mog": {"cep50": float(cep(fg_e,50)), "cep95": float(cep(fg_e,95)), "mean": float(fg_e.mean()), "n": len(fg_e)},
    }
    all_metrics[short] = metrics
    
    print("LS:   CEP50={:.1f}m  CEP95={:.1f}m  Mean={:.1f}m".format(cep(ls_e,50), cep(ls_e,95), ls_e.mean()))
    print("WLS:  CEP50={:.1f}m  CEP95={:.1f}m  Mean={:.1f}m".format(cep(wls_e,50), cep(wls_e,95), wls_e.mean()))
    print("FG:   CEP50={:.1f}m  CEP95={:.1f}m  Mean={:.1f}m".format(cep(fg_e,50), cep(fg_e,95), fg_e.mean()))

# Save
with open(os.path.join(RESULT_DIR, "positioning_metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

# Generate report
report = "# Positioning Comparison (5 Datasets)\n\n"
report += "| Dataset | Method | CEP50 (m) | CEP95 (m) | Mean (m) |\n"
report += "|---------|--------|:---------:|:---------:|:--------:|\n"
for short, metrics in all_metrics.items():
    for method, m in metrics.items():
        report += "| {0} | {1} | {2:.1f} | {3:.1f} | {4:.1f} |\n".format(short, method, m["cep50"], m["cep95"], m["mean"])

with open(os.path.join(RESULT_DIR, "comparison_report.md"), "w") as f:
    f.write(report)
print("\n" + report)
print("\nResults saved to:", RESULT_DIR)

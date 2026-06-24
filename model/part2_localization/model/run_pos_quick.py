"""Quick positioning baseline for 4 European datasets only."""
import os, sys, json, time, pickle, numpy as np, torch
from scipy.optimize import minimize

_PART1_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model"
sys.path.insert(0, _PART1_DIR)
import GAT_V2025 as G

RESULT_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\result\exp_hk_v2"
os.makedirs(RESULT_DIR, exist_ok=True)

def lla_to_ecef(lat, lon, alt):
    a, f = 6378137.0, 1.0/298.257223563
    rlat, rlon = np.radians(lat), np.radians(lon)
    e2 = 2*f - f*f
    N = a / np.sqrt(1 - e2*np.sin(rlat)**2)
    return np.array([(N+alt)*np.cos(rlat)*np.cos(rlon),
                     (N+alt)*np.cos(rlat)*np.sin(rlon),
                     (N*(1-e2)+alt)*np.sin(rlat)])

def load_dataset(name):
    pkl = r"D:\3_document\4_research\NLOS Signal Identification and Correction\data\processedData\{0}_processed.pkl".format(name)
    with open(pkl, "rb") as f: data = pickle.load(f)
    result = []
    for ep in data:
        obs = []
        for o in ep.observations:
            obs.append({"elevation": o.elevation, "azimuth": o.azimuth, "cno": o.cno,
                        "pr_mes": o.pr_mes, "nlos_label": o.nlos_label,
                        "gnss": o.gnss_id, "svid": o.sv_id})
        result.append({"gps_week": ep.gps_week, "gps_seconds": ep.gps_seconds,
                       "gt_lat": ep.gt_lat, "gt_lon": ep.gt_lon, "gt_alt": ep.gt_height,
                       "obs": obs})
    return result

def solve_ls(sv_pos, pr):
    state = np.zeros(4)
    for _ in range(10):
        H = np.zeros((len(pr),4)); y = np.zeros(len(pr))
        for i in range(len(pr)):
            r = np.linalg.norm(sv_pos[i]-state[:3])
            H[i,:3] = (state[:3]-sv_pos[i])/r; H[i,3]=1.0
            y[i]=pr[i]-r-state[3]
        try: dx = np.linalg.lstsq(H, y, rcond=None)[0]
        except: break
        state += dx
        if np.linalg.norm(dx)<1e-3: break
    return state

def solve_wls(sv_pos, pr, w):
    state = np.zeros(4)
    W = np.diag(np.clip(w, 0.01, 10.0))
    for _ in range(10):
        H = np.zeros((len(pr),4)); y = np.zeros(len(pr))
        for i in range(len(pr)):
            r = np.linalg.norm(sv_pos[i]-state[:3])
            H[i,:3] = (state[:3]-sv_pos[i])/r; H[i,3]=1.0
            y[i]=pr[i]-r-state[3]
        try: dx = np.linalg.lstsq(H.T@W@H, H.T@W@y, rcond=None)[0]
        except: break
        state += dx
        if np.linalg.norm(dx)<1e-3: break
    return state

# MoG inference
def mog_infer(epochs, exp_name):
    mp = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result\{0}\best_model.pth".format(exp_name)
    dev = torch.device("cuda")
    ckpt = torch.load(mp, map_location=dev, weights_only=False)
    m = G.NLOSGAT(11,128,8,2,0.1).to(dev)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    gmap = {"GPS":7,"Glonass":8,"Galileo":9,"BeiDou":10}
    outs = []
    for ep in epochs:
        obs = ep["obs"]; N = len(obs)
        nf = np.zeros((N,11),dtype=np.float32)
        for i,o in enumerate(obs):
            nf[i,0]=o["elevation"]/90; nf[i,1]=o["azimuth"]/360
            nf[i,2]=o["cno"]/60; nf[i,3]=1.0/5; nf[i,4]=o["pr_mes"]/3e7
            nf[i,6]=np.cos(np.radians(o["elevation"])); nf[i,gmap.get(o["gnss"],7)]=1.0
        az=np.array([o["azimuth"] for o in obs]); edges=[]
        for i in range(N):
            for j in range(i+1,N):
                d=abs(az[i]-az[j])
                if d>180: d=360-d
                if d<90: edges.extend([[i,j],[j,i]])
        if len(edges)==0: ei=torch.tensor([[i,i] for i in range(N)],dtype=torch.long,device=dev)
        else: ei=torch.tensor(edges,dtype=torch.long).T.to(dev)
        with torch.no_grad(): p,mu,sl,sn = m(torch.tensor(nf,device=dev), ei)
        outs.append({"p_los":p.cpu().numpy().flatten(),"mu_nlos":mu.cpu().numpy().flatten(),
                     "sigma_los":torch.exp(sl).cpu().numpy().flatten(),
                     "sigma_nlos":torch.exp(sn).cpu().numpy().flatten()})
    return outs

# Run
datasets = [
    ("berlin1", "berlin1_potsdamer_platz", "exp_001"),
    ("berlin2", "berlin2_gendarmenmarkt", "exp_002"),
    ("frankfurt1", "frankfurt1_maintower", "exp_003"),
    ("frankfurt2", "frankfurt2_westendtower", "exp_004"),
]

all_metrics = {}

for short, ds_name, exp_name in datasets:
    print("\n" + "="*50)
    print(short, ds_name)
    print("="*50)
    epochs = load_dataset(ds_name)
    mog = mog_infer(epochs, exp_name)
    print("Data:", len(epochs), "epochs, MoG done")
    
    split = int(len(epochs)*0.7)
    test_ep = epochs[split:]; test_mog = mog[split:]
    
    ls_errs, wls_errs = [], []
    n_skipped = 0
    for ep, m in zip(test_ep, test_mog):
        obs = ep["obs"]; N = len(obs)
        gt_ecef = lla_to_ecef(ep["gt_lat"], ep["gt_lon"], ep["gt_alt"])
        
        # Approximate SV positions (20,000 km away in ENU direction)
        sv_pos = np.zeros((N,3))
        for i,o in enumerate(obs):
            el_r, az_r = np.radians(o["elevation"]), np.radians(o["azimuth"])
            enu = np.array([np.cos(el_r)*np.sin(az_r), np.cos(el_r)*np.cos(az_r), np.sin(el_r)])
            rlat, rlon = np.radians(ep["gt_lat"]), np.radians(ep["gt_lon"])
            R = np.array([[-np.sin(rlon),-np.sin(rlat)*np.cos(rlon),np.cos(rlat)*np.cos(rlon)],
                          [np.cos(rlon),-np.sin(rlat)*np.sin(rlon),np.cos(rlat)*np.sin(rlon)],
                          [0,np.cos(rlat),np.sin(rlat)]])
            sv_pos[i] = gt_ecef + R@enu*20200000
        
        pr = np.array([o["pr_mes"] for o in obs])
        
        try:
            ls_s = solve_ls(sv_pos, pr)
            ls_errs.append(np.linalg.norm(ls_s[:3]-gt_ecef))
        except: n_skipped += 1
        
        try:
            wls_s = solve_wls(sv_pos, pr, m["p_los"]/(m["sigma_los"]**2+0.01))
            wls_errs.append(np.linalg.norm(wls_s[:3]-gt_ecef))
        except: pass
    
    ls_e = np.array(ls_errs); wls_e = np.array(wls_errs)
    print("LS:  CEP50={:.1f}m CEP95={:.1f}m Mean={:.1f}m (n={})".format(
        np.percentile(ls_e,50), np.percentile(ls_e,95), ls_e.mean(), len(ls_e)))
    print("WLS: CEP50={:.1f}m CEP95={:.1f}m Mean={:.1f}m (n={})".format(
        np.percentile(wls_e,50), np.percentile(wls_e,95), wls_e.mean(), len(wls_e)))
    
    all_metrics[short] = {
        "ls": {"cep50": float(np.percentile(ls_e,50)), "cep95": float(np.percentile(ls_e,95)),
               "mean": float(ls_e.mean()), "n": len(ls_e)},
        "wls_mog": {"cep50": float(np.percentile(wls_e,50)), "cep95": float(np.percentile(wls_e,95)),
                    "mean": float(wls_e.mean()), "n": len(wls_e)},
    }

# Save and report
with open(os.path.join(RESULT_DIR, "positioning_metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

report = "# Positioning Comparison\n\n| Dataset | Method | CEP50 (m) | CEP95 (m) | Mean (m) | N |\n|---------|--------|:---------:|:---------:|:--------:|:-:|\n"
for short, metrics in all_metrics.items():
    for method in ["ls", "wls_mog"]:
        m = metrics[method]
        report += "| {0} | {1} | {2:.1f} | {3:.1f} | {4:.1f} | {5} |\n".format(
            short, method, m["cep50"], m["cep95"], m["mean"], m["n"])
with open(os.path.join(RESULT_DIR, "comparison_report.md"), "w") as f:
    f.write(report)
print("\n" + report)
print("Saved to:", RESULT_DIR)

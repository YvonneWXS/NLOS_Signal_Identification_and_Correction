import os, sys, json, time, pickle, numpy as np, torch
from torch.utils.data import DataLoader

_PART1_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model"
sys.path.insert(0, _PART1_DIR)
import GAT_V2025 as G
from run_urbannav import UrbanNavDataset, load_hk_data

def load_hk_epoch_data():
    """Load HK data and convert to Module 2-compatible format."""
    def _lla_to_ecef(lat, lon, alt):
        a, f = 6378137.0, 1.0/298.257223563
        lat_r, lon_r = np.radians(lat), np.radians(lon)
        e2 = 2*f - f*f
        N = a / np.sqrt(1 - e2 * np.sin(lat_r)**2)
        x = (N + alt) * np.cos(lat_r) * np.cos(lon_r)
        y = (N + alt) * np.cos(lat_r) * np.sin(lon_r)
        z = (N*(1-e2) + alt) * np.sin(lat_r)
        return np.array([x/1000.0, y/1000.0, z/1000.0])
    
    train, val = load_hk_data()
    result = []
    for d in train + val:
        gt_ecef = _lla_to_ecef(d["gt_lat"], d["gt_lon"], d["gt_alt"])
        nf, nl = d["node_features"], d["nlos_labels"]
        N = d["num_satellites"]
        gnss_map = {7: "GPS", 8: "Glonass", 9: "Galileo", 10: "BeiDou"}
        obs_list = []
        for i in range(N):
            gnss = "GPS"
            for j, name in gnss_map.items():
                if nf[i, j] > 0.5: gnss = name; break
            obs_list.append({
                "svid": i, "gnss": gnss,
                "pr_mes_m": float(nf[i, 4] * 3e7),
                "cno": float(nf[i, 2] * 60.0),
                "pr_stdev_m": 1.0,
                "nlos_label": int(nl[i]),
                "elevation_deg": float(nf[i, 0] * 90.0),
                "azimuth_deg": float(nf[i, 1] * 360.0),
            })
        result.append({
            "gps_week": d["gps_week"], "gps_seconds": d["gps_seconds"],
            "gt_ecef": gt_ecef, "obs": obs_list,
        })
    return result

def run_hk_mog_inference(epochs):
    """One-epoch-at-a-time MoG inference for HK."""
    exp_dir = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result\exp_hk"
    mp = os.path.join(exp_dir, "best_model.pth")
    if not os.path.exists(mp):
        mp = os.path.join(exp_dir, "best_model_bce.pth")
    print("  Loading:", mp)
    
    device = torch.device("cuda")
    ckpt = torch.load(mp, map_location=device, weights_only=False)
    model = G.NLOSGAT(11, 128, 8, 2, 0.1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    outputs = []
    gnss_idx = {"GPS": 7, "Glonass": 8, "Galileo": 9, "BeiDou": 10}
    
    for ep_idx, ep in enumerate(epochs):
        obs = ep["obs"]
        N = len(obs)
        nf = np.zeros((N, 11), dtype=np.float32)
        for i, o in enumerate(obs):
            nf[i, 0] = o["elevation_deg"] / 90.0
            nf[i, 1] = o["azimuth_deg"] / 360.0
            nf[i, 2] = o["cno"] / 60.0
            nf[i, 3] = o["pr_stdev_m"] / 5.0
            nf[i, 4] = o["pr_mes_m"] / 3e7
            nf[i, 6] = np.cos(np.radians(o["elevation_deg"]))
            idx = gnss_idx.get(o["gnss"], 7)
            nf[i, idx] = 1.0
        
        # Build edges
        az = np.array([o["azimuth_deg"] for o in obs])
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                d = abs(az[i] - az[j])
                if d > 180: d = 360 - d
                if d < 90: edges.extend([[i,j], [j,i]])
        if len(edges) == 0:
            ei = torch.tensor([[i,i] for i in range(N)], dtype=torch.long, device=device)
        else:
            ei = torch.tensor(edges, dtype=torch.long).T.to(device)
        
        nf_t = torch.tensor(nf, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            p, mu, sl, sn = model(nf_t, ei)
        
        outputs.append({
            "p_los": p.cpu().numpy().flatten(),
            "mu_nlos": mu.cpu().numpy().flatten(),
            "sigma_los": torch.exp(sl).cpu().numpy().flatten(),
            "sigma_nlos": torch.exp(sn).cpu().numpy().flatten(),
        })
        
        if (ep_idx + 1) % 100 == 0:
            print("  ...", ep_idx + 1, "/", len(epochs))
    
    return outputs

if __name__ == "__main__":
    print("Loading HK data...")
    epochs = load_hk_epoch_data()
    print("Loaded", len(epochs), "epochs")
    print("\nRunning MoG inference...")
    t0 = time.time()
    outputs = run_hk_mog_inference(epochs)
    print("Done in {:.1f}s".format(time.time()-t0))
    
    # Quick stats
    all_p = np.concatenate([o["p_los"] for o in outputs])
    print("p_los mean={:.4f} std={:.4f}".format(all_p.mean(), all_p.std()))

# -*- coding: utf-8 -*-
"""
run_all.py — Unified Pipeline: Module 1 Inference + Module 2 Localization
=========================================================================
Handles 5 datasets: berlin1, berlin2, frankfurt1, frankfurt2, UrbanNav-HK_TST
Results saved to part2_FactorGraphLocalizationFusion/result/exp_hk_eval/
"""
import os, sys, json, time, pickle, numpy as np, torch

# Paths
_PART2_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_PART2_DIR, "fusion")
sys.path.insert(0, _PART2_DIR)
sys.path.insert(0, _MODEL_DIR)

_PART1_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model"
sys.path.insert(0, _PART1_DIR)

from fusion.utils import load_epoch_data, load_mog_model as _load_mog_model, run_mog_inference as _run_mog_inference
from fusion.evaluate_fusion import evaluate_all_methods, generate_report_table
import GAT_V2025 as G
from config import Config

# === Dataset→Experiment Mapping ===
DATASET_EXP_MAP = {
    "berlin1_potsdamer_platz": "exp_001",
    "berlin2_gendarmenmarkt": "exp_002",
    "frankfurt1_maintower": "exp_003",
    "frankfurt2_westendtower": "exp_004",
    "UrbanNav-HK_TST": "exp_hk",
}

# === HK Data Adapter ===
def load_hk_epoch_data():
    """Load HK data in Module 2-compatible format."""
    hk_base = r"D:\3_document\4_research\NLOS Signal Identification and Correction\data\processedData\UrbanNav-HK_TST\processed"
    with open(os.path.join(hk_base, "train_dataset.pkl"), "rb") as f:
        train = pickle.load(f)
    with open(os.path.join(hk_base, "val_dataset.pkl"), "rb") as f:
        val = pickle.load(f)
    
    # Convert to Module 2 format
    def _lla_to_ecef(lat, lon, alt):
        a, f = 6378137.0, 1.0/298.257223563
        lat_r, lon_r = np.radians(lat), np.radians(lon)
        e2 = 2*f - f*f
        N = a / np.sqrt(1 - e2 * np.sin(lat_r)**2)
        x = (N + alt) * np.cos(lat_r) * np.cos(lon_r)
        y = (N + alt) * np.cos(lat_r) * np.sin(lon_r)
        z = (N*(1-e2) + alt) * np.sin(lat_r)
        return np.array([x/1000.0, y/1000.0, z/1000.0])  # km
    
    result = []
    for d in train + val:
        gt_ecef = _lla_to_ecef(d["gt_lat"], d["gt_lon"], d["gt_alt"])
        nf = d["node_features"]
        nl = d["nlos_labels"]
        pe = d["pseudorange_errors"]
        N = d["num_satellites"]
        
        obs_list = []
        for i in range(N):
            elev = float(nf[i, 0] * 90.0)
            az = float(nf[i, 1] * 360.0)
            cno = float(nf[i, 2] * 60.0)
            # Derive GNSS from one-hot (dims 7-10)
            gnss_map = {7: "GPS", 8: "Glonass", 9: "Galileo", 10: "BeiDou"}
            gnss = "GPS"
            for j, name in gnss_map.items():
                if nf[i, j] > 0.5:
                    gnss = name
                    break
            obs_list.append({
                "svid": i,
                "gnss": gnss,
                "pr_mes_m": float(nf[i, 4] * 3e7),
                "cno": cno,
                "pr_stdev_m": 1.0,
                "nlos_label": int(nl[i]),
                "elevation_deg": elev,
                "azimuth_deg": az,
            })
        result.append({
            "gps_week": d["gps_week"],
            "gps_seconds": d["gps_seconds"],
            "gt_ecef": gt_ecef,
            "obs": obs_list,
        })
    return result


def run_hk_mog_inference(all_epochs, exp_name):
    """Run MoG inference for HK using UrbanNavDataset approach."""
    from run_urbannav import UrbanNavDataset, load_hk_data as _load_hk
    from torch.utils.data import DataLoader
    
    exp_dir = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result" + "\\" + exp_name
    mp = os.path.join(exp_dir, "best_model.pth")
    if not os.path.exists(mp):
        mp = os.path.join(exp_dir, "best_model_bce.pth")
    
    print("  Loading MoG model:", mp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(mp, map_location=device, weights_only=False)
    
    model = G.NLOSGAT(11, 128, 8, 2, 0.1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Build dataset from epoch data
    class EpochDataset(torch.utils.data.Dataset):
        def __init__(self, epochs):
            self.data = []
            for ep in epochs:
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
                    gnss_idx = {"GPS": 7, "Glonass": 8, "Galileo": 9, "BeiDou": 10}.get(o["gnss"], 7)
                    nf[i, gnss_idx] = 1.0
                # Build edges
                az = np.array([o["azimuth_deg"] for o in obs])
                edges = []
                for i in range(N):
                    for j in range(i+1, N):
                        d = abs(az[i] - az[j])
                        if d > 180: d = 360 - d
                        if d < 90:
                            edges.extend([[i,j], [j,i]])
                if len(edges) == 0:
                    ei = np.array([[i,i] for i in range(N)], dtype=np.int64).T
                else:
                    ei = np.array(edges, dtype=np.int64).T
                ea = np.ones(ei.shape[1], dtype=np.float32) * 0.5
                self.data.append({"nf": nf, "ei": ei, "ea": ea})
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            d = self.data[idx]
            return (torch.tensor(d["nf"]), torch.tensor(d["ei"]),
                    torch.tensor(d["ea"]), torch.zeros(len(d["nf"])),
                    torch.zeros(len(d["nf"])))
    
    ds = EpochDataset(all_epochs)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                    collate_fn=G.batch_collate_fn, drop_last=False)
    
    mog_outputs = []
    with torch.no_grad():
        for nf, ei, ea, pe, lb in dl:
            if nf.size(0) == 0: continue
            nf, ei = nf.to(device), ei.to(device)
            p, mu, sl, sn = model(nf, ei)
            p_np = p.cpu().numpy().flatten()
            mu_np = mu.cpu().numpy().flatten()
            sl_np = torch.exp(sl).cpu().numpy().flatten()
            sn_np = torch.exp(sn).cpu().numpy().flatten()
            
            # Split back by epoch (batch_collate_fn creates one big graph)
            offset = 0
            for d in ds.data[len(mog_outputs)*64:]:
                N = d["nf"].shape[0]
                if offset + N > len(p_np): break
                mog_outputs.append({
                    "p_los": p_np[offset:offset+N],
                    "mu_nlos": mu_np[offset:offset+N],
                    "sigma_los": sl_np[offset:offset+N],
                    "sigma_nlos": sn_np[offset:offset+N],
                })
                offset += N
    
    return mog_outputs


def run_single_dataset(dataset_name, exp_name, result_dir):
    print("\n" + "#"*60)
    print("# Dataset:", dataset_name, "(exp:", exp_name + ")")
    print("#"*60)
    
    is_hk = (dataset_name == "UrbanNav-HK_TST")
    
    # [1] Load data
    print("\n[1/4] Loading epoch data ...")
    t0 = time.time()
    if is_hk:
        all_epochs = load_hk_epoch_data()
    else:
        all_epochs = load_epoch_data(dataset_name)
    print("  Loaded", len(all_epochs), "epochs ({:.1f}s)".format(time.time()-t0))
    
    # [2] MoG inference
    print("\n[2/4] Module 1 MoG inference ...")
    t0 = time.time()
    if is_hk:
        mog_outputs = run_hk_mog_inference(all_epochs, exp_name)
    else:
        model, config, device = _load_mog_model(exp_name)
        print("  Model loaded:", exp_name + "/best_model.pth")
        mog_outputs = []
        for i, ep in enumerate(all_epochs):
            mog = _run_mog_inference(model, config, device, ep)
            mog_outputs.append(mog)
            if (i+1) % 500 == 0:
                print("  ...", i+1, "/", len(all_epochs))
    print("  Inference complete ({:.1f}s)".format(time.time()-t0))
    
    # [3] Positioning
    print("\n[3/4] Running positioning methods (LS / WLS / FG) ...")
    t0 = time.time()
    results = evaluate_all_methods(all_epochs, mog_outputs, dataset_name, result_dir)
    print("  Evaluation complete ({:.1f}s)".format(time.time()-t0))
    
    # [4] Save
    print("\n[4/4] Saving results ...")
    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def main():
    print("="*60)
    print("Unified Pipeline: Module 1 + Module 2 (5 Datasets)")
    print("="*60)
    
    result_root = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\result"
    exp_dirs = sorted([d for d in os.listdir(result_root) if d.startswith("exp_") and os.path.isdir(os.path.join(result_root, d))])
    exp_id = len(exp_dirs) + 1
    exp_dir = os.path.join(result_root, "exp_" + str(exp_id).zfill(3))
    os.makedirs(exp_dir, exist_ok=True)
    print("Experiment:", "exp_" + str(exp_id).zfill(3))
    
    all_results = {}
    total_start = time.time()
    
    for ds_name, exp_name in DATASET_EXP_MAP.items():
        ds_result_dir = os.path.join(exp_dir, ds_name.split("_")[0] if "_" in ds_name else ds_name)
        os.makedirs(ds_result_dir, exist_ok=True)
        
        try:
            results = run_single_dataset(ds_name, exp_name, ds_result_dir)
            short = ds_name.split("_")[0] if "_" in ds_name else ds_name[:8]
            all_results[short] = results
        except Exception as e:
            print("  ERROR:", e)
            import traceback; traceback.print_exc()
    
    # Report
    print("\n" + "="*60)
    print("Comparison Report")
    print("="*60)
    if all_results:
        report_path = os.path.join(exp_dir, "comparison_report.md")
        report = generate_report_table(all_results, report_path)
        print(report)
    
    total_t = (time.time() - total_start) / 60
    print("\nTotal time: {:.1f} min".format(total_t))
    print("Results:", exp_dir)


if __name__ == "__main__":
    main()

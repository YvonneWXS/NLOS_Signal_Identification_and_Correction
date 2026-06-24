"""Generate predictions.json for European experiments from best models."""
import os, sys, json, pickle, numpy as np, torch

sys.path.insert(0, r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model")
import GAT_V2025 as G

RESULT_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result"
EXPS = {
    "exp_001": "berlin1_potsdamer_platz",
    "exp_002": "berlin2_gendarmenmarkt",
    "exp_003": "frankfurt1_maintower",
    "exp_004": "frankfurt2_westendtower",
}

for exp_name, ds_name in EXPS.items():
    exp_dir = os.path.join(RESULT_DIR, exp_name)
    mp = os.path.join(exp_dir, "best_model.pth")
    if not os.path.exists(mp):
        print(f"SKIP {exp_name}: no best_model.pth")
        continue
    
    # Load data
    pkl_path = rf"D:\3_document\4_research\NLOS Signal Identification and Correction\data\processedData\{ds_name}_processed.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # Model
    device = torch.device("cuda")
    ckpt = torch.load(mp, map_location=device, weights_only=False)
    model = G.NLOSGAT(11, 128, 8, 2, 0.1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    total_nlos, total_obs = 0, 0
    gmap = {"GPS": 7, "Glonass": 8, "Galileo": 9, "BeiDou": 10}
    
    # Validate (last 20%)
    split = int(len(data) * 0.8)
    val_data = data[split:]
    
    ap, al = [], []
    for ep in val_data:
        obs = ep.observations
        N = len(obs)
        nf = np.zeros((N, 11), dtype=np.float32)
        for i, o in enumerate(obs):
            nf[i,0]=o.elevation/90; nf[i,1]=o.azimuth/360
            nf[i,2]=o.cno/60; nf[i,3]=o.pr_stdev/5; nf[i,4]=o.pr_mes/3e7
            nf[i,5]=o.pseudorange_error/100; nf[i,6]=np.cos(np.radians(o.elevation))
            idx = gmap.get(o.gnss_id, 7)
            if 7 <= idx <= 10: nf[i, idx] = 1.0
        
        az = np.array([o.azimuth for o in obs])
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                d = abs(az[i] - az[j])
                if d > 180: d = 360 - d
                if d < 90: edges.extend([[i,j],[j,i]])
        if len(edges) == 0:
            ei = torch.tensor([[i,i] for i in range(N)], dtype=torch.long, device=device)
        else:
            ei = torch.tensor(edges, dtype=torch.long).T.to(device)
        
        with torch.no_grad():
            p, mu, sl, sn = model(torch.tensor(nf, device=device), ei)
        
        ap.append(p.cpu().numpy().flatten())
        al.append(np.array([o.nlos_label for o in obs], dtype=np.float32))
        total_nlos += sum(1 for o in obs if o.nlos_label == 1)
        total_obs += N
    
    pl = np.concatenate(ap); lb = np.concatenate(al)
    pred = (pl < 0.5).astype(int)
    tp = ((pred==1)&(lb==1)).sum(); fp = ((pred==1)&(lb==0)).sum()
    tn = ((pred==0)&(lb==0)).sum(); fn = ((pred==0)&(lb==1)).sum()
    acc = (tp+tn)/len(lb); prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0; f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    los_m = lb == 0; nlos_m = lb == 1
    p_gap = pl[los_m].mean() - pl[nlos_m].mean() if nlos_m.any() else 0
    
    results = {
        "p_los": pl.tolist(), "labels": lb.tolist(),
        "metrics": {"accuracy": float(acc), "f1": float(f1), "precision": float(prec),
                    "recall": float(rec), "plos_gap": float(p_gap)},
        "dataset_stats": {"epochs": len(val_data), "total_sats": len(lb),
                          "nlos_count": total_nlos, "nlos_pct": float(100*total_nlos/total_obs)},
    }
    out_path = os.path.join(exp_dir, "predictions.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{exp_name}: Acc={acc:.4f} F1={f1:.4f} p_gap={p_gap:.4f} | saved predictions.json")

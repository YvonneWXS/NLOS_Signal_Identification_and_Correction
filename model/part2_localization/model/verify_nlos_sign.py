import os, sys, pickle, json, numpy as np
sys.path.insert(0, r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\model")
from utils import load_epoch_data, compute_satellite_positions

CACHE = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\cache"
DATASETS = ["berlin1_potsdamer_platz","berlin2_gendarmenmarkt","frankfurt1_maintower","frankfurt2_westendtower"]

results = {}
for ds in DATASETS:
    print(f"\n{'='*60}\n{ds}\n{'='*60}")
    all_epochs = load_epoch_data(ds)
    mp = os.path.join(CACHE, f"{ds}_mog_outputs.pkl")
    with open(mp, "rb") as f: mog_outputs = pickle.load(f)
    
    los_errs, nlos_errs = [], []
    mu_m1_nlos, mu_emp_nlos = [], []
    p_los_binned = {i: [] for i in range(5)}  # [0,0.2),[0.2,0.4),...
    nlos_binned = {i: [] for i in range(5)}
    
    for ep, mog in zip(all_epochs, mog_outputs):
        if mog is None: continue
        sv_pos, _ = compute_satellite_positions(ep, ds)
        pr = np.array([o["pr_mes_m"]/1000.0 for o in ep["obs"]])
        nlos_lbl = np.array([o["nlos_label"] for o in ep["obs"]])
        if len(pr) < 4: continue
        
        # LS for clock estimate + position
        x0 = np.zeros(4); x0[:3] = ep["gt_ecef"]
        try:
            from scipy.linalg import lstsq
            dist0 = np.linalg.norm(sv_pos - x0[:3], axis=1)
            H = np.zeros((len(pr),4))
            H[:,:3] = -(sv_pos - x0[:3]) / np.maximum(dist0[:,None], 1e-8)
            H[:,3] = 1.0
            res = pr - dist0
            delta = lstsq(H.T @ H, H.T @ res)[0]
            x = x0 + delta
        except:
            continue
        
        dist = np.linalg.norm(sv_pos - x[:3], axis=1)
        pr_err = (pr - dist - x[3]) * 1000  # meters
        
        los_mask = nlos_lbl == 0
        nlos_mask = nlos_lbl == 1
        los_errs.extend(pr_err[los_mask].tolist())
        nlos_errs.extend(pr_err[nlos_mask].tolist())
        
        # mu_nlos comparison
        mu = mog.get("mu_nlos", np.zeros(len(pr)))
        mu_m1_nlos.extend(mu[nlos_mask].tolist())
        mu_emp_nlos.extend(np.maximum(pr_err[nlos_mask], 0).tolist())
        
        # p_los binning
        p_los = mog.get("p_los_sharp", mog["p_los"])
        for j in range(len(pr)):
            b = min(int(p_los[j] / 0.2), 4)
            p_los_binned[b].append(abs(pr_err[j]))
            nlos_binned[b].append(nlos_lbl[j])
    
    los_errs = np.array(los_errs)
    nlos_errs = np.array(nlos_errs)
    
    print(f"  Check 1: NLOS error sign")
    print(f"    LOS:  mean={np.mean(los_errs):.0f}m, P50={np.percentile(los_errs,50):.0f}m, frac>0={np.mean(los_errs>0)*100:.0f}%")
    print(f"    NLOS: mean={np.mean(nlos_errs):.0f}m, P50={np.percentile(nlos_errs,50):.0f}m, frac>0={np.mean(nlos_errs>0)*100:.0f}%")
    print(f"    NLOS mean|err|>300m: {np.mean(np.abs(nlos_errs)>300)*100:.0f}%")
    
    print(f"  Check 2: p_los vs |error|")
    for b in range(5):
        vals = np.array(p_los_binned[b])
        nl = np.array(nlos_binned[b])
        print(f"    p_los [{b*0.2:.1f}-{(b+1)*0.2:.1f}): N={len(vals):5d}, mean|e|={np.mean(vals):.0f}m, %NLOS={np.mean(nl)*100:.0f}%")
    
    print(f"  Check 3: mu_nlos quality")
    m1_mu = np.mean(np.array(mu_m1_nlos))
    emp_mu = np.mean(np.array(mu_emp_nlos))
    print(f"    Module 1 mu_nlos mean: {m1_mu*1000:.0f}m")
    print(f"    Empirical mu_nlos mean: {emp_mu:.0f}m")
    print(f"    Underestimate ratio: {emp_mu/max(m1_mu*1000,1):.1f}x")
    
    results[ds] = {
        "los_frac_positive": float(np.mean(los_errs>0)),
        "nlos_frac_positive": float(np.mean(nlos_errs>0)),
        "nlos_mean_error_m": float(np.mean(nlos_errs)),
        "nlos_mean_positive_m": float(np.mean(nlos_errs[nlos_errs>0])),
        "mu_m1_mean_m": float(m1_mu * 1000),
        "mu_emp_mean_m": float(emp_mu),
    }

with open(os.path.join(CACHE, "nlos_sign_analysis.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for ds in DATASETS:
    r = results[ds]
    status = "CONFIRMED" if r["nlos_frac_positive"] > 0.65 else "WEAK"
    print(f"  {ds}: NLOS>0={r['nlos_frac_positive']*100:.0f}% [{status}], mean NLOS={r['nlos_mean_error_m']:.0f}m, mu_ratio={r['mu_emp_mean_m']/max(r['mu_m1_mean_m'],1):.1f}x underestimate")
print("\nSaved to cache/nlos_sign_analysis.json")

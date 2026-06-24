import os, sys, json, pickle, time, numpy as np

_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _MODEL_DIR)
sys.path.insert(0, os.path.dirname(_MODEL_DIR))
sys.path.insert(0, r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model")

from utils import load_epoch_data, load_mog_model, run_mog_inference, compute_satellite_positions
from baselines import solve_standard_ls, solve_wls_mog

RESULT_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\result"
CACHE_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DATASETS = {
    "berlin1_potsdamer_platz": "exp_034",
    "berlin2_gendarmenmarkt": "exp_035",
    "frankfurt1_maintower": "exp_038",
    "frankfurt2_westendtower": "exp_039",
}

def compute_pdop(H, W=None):
    """PDOP from design matrix H (Nx4). If W given, use weighted covariance."""
    if W is None:
        try:
            P = np.linalg.inv(H.T @ H)
        except np.linalg.LinAlgError:
            return float("nan")
    else:
        try:
            P = np.linalg.inv(H.T @ W @ H)
        except np.linalg.LinAlgError:
            return float("nan")
    return np.sqrt(max(P[0, 0] + P[1, 1] + P[2, 2], 0))

def run_diagnosis():
    all_diagnoses = {}
    for ds_name, exp_name in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"DIAGNOSIS: {ds_name} ({exp_name})")
        print(f"{'='*70}")

        # Load cached MoG outputs or compute
        cache_path = os.path.join(CACHE_DIR, f"{ds_name}_mog_outputs.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                mog_outputs = pickle.load(f)
            print(f"  Loaded {len(mog_outputs)} cached MoG outputs")
            all_epochs = load_epoch_data(ds_name)
        else:
            all_epochs = load_epoch_data(ds_name)
            model, config, device = load_mog_model(exp_name)
            mog_outputs = []
            for i, ep in enumerate(all_epochs):
                mog = run_mog_inference(model, config, device, ep)
                mog_outputs.append(mog)
                if (i + 1) % 500 == 0:
                    print(f"  Inference: {i+1}/{len(all_epochs)}")
            with open(cache_path, "wb") as f:
                pickle.dump(mog_outputs, f)
            print(f"  Cached {len(mog_outputs)} MoG outputs")

        n = len(all_epochs)
        short = ds_name.split("_")[0]

        # ---- Diagnosis A: Weight distribution ----
        print(f"\n  --- A: Weight Distribution ---")
        w_los_all, w_nlos_all = [], []
        max_min_ratios = []
        all_weights = []

        for ep, mog in zip(all_epochs, mog_outputs):
            if mog is None or len(mog.get("p_los", [])) == 0:
                continue
            nlos_labels = np.array([o["nlos_label"] for o in ep["obs"]])
            p_los = np.array(mog.get("p_los_sharp", mog["p_los"]))
            sigma_los = np.array(mog["sigma_los"])
            # WLS-MoG weight formula
            w = np.clip(p_los, 0.01, None) / np.maximum(sigma_los, 0.01) ** 2

            los_mask = nlos_labels == 0
            nlos_mask = nlos_labels == 1
            if los_mask.sum() > 0:
                w_los_all.extend(w[los_mask].tolist())
            if nlos_mask.sum() > 0:
                w_nlos_all.extend(w[nlos_mask].tolist())

            if w.max() > 0 and w.min() > 0:
                max_min_ratios.append(w.max() / w.min())
            all_weights.extend(w.tolist())

        w_los_mean = np.mean(w_los_all) if w_los_all else 0
        w_nlos_mean = np.mean(w_nlos_all) if w_nlos_all else 0
        ratio = w_los_mean / w_nlos_mean if w_nlos_mean > 0 else float("inf")
        frac_near_uniform = np.mean(np.array(max_min_ratios) < 2.0)

        print(f"    mean weight LOS:  {w_los_mean:.4f}")
        print(f"    mean weight NLOS: {w_nlos_mean:.4f}")
        print(f"    LOS/NLOS ratio:   {ratio:.2f}  {'<<< DISCRIMINATION TOO WEAK' if ratio < 2.0 else ''}")
        print(f"    Frac near-uniform: {frac_near_uniform:.2%} (max/min < 2.0)")

        # Histogram
        hist, bins = np.histogram(all_weights, bins=10)
        print(f"    Weight histogram (10 bins):")
        for i in range(len(hist)):
            bar = "#" * int(hist[i] / max(hist) * 40)
            print(f"      [{bins[i]:.2f}-{bins[i+1]:.2f}]: {hist[i]:6d} {bar}")

        # ---- Diagnosis B: DOP ----
        print(f"\n  --- B: Geometric Dilution ---")
        pdop_std_list, pdop_mog_list = [], []
        frac_worse_dop = 0

        for i, (ep, mog) in enumerate(zip(all_epochs, mog_outputs)):
            if mog is None or len(mog.get("p_los", [])) == 0:
                continue
            sv_pos, _ = compute_satellite_positions(ep, ds_name)
            pr = np.array([o["pr_mes_m"] / 1000.0 for o in ep["obs"]])
            if len(pr) < 4:
                continue

            # Standard LS PDOP
            x0 = np.zeros(4)
            x0[:3] = ep["gt_ecef"]
            try:
                dist = np.linalg.norm(sv_pos - x0[:3], axis=1)
                H = np.zeros((len(pr), 4))
                H[:, :3] = -(sv_pos - x0[:3]) / np.maximum(dist[:, None], 1e-8)
                H[:, 3] = 1.0
            except Exception:
                continue

            pdop_std = compute_pdop(H)
            pdop_std_list.append(pdop_std)

            # WLS-MoG PDOP
            p_los = np.array(mog.get("p_los_sharp", mog["p_los"]))
            sigma_los = np.array(mog["sigma_los"])
            w = np.clip(p_los, 0.01, None) / np.maximum(sigma_los, 0.01) ** 2
            W = np.diag(w)
            pdop_mog = compute_pdop(H, W)
            pdop_mog_list.append(pdop_mog)

            if pdop_mog > pdop_std * 1.1:
                frac_worse_dop += 1

        pdop_std_mean = np.mean(pdop_std_list)
        pdop_mog_mean = np.mean(pdop_mog_list)
        frac_worse_dop_pct = frac_worse_dop / max(len(pdop_std_list), 1)

        print(f"    Mean PDOP Standard: {pdop_std_mean:.2f}")
        print(f"    Mean PDOP MoG:      {pdop_mog_mean:.2f}")
        print(f"    DOP inflation %:    {frac_worse_dop_pct:.1%} (MoG > Std x1.1)")

        # ---- Diagnosis C: Clock coupling ----
        print(f"\n  --- C: Clock Bias Coupling ---")
        delta_clk_list = []
        delta_err_list = []

        for i, (ep, mog) in enumerate(zip(all_epochs, mog_outputs)):
            if mog is None or len(mog.get("p_los", [])) == 0:
                continue
            sv_pos, _ = compute_satellite_positions(ep, ds_name)
            pr = np.array([o["pr_mes_m"] / 1000.0 for o in ep["obs"]])
            if len(pr) < 4:
                continue

            try:
                x_std = solve_standard_ls(sv_pos, pr)
            except Exception:
                continue
            clk_std = x_std[3] * 1000.0
            err_std = np.linalg.norm((x_std[:3] - ep["gt_ecef"]) * 1000.0)

            p_los = np.array(mog.get("p_los_sharp", mog["p_los"]))
            sigma_los = np.array(mog["sigma_los"])
            try:
                x_mog = solve_wls_mog(sv_pos, pr, p_los, sigma_los)
            except Exception:
                continue
            clk_mog = x_mog[3] * 1000.0
            err_mog = np.linalg.norm((x_mog[:3] - ep["gt_ecef"]) * 1000.0)

            delta_clk_list.append(clk_mog - clk_std)
            delta_err_list.append(err_mog - err_std)

        delta_clk_arr = np.array(delta_clk_list)
        delta_err_arr = np.array(delta_err_list)
        mean_abs_delta_clk = np.mean(np.abs(delta_clk_arr))
        std_delta_clk = np.std(delta_clk_arr)
        corr = np.corrcoef(np.abs(delta_clk_arr), delta_err_arr)[0, 1] if len(delta_clk_arr) > 2 else 0

        print(f"    Mean |delta_clk|: {mean_abs_delta_clk:.1f} m")
        print(f"    Std delta_clk:    {std_delta_clk:.1f} m")
        print(f"    Corr(|clk|, err): {corr:.3f}")

        # ---- Diagnosis D: Per-satellite residuals ----
        print(f"\n  --- D: Per-Satellite Residuals ---")
        res_los_std, res_nlos_std = [], []
        res_los_mog, res_nlos_mog = [], []

        for i, (ep, mog) in enumerate(zip(all_epochs, mog_outputs)):
            if mog is None or len(mog.get("p_los", [])) == 0:
                continue
            sv_pos, _ = compute_satellite_positions(ep, ds_name)
            pr = np.array([o["pr_mes_m"] / 1000.0 for o in ep["obs"]])
            nlos_labels = np.array([o["nlos_label"] for o in ep["obs"]])
            if len(pr) < 4:
                continue

            try:
                x_std = solve_standard_ls(sv_pos, pr)
            except Exception:
                continue
            dist_std = np.linalg.norm(sv_pos - x_std[:3], axis=1)
            res_std = (pr - dist_std - x_std[3]) * 1000.0
            los_mask = nlos_labels == 0
            nlos_mask = nlos_labels == 1
            res_los_std.extend(np.abs(res_std[los_mask]).tolist())
            res_nlos_std.extend(np.abs(res_std[nlos_mask]).tolist())

            p_los = np.array(mog.get("p_los_sharp", mog["p_los"]))
            sigma_los = np.array(mog["sigma_los"])
            try:
                x_mog = solve_wls_mog(sv_pos, pr, p_los, sigma_los)
            except Exception:
                continue
            dist_mog = np.linalg.norm(sv_pos - x_mog[:3], axis=1)
            res_mog = (pr - dist_mog - x_mog[3]) * 1000.0
            res_los_mog.extend(np.abs(res_mog[los_mask]).tolist())
            res_nlos_mog.extend(np.abs(res_mog[nlos_mask]).tolist())

        print(f"    Standard LS: |res| LOS={np.mean(res_los_std):.1f}m NLOS={np.mean(res_nlos_std):.1f}m")
        print(f"    WLS-MoG:     |res| LOS={np.mean(res_los_mog):.1f}m NLOS={np.mean(res_nlos_mog):.1f}m")
        nlos_reduction = (np.mean(res_nlos_std) - np.mean(res_nlos_mog)) / max(np.mean(res_nlos_std), 1e-6) * 100
        print(f"    NLOS residual reduction: {nlos_reduction:+.1f}%")

        # ---- Summary ----
        causes = []
        if ratio < 2.0:
            causes.append("WEIGHT_NOT_DISCRIMINATIVE")
        if frac_worse_dop_pct > 0.3:
            causes.append("DOP_INFLATION")
        if mean_abs_delta_clk > 50 and corr > 0.3:
            causes.append("CLOCK_COUPLING")
        if not causes:
            causes.append("UNKNOWN")

        primary = causes[0]
        secondary = causes[1] if len(causes) > 1 else "NONE"
        print(f"\n  >>> DIAGNOSIS {short}: PRIMARY={primary} SECONDARY={secondary}")

        all_diagnoses[short] = {
            "primary": primary,
            "secondary": secondary,
            "ratio": ratio,
            "frac_dop_inflation": frac_worse_dop_pct,
            "mean_abs_delta_clk": float(mean_abs_delta_clk),
            "corr_clk_err": float(corr),
            "nlos_residual_reduction": float(nlos_reduction),
        }

    # Final summary
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*70}")
    for ds, d in all_diagnoses.items():
        print(f"  {ds}: PRIMARY={d['primary']}, SECONDARY={d['secondary']}")
        print(f"        ratio={d['ratio']:.2f}, DOP_infl={d['frac_dop_inflation']:.1%}, |dClk|={d['mean_abs_delta_clk']:.0f}m, corr={d['corr_clk_err']:.3f}")

    # Save
    with open(os.path.join(CACHE_DIR, "diagnosis_v4.json"), "w") as f:
        json.dump(all_diagnoses, f, indent=2)
    print("\nSaved to cache/diagnosis_v4.json")

if __name__ == "__main__":
    run_diagnosis()

"""generate_report.py -- Generate Markdown report + env.md for experiment"""
import os, sys, json, time, argparse, platform
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from analyze_experiment import analyze_experiment

CFG_ATTRS = [
    "IN_FEATURES", "HIDDEN_FEATURES", "NUM_HEADS", "NUM_LAYERS", "DROPOUT",
    "LEARNING_RATE", "NUM_EPOCHS", "BATCH_SIZE", "GRADIENT_ACCUMULATION",
    "GRADIENT_CLIP", "AZIMUTH_THRESHOLD", "VALIDATION_SPLIT",
    "SIGMA_MIN", "SIGMA_MAX", "LAMBDA_BCE", "LAMBDA_ENTROPY",
    "LAMBDA_UNC", "LAMBDA_ELEVATION_PRIOR", "P_LOS_SMOOTHING",
    "LABEL_SMOOTHING", "POS_WEIGHT", "EARLY_STOPPING_PATIENCE",
    "USE_MIXTURE_GAUSSIAN"
]

FEAT_MAP = [
    ("elevation", "Elevation(deg)"),
    ("cno", "CNO(dBHz)"),
    ("pseudorange_error", "|prError|(km)"),
    ("pr_stdev", "prStdev(m)")
]

GNSS_LIST = ["GPS", "Glonass", "Galileo", "BeiDou"]

def generate_env_md(exp_dir, config, checkpoint, elapsed_sec, dataset_name):
    md = ["# Environment & Experiment Info", ""]
    md.append("| Item | Value |")
    md.append("|------|-------|")
    md.append(f"| Dataset | {dataset_name} |")
    md.append(f"| Experiment | {os.path.basename(exp_dir)} |")
    md.append(f"| Python | {sys.version.split()[0]} |")
    md.append(f"| PyTorch | {torch.__version__} |")
    try:
        import numpy as _np
        md.append(f"| NumPy | {_np.__version__} |")
    except: pass
    try:
        import pandas as _pd
        md.append(f"| Pandas | {_pd.__version__} |")
    except: pass
    md.append(f"| CUDA Available | {torch.cuda.is_available()} |")
    if torch.cuda.is_available():
        md.append(f"| GPU | {torch.cuda.get_device_name(0)} |")
        md.append(f"| CUDA Version | {torch.version.cuda} |")
    md.append(f"| Platform | {platform.system()} {platform.release()} |")
    md.append("")
    md.append("## Training Config")
    md.append("")
    md.append("| Parameter | Value |")
    md.append("|-----------|-------|")
    for attr in CFG_ATTRS:
        val = getattr(config, attr, "N/A")
        md.append(f"| {attr} | {val} |")
    md.append("")
    md.append("## Training Summary")
    md.append("")
    md.append("| Item | Value |")
    md.append("|------|-------|")
    md.append(f"| Training time | {elapsed_sec/60:.1f} min ({elapsed_sec:.0f}s) |")
    if "epoch" in checkpoint:
        md.append(f"| Best epoch | {checkpoint['epoch']+1} |")
    if "val_loss" in checkpoint:
        md.append(f"| Best val loss | {checkpoint['val_loss']:.4f} |")
    path = os.path.join(exp_dir, "env.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"env.md saved to {path}")

def generate_result_md(exp_dir, dataset_name, results):
    md = [f"# Experiment Analysis: {dataset_name}", ""]
    md.append("## 1. Classification Performance")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Accuracy | {results['accuracy']:.4f} |")
    md.append(f"| Precision | {results['precision']:.4f} |")
    md.append(f"| Recall | {results['recall']:.4f} |")
    md.append(f"| F1 | {results['f1']:.4f} |")
    md.append(f"| TP | {results['tp']} |")
    md.append(f"| FP | {results['fp']} |")
    md.append(f"| FN | {results['fn']} |")
    md.append(f"| TN | {results['tn']} |")
    md.append(f"| Samples (LOS/NLOS) | {results['num_los']} / {results['num_nlos']} |")
    md.append(f"| NLOS Ratio | {results['nlos_ratio']:.3f} |")
    md.append("")

    md.append("## 2. p_los Distribution")
    md.append("")
    md.append("| Group | Mean | Count |")
    md.append("|-------|------|-------|")
    md.append(f"| All | {results['p_los_all_mean']:.4f} | {results['num_samples']} |")
    md.append(f"| LOS | {results['p_los_los_avg']:.4f} | {results['num_los']} |")
    md.append(f"| NLOS | {results['p_los_nlos_avg']:.4f} | {results['num_nlos']} |")
    md.append(f"| **Gap (LOS-NLOS)** | **{results['p_los_gap']:.4f}** | |")
    md.append("")
    bq = results.get("bimodal_quality", "N/A")
    lh = results.get("los_p_high_pct", 0)
    nl = results.get("nlos_p_low_pct", 0)
    md.append(f"**Bimodality**: {bq} (LOS>0.7: {lh:.1f}%, NLOS<0.3: {nl:.1f}%)")
    md.append("")

    md.append("## 3. Uncertainty (Sigma) Analysis")
    md.append("")
    sigma_gap = results["sigma_nlos_mean"] - results["sigma_los_mean"]
    if sigma_gap > 0.5:
        status = "OK"
    elif sigma_gap > 0:
        status = "WARNING: sigma diff < 0.5 km"
    else:
        status = "FAIL: sigma(NLOS) <= sigma(LOS)"
    md.append("| Group | Mean Sigma |")
    md.append("|-------|------------|")
    md.append(f"| All | {results['sigma_mean']:.2f} km |")
    md.append(f"| LOS | {results['sigma_los_mean']:.2f} km |")
    md.append(f"| NLOS | {results['sigma_nlos_mean']:.2f} km |")
    md.append(f"| **Gap (NLOS-LOS)** | **{sigma_gap:.2f} km** -- {status} |")
    md.append("")

    md.append("## 4. Error Case Analysis")
    md.append("")
    md.append(f"### Type A: NLOS predicted as LOS (FN, p_los>0.5)")
    md.append(f"- Count: {results['fn_count']} / {results['num_nlos']} ({results['fn_pct']:.1f}%)")
    if "fn_top10_elev_mean" in results:
        md.append(f"- Top 10 mean elevation: {results['fn_top10_elev_mean']:.1f} deg")
        md.append(f"- Top 10 mean CNO: {results['fn_top10_cno_mean']:.1f} dBHz")
        md.append(f"- Top 10 mean |prError|: {results['fn_top10_prerr_mean']:.2f} km")
        md.append(f"- Top 10 mean prStdev: {results['fn_top10_prstdev_mean']:.2f} m")
    md.append("")
    md.append(f"### Type B: LOS predicted as NLOS (FP, p_los<0.5)")
    md.append(f"- Count: {results['fp_count']} / {results['num_los']} ({results['fp_pct']:.1f}%)")
    if "fp_top10_elev_mean" in results:
        md.append(f"- Top 10 mean elevation: {results['fp_top10_elev_mean']:.1f} deg")
        md.append(f"- Top 10 mean CNO: {results['fp_top10_cno_mean']:.1f} dBHz")
        md.append(f"- Top 10 mean |prError|: {results['fp_top10_prerr_mean']:.2f} km")
        md.append(f"- Top 10 mean prStdev: {results['fp_top10_prstdev_mean']:.2f} m")
    md.append("")

    md.append("## 5. Feature Comparison: Correct vs Error")
    md.append("")
    for label, label_name in [(1, "NLOS"), (0, "LOS")]:
        etype = "FN" if label == 1 else "FP"
        md.append(f"### {label_name} Samples (Correct vs {etype})")
        md.append("")
        md.append("| Feature | Correct Mean | Error Mean | Diff |")
        md.append("|---------|-------------|------------|------|")
        prefix = "nlos_" if label == 1 else "los_"
        for fkey, fname in FEAT_MAP:
            ck = f"{prefix}{fkey}_correct"
            ek = f"{prefix}{fkey}_error"
            if ck in results and ek in results:
                diff = results[ek] - results[ck]
                md.append(f"| {fname} | {results[ck]:.3f} | {results[ek]:.3f} | {diff:+.3f} |")
        md.append("")

    md.append("## 6. Per-Constellation Performance")
    md.append("")
    md.append("| GNSS | N | p_los Mean | LOS Avg | NLOS Avg | Gap |")
    md.append("|------|---|------------|---------|----------|-----|")
    for gnss in GNSS_LIST:
        nk = f"{gnss}_n"
        if nk in results:
            md.append(f"| {gnss} | {results[nk]} | "
                      f"{results[f'{gnss}_p_los_mean']:.4f} | "
                      f"{results[f'{gnss}_los_avg']:.4f} | "
                      f"{results[f'{gnss}_nlos_avg']:.4f} | "
                      f"{results[f'{gnss}_gap']:.4f} |")
    md.append("")

    path = os.path.join(exp_dir, "result.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"result.md saved to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--elapsed", type=float, default=0.0)
    args = parser.parse_args()

    config = get_config(DATASETS=[args.dataset])
    config.ensure_dirs()

    exp_dir = os.path.join(config.RESULT_DIR, args.exp)
    best_model = os.path.join(exp_dir, "best_model.pth")
    if not os.path.exists(best_model):
        print(f"ERROR: {best_model} not found")
        return

    checkpoint = torch.load(best_model, map_location="cpu", weights_only=False)
    generate_env_md(exp_dir, config, checkpoint, args.elapsed, args.dataset)

    print(f"Running analysis for {args.exp} / {args.dataset}...")
    results = analyze_experiment(args.exp, args.dataset, config)
    if results:
        json_path = os.path.join(exp_dir, f"analysis_{args.dataset}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"JSON saved to {json_path}")
        generate_result_md(exp_dir, args.dataset, results)
    else:
        print("Analysis failed!")

if __name__ == "__main__":
    main()

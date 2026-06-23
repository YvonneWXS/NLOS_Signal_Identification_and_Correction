# -*- coding: utf-8 -*-
"""
compare.py -- Cross-Dataset Comparison Module
==============================================
Usage: python compare.py --results_root <path> --metrics <m1,m2,...> --output <path>
       python compare.py --results_root <path> --all --output <path>

Available metrics: accuracy, precision, recall, f1, plos_gap, sigma_gap, mu_nlos, auc, best_epoch
"""
import os, sys, json, argparse, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_MAP = {
    "accuracy": ("Accuracy", "accuracy"),
    "precision": ("Precision", "precision"),
    "recall": ("Recall", "recall"),
    "f1": ("F1 Score", "f1"),
    "plos_gap": ("p_los Gap", "p_los_gap"),
    "sigma_gap": ("sigma_nlos Gap (km)", "sigma_gap"),
    "mu_nlos": ("mu_nlos NLOS (km)", "mu_nlos_nlos"),
    "auc": ("AUC-ROC", "auc"),
    "best_epoch": ("Best Epoch", "best_epoch"),
    "nlos_pct": ("NLOS Rate (%)", "nlos_pct"),
}


def load_all_results(results_root):
    """Load predictions.json from all subdirectories."""
    datasets = {}
    for subdir in sorted(os.listdir(results_root)):
        path = os.path.join(results_root, subdir)
        if not os.path.isdir(path):
            continue
        pred_file = os.path.join(path, "predictions.json")
        if os.path.exists(pred_file):
            with open(pred_file) as f:
                data = json.load(f)
            datasets[subdir] = data
    return datasets


def extract_metric(data, metric_key):
    """Extract metric value from predictions data."""
    metrics = data.get("metrics", {})
    ds_stats = data.get("dataset_stats", {})

    if metric_key == "mu_nlos_nlos":
        mu = np.array(data.get("mu_nlos", []))
        labels = np.array(data.get("labels", []))
        if len(mu) == 0 or len(labels) == 0 or len(mu) != len(labels):
            return 0.0
        return float(mu[labels == 1].mean()) if (labels == 1).any() else 0.0

    if metric_key == "nlos_pct":
        ds_stats = data.get("dataset_stats", {})
        return ds_stats.get("nlos_pct", 0.0)
    
    if metric_key == "sigma_gap":
        return metrics.get("sigma_gap", metrics.get("plos_gap", 0.0))

    if metric_key in metrics:
        return metrics[metric_key]

    return 0.0


def generate_comparison_table(datasets, metrics, output_path):
    """Generate a comparison table CSV and Markdown."""
    metric_names = [METRICS_MAP[m][0] for m in metrics if m in METRICS_MAP]
    metric_keys = [METRICS_MAP[m][1] for m in metrics if m in METRICS_MAP]

    # CSV
    csv_path = output_path.replace(".md", ".csv") if output_path.endswith(".md") else output_path + ".csv"
    with open(csv_path, "w") as f:
        f.write("Dataset," + ",".join(metric_names) + "\n")
        for ds_name, data in datasets.items():
            vals = [f"{extract_metric(data, k):.4f}" for k in metric_keys]
            f.write(f"{ds_name}," + ",".join(vals) + "\n")
    print(f"CSV saved: {csv_path}")

    # Markdown
    md_path = output_path if output_path.endswith(".md") else output_path + ".md"
    with open(md_path, "w") as f:
        f.write("# Cross-Dataset Comparison\n\n")
        f.write("| Dataset | " + " | ".join(metric_names) + " |\n")
        f.write("|---------|" + "|".join([":---:" for _ in metric_names]) + "|\n")
        for ds_name, data in datasets.items():
            vals = [f"{extract_metric(data, k):.4f}" for k in metric_keys]
            f.write(f"| {ds_name} | " + " | ".join(vals) + " |\n")

    print(f"Markdown saved: {md_path}")
    return md_path


def generate_comparison_bar_chart(datasets, metrics, output_dir):
    """Generate bar chart comparison for each metric."""
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        if metric not in METRICS_MAP:
            continue
        metric_name, metric_key = METRICS_MAP[metric]

        names = list(datasets.keys())
        values = [extract_metric(datasets[n], metric_key) for n in names]

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
        bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Comparison")
        ax.tick_params(axis="x", rotation=30)

        out_path = os.path.join(output_dir, f"bar_{metric}.png")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Bar chart: {out_path}")

    # --- Radar/Spider chart (overview) ---
    from math import pi
    radar_metrics = [m for m in metrics if m in METRICS_MAP and m not in ("best_epoch", "nlos_pct")]
    if len(radar_metrics) >= 3:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = [n / float(len(radar_metrics)) * 2 * pi for n in range(len(radar_metrics))]
        angles += angles[:1]

        for ds_name, data in datasets.items():
            vals = [extract_metric(data, METRICS_MAP[m][1]) for m in radar_metrics]
            # Normalize to [0, 1] for radar
            max_vals = [max([extract_metric(d, METRICS_MAP[m][1]) for d in datasets.values()]) for m in radar_metrics]
            norm_vals = [v / max(max_v, 0.001) for v, max_v in zip(vals, max_vals)]
            norm_vals += norm_vals[:1]
            ax.plot(angles, norm_vals, "o-", linewidth=2, label=ds_name)
            ax.fill(angles, norm_vals, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRICS_MAP[m][0] for m in radar_metrics], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.set_title("Normalized Metric Radar")

        radar_path = os.path.join(output_dir, "radar_comparison.png")
        plt.tight_layout()
        fig.savefig(radar_path, dpi=150)
        plt.close()
        print(f"  Radar chart: {radar_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Dataset NLOS GAT Comparison")
    parser.add_argument("--results_root", type=str, required=True, help="Root dir with subdirs containing predictions.json")
    parser.add_argument("--metrics", type=str, default="accuracy,f1,plos_gap,sigma_gap",
                        help="Comma-separated metrics: accuracy,precision,recall,f1,plos_gap,sigma_gap,mu_nlos,auc,nlos_pct")
    parser.add_argument("--all", action="store_true", help="Use all available metrics")
    parser.add_argument("--output", type=str, default=None, help="Output path (.md or directory)")
    args = parser.parse_args()

    if args.all:
        metrics = list(METRICS_MAP.keys())
    else:
        metrics = [m.strip() for m in args.metrics.split(",")]

    print(f"Metrics: {metrics}")
    datasets = load_all_results(args.results_root)
    print(f"Datasets found: {list(datasets.keys())}")

    if not datasets:
        print("ERROR: No datasets with predictions.json found!")
        sys.exit(1)

    output = args.output or args.results_root
    os.makedirs(output, exist_ok=True)

    # Generate table
    table_path = os.path.join(output, "comparison_table.md") if os.path.isdir(output) else output
    generate_comparison_table(datasets, metrics, table_path)

    # Generate bar charts
    img_dir = output if os.path.isdir(output) else os.path.dirname(output)
    generate_comparison_bar_chart(datasets, metrics, img_dir)

    print(f"\nDone! Results in {output}")

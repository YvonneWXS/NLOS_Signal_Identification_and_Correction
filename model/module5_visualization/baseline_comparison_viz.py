# module5_visualization/baseline_comparison_viz.py — Generate baseline comparison bar charts
import sys, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here / "common"))


def plot_baseline_comparison(results_dir="results/baseline", output_dir="results/baseline"):
    """Generate bar chart comparing CEP50 across methods for each dataset."""
    results_path = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(results_path):
        print(f"No results at {results_path}")
        return
    
    with open(results_path) as f:
        all_results = json.load(f)
    
    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    os.makedirs(output_dir, exist_ok=True)
    
    # Per-dataset bar charts
    for ds in datasets:
        methods = all_results[ds]
        names = list(methods.keys())
        cep50s = [methods[n].get("cep50", 0) for n in names]
        
        # Sort by CEP50
        order = np.argsort(cep50s)
        names = [names[i] for i in order]
        cep50s = [cep50s[i] for i in order]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(names))]
        bars = ax.bar(range(len(names)), cep50s, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("CEP50 (km)")
        ax.set_title(f"Baseline Comparison — {ds}")
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, cep50s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"baseline_{ds}.png")
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"Saved {filepath}")
    
    # Summary bar chart (all datasets side by side)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get union of methods across datasets
    all_methods = sorted(set().union(*[set(r.keys()) for r in all_results.values()]))
    # Only show top methods
    avg_cep = {}
    for m in all_methods:
        vals = [all_results[ds].get(m, {}).get("cep50", np.nan) for ds in datasets]
        avg_cep[m] = np.nanmean(vals) if vals else np.inf
    top_methods = sorted(avg_cep, key=avg_cep.get)[:8]
    
    x = np.arange(len(datasets))
    width = 0.12
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_methods)))
    
    for i, method in enumerate(top_methods):
        vals = [all_results[ds].get(method, {}).get("cep50", np.nan) for ds in datasets]
        ax.bar(x + i * width, vals, width, label=method, color=colors[i])
    
    ax.set_xticks(x + width * len(top_methods) / 2)
    ax.set_xticklabels([d.replace("_", " ")[:12] for d in datasets])
    ax.set_ylabel("CEP50 (km)")
    ax.set_title("Baseline CEP50 Comparison — All Datasets")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "baseline_summary.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved {filepath}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/baseline")
    parser.add_argument("--output", type=str, default="results/baseline")
    args = parser.parse_args()
    plot_baseline_comparison(args.input, args.output)

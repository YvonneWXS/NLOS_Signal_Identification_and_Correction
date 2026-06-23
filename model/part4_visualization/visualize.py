# -*- coding: utf-8 -*-
"""
visualize.py -- NLOS GAT Visualization Module
==============================================
Generates 8 visualization plots for Module 1 results.
Usage: python visualize.py --results_dir <path> --output_dir <path> [--dataset <name>]
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# --- Plot 1: 2D Trajectory with NLOS overlay ---
def plot_trajectory_2d(predictions, output_path, dataset_name=""):
    """Plot ground-truth trajectory with NLOS predictions color-coded."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # If no trajectory data, create a simple epoch-index plot
    p_los = np.array(predictions["p_los"])
    labels = np.array(predictions["labels"])
    elevation = np.array(predictions.get("elevation", np.zeros_like(p_los)))

    # Simulate trajectory as epoch index (placeholder for actual lat/lon)
    epochs = np.arange(len(p_los))
    los_mask = labels == 0
    nlos_mask = labels == 1

    ax.scatter(epochs[los_mask], p_los[los_mask], c="green", s=2, alpha=0.5, label="LOS (true)")
    ax.scatter(epochs[nlos_mask], p_los[nlos_mask], c="red", s=8, alpha=0.7, label="NLOS (true)")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("p_los (LOS probability)")
    ax.set_title(f"NLOS Detection Overview [{dataset_name}]")
    ax.legend(markerscale=3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [1/8] trajectory_2d -> {output_path}")

# --- Plot 2: p_los Distribution ---
def plot_plos_distribution(predictions, output_path, dataset_name=""):
    """Histogram of p_los for LOS vs NLOS."""
    p_los = np.array(predictions["p_los"])
    labels = np.array(predictions["labels"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(p_los[labels == 0], bins=40, alpha=0.6, color="green", label="LOS", density=True)
    ax1.hist(p_los[labels == 1], bins=40, alpha=0.6, color="red", label="NLOS", density=True)
    ax1.axvline(x=0.5, color="gray", linestyle="--")
    ax1.set_xlabel("p_los")
    ax1.set_ylabel("Density")
    ax1.set_title("p_los Distribution by True Label")
    ax1.legend()

    # Cumulative
    ax2.hist(p_los[labels == 0], bins=40, alpha=0.6, color="green", cumulative=True, density=True, histtype="step", label="LOS")
    ax2.hist(p_los[labels == 1], bins=40, alpha=0.6, color="red", cumulative=True, density=True, histtype="step", label="NLOS")
    ax2.axvline(x=0.5, color="gray", linestyle="--")
    ax2.set_xlabel("p_los")
    ax2.set_ylabel("Cumulative Density")
    ax2.set_title("Cumulative p_los Distribution")
    ax2.legend()

    fig.suptitle(f"p_los Distribution Analysis [{dataset_name}]")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [2/8] plos_distribution -> {output_path}")

# --- Plot 3: Confusion Matrix ---
def plot_confusion_matrix(predictions, output_path, dataset_name=""):
    """Confusion matrix visualization."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    p_los = np.array(predictions["p_los"])
    labels = np.array(predictions["labels"])
    pred = (p_los < 0.5).astype(int)
    cm = confusion_matrix(labels, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["LOS", "NLOS"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix [{dataset_name}]")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [3/8] confusion_matrix -> {output_path}")

# --- Plot 4: sigma_nlos Distribution ---
def plot_sigma_distribution(predictions, output_path, dataset_name=""):
    """sigma_nlos distribution for LOS vs NLOS."""
    sn = np.array(predictions.get("sigma_nlos", [0]))
    labels = np.array(predictions["labels"])
    if sn.max() == 0:
        print("  [4/8] sigma_distribution SKIPPED (no sigma data)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sn[labels == 0], bins=30, alpha=0.6, color="green", label="LOS", density=True)
    ax.hist(sn[labels == 1], bins=30, alpha=0.6, color="red", label="NLOS", density=True)
    ax.axvline(x=sn[labels == 1].mean() if labels.sum() > 0 else 0, color="darkred", linestyle="--", label=f"NLOS mean={sn[labels==1].mean():.3f}")
    ax.set_xlabel("sigma_nlos (km)")
    ax.set_ylabel("Density")
    ax.set_title(f"Uncertainty (sigma_nlos) Distribution [{dataset_name}]")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [4/8] sigma_distribution -> {output_path}")

# --- Plot 5: Elevation vs p_los ---
def plot_elevation_vs_plos(predictions, output_path, dataset_name=""):
    """Scatter of elevation vs p_los, color-coded by true label."""
    p_los = np.array(predictions["p_los"])
    labels = np.array(predictions["labels"])
    elevation = np.array(predictions.get("elevation", np.zeros_like(p_los)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(elevation[labels == 0], p_los[labels == 0], c="green", s=3, alpha=0.4, label="LOS")
    ax.scatter(elevation[labels == 1], p_los[labels == 1], c="red", s=5, alpha=0.6, label="NLOS")
    ax.axhline(y=0.5, color="gray", linestyle="--")
    ax.set_xlabel("Elevation (deg)")
    ax.set_ylabel("p_los")
    ax.set_title(f"Elevation vs p_los [{dataset_name}]")
    ax.legend(markerscale=2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [5/8] elevation_vs_plos -> {output_path}")

# --- Plot 6: mu_nlos Distribution ---
def plot_mu_distribution(predictions, output_path, dataset_name=""):
    """mu_nlos distribution for LOS vs NLOS."""
    mu = np.array(predictions.get("mu_nlos", [0]))
    labels = np.array(predictions["labels"])
    if mu.max() == 0:
        print("  [6/8] mu_distribution SKIPPED")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mu[labels == 0], bins=30, alpha=0.6, color="green", label="LOS", density=True)
    ax.hist(mu[labels == 1], bins=30, alpha=0.6, color="red", label="NLOS", density=True)
    ax.axvline(x=mu[labels == 1].mean() if labels.sum() > 0 else 0, color="darkred", linestyle="--")
    ax.set_xlabel("mu_nlos (km)")
    ax.set_ylabel("Density")
    ax.set_title(f"mu_NLOS Distribution [{dataset_name}]")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [6/8] mu_distribution -> {output_path}")

# --- Plot 7: Error Analysis (FN/FP characteristics) ---
def plot_error_analysis(predictions, output_path, dataset_name=""):
    """Analyze FN and FP samples by elevation and C/N0."""
    p_los = np.array(predictions["p_los"])
    labels = np.array(predictions["labels"])
    elevation = np.array(predictions.get("elevation", np.zeros_like(p_los)))
    cno = np.array(predictions.get("cno", np.zeros_like(p_los)))
    pred = (p_los < 0.5).astype(int)
    fn = (pred == 0) & (labels == 1)
    fp = (pred == 1) & (labels == 0)
    correct = pred == labels

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # FN elevation
    axes[0, 0].hist(elevation[correct], bins=20, alpha=0.5, color="gray", label="Correct", density=True)
    axes[0, 0].hist(elevation[fn], bins=20, alpha=0.7, color="red", label=f"FN (n={fn.sum()})", density=True)
    axes[0, 0].set_xlabel("Elevation (deg)"); axes[0, 0].set_title("FN: Elevation Distribution")
    axes[0, 0].legend()
    # FP elevation
    axes[0, 1].hist(elevation[correct], bins=20, alpha=0.5, color="gray", label="Correct", density=True)
    axes[0, 1].hist(elevation[fp], bins=20, alpha=0.7, color="orange", label=f"FP (n={fp.sum()})", density=True)
    axes[0, 1].set_xlabel("Elevation (deg)"); axes[0, 1].set_title("FP: Elevation Distribution")
    axes[0, 1].legend()
    # FN C/N0
    axes[1, 0].hist(cno[correct], bins=20, alpha=0.5, color="gray", label="Correct", density=True)
    axes[1, 0].hist(cno[fn], bins=20, alpha=0.7, color="red", label=f"FN", density=True)
    axes[1, 0].set_xlabel("C/N0 (dBHz)"); axes[1, 0].set_title("FN: C/N0 Distribution")
    axes[1, 0].legend()
    # FP C/N0
    axes[1, 1].hist(cno[correct], bins=20, alpha=0.5, color="gray", label="Correct", density=True)
    axes[1, 1].hist(cno[fp], bins=20, alpha=0.7, color="orange", label=f"FP", density=True)
    axes[1, 1].set_xlabel("C/N0 (dBHz)"); axes[1, 1].set_title("FP: C/N0 Distribution")
    axes[1, 1].legend()

    fig.suptitle(f"Error Case Analysis [{dataset_name}]")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [7/8] error_analysis -> {output_path}")

# --- Plot 8: Training Curves ---
def plot_training_curves(history, output_path, dataset_name=""):
    """Plot training and validation loss/accuracy/F1 curves."""
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(epochs, history.get("train_loss", []), "b-", label="Train Loss", linewidth=1)
    axes[0, 0].plot(epochs, history.get("val_loss", []), "r-", label="Val Loss", linewidth=1)
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves"); axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history.get("val_acc", []), "g-", label="Val Accuracy", linewidth=1)
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Validation Accuracy"); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history.get("val_f1", []), "purple", label="Val F1", linewidth=1)
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_title("Validation F1"); axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Smooth loss
    if len(epochs) > 5:
        from scipy.ndimage import uniform_filter1d
        tls = np.array(history.get("train_loss", []))
        vls = np.array(history.get("val_loss", []))
        tls_s = uniform_filter1d(tls[~np.isnan(tls)], size=min(5, len(tls)//2)) if len(tls) > 0 else tls
        ax_s = axes[1, 1]
        ax_s.plot(epochs[:len(tls_s)], tls_s, "b-", alpha=0.6, linewidth=1.5, label="Train (smooth)")
        ax_s.set_xlabel("Epoch"); ax_s.set_ylabel("Loss")
        ax_s.set_title("Smoothed Loss"); ax_s.legend()
        ax_s.grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves [{dataset_name}]")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [8/8] training_curves -> {output_path}")


def generate_all_visualizations(results_dir, output_dir, dataset_name=""):
    """Generate all 8 visualization plots from predictions.json."""
    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(results_dir, "predictions.json")
    if not os.path.exists(pred_path):
        print(f"ERROR: predictions.json not found at {pred_path}")
        return False

    with open(pred_path) as f:
        predictions = json.load(f)

    history = predictions.get("history", {})
    ds = dataset_name or os.path.basename(results_dir)

    plot_trajectory_2d(predictions, os.path.join(output_dir, "01_trajectory_2d.png"), ds)
    plot_plos_distribution(predictions, os.path.join(output_dir, "02_plos_distribution.png"), ds)
    plot_confusion_matrix(predictions, os.path.join(output_dir, "03_confusion_matrix.png"), ds)
    plot_sigma_distribution(predictions, os.path.join(output_dir, "04_sigma_distribution.png"), ds)
    plot_elevation_vs_plos(predictions, os.path.join(output_dir, "05_elevation_vs_plos.png"), ds)
    plot_mu_distribution(predictions, os.path.join(output_dir, "06_mu_distribution.png"), ds)
    plot_error_analysis(predictions, os.path.join(output_dir, "07_error_analysis.png"), ds)
    plot_training_curves(history, os.path.join(output_dir, "08_training_curves.png"), ds)

    print(f"\nAll 8 visualizations saved to {output_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLOS GAT Visualization")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results dir with predictions.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir for images")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name for titles")
    args = parser.parse_args()

    output = args.output_dir or os.path.join(args.results_dir, "visualizations")
    generate_all_visualizations(args.results_dir, output, args.dataset)

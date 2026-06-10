# -*- coding: utf-8 -*-
"""
analyze_mog.py -- MoG NLOS-GAT Model Analysis Script
======================================================
Loads a trained best_model.pth, runs full dataset inference, and produces
comprehensive metrics + error case analysis.

Usage:
    python analyze_mog.py --exp exp_008 --dataset berlin1_potsdamer_platz
    python analyze_mog.py --exp exp_009 --dataset berlin2_gendarmenmarkt
    python analyze_mog.py --exp exp_010 --dataset frankfurt1_maintower
    python analyze_mog.py --exp exp_011 --dataset frankfurt2_westendtower
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config, Config
from GAT_V2025 import NLOSGAT, GNSDataset, batch_collate_fn, _extract_elevation
from Data_read import load_and_process_dataset
from NodeFeature_Generate import FEATURE_DIM


def _safe_float(val):
    """Convert numpy/torch scalar to Python float safely."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def compute_classification_metrics(labels, preds):
    """Compute binary classification metrics from numpy arrays."""
    labels = np.asarray(labels, dtype=np.int32)
    preds = np.asarray(preds, dtype=np.int32)

    total = len(labels)
    if total == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0
        }

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": _safe_float(accuracy),
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "f1": _safe_float(f1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
    }


def compute_distribution_stats(values, name=""):
    """Compute distribution statistics for a numpy array."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v) & ~np.isinf(v)]
    if len(v) == 0:
        return {"count": 0}
    return {
        "count": int(len(v)),
        "mean": _safe_float(np.mean(v)),
        "std": _safe_float(np.std(v)),
        "min": _safe_float(np.min(v)),
        "p5": _safe_float(np.percentile(v, 5)),
        "p25": _safe_float(np.percentile(v, 25)),
        "median": _safe_float(np.percentile(v, 50)),
        "p75": _safe_float(np.percentile(v, 75)),
        "p95": _safe_float(np.percentile(v, 95)),
        "max": _safe_float(np.max(v)),
    }


def top_k_error_indices(errors, k=10):
    """Return indices of top-K largest values in array."""
    if len(errors) == 0:
        return []
    order = np.argsort(errors)[::-1]
    return order[:k].tolist()


def analyze_experiment(exp_name, dataset_name, result_dir, output_dir, device_str="cpu",
                       batch_size=64):
    """
    Load model, run inference on the full dataset, produce comprehensive analysis.

    Returns a dict with all metrics, also saves to JSON.
    """
    # ------------------------------------------------------------------
    # 1. Locate checkpoint
    # ------------------------------------------------------------------
    exp_dir = os.path.join(result_dir, exp_name)
    checkpoint_path = os.path.join(exp_dir, "best_model.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[{exp_name}] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_epoch = checkpoint.get("epoch", "?")
    ckpt_val_loss = checkpoint.get("val_loss", None)
    print(f"  Checkpoint epoch: {ckpt_epoch}, stored val_loss/F1: {ckpt_val_loss}")

    # ------------------------------------------------------------------
    # 2. Build config & model
    # ------------------------------------------------------------------
    config = get_config()
    config.DATASETS = [dataset_name]
    config.BATCH_SIZE = batch_size
    config.USE_BLOCK_DIAGONAL = True

    device = torch.device(device_str)
    model = NLOSGAT(
        in_features=config.IN_FEATURES,
        hidden_features=config.HIDDEN_FEATURES,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    state_dict = checkpoint["model_state_dict"]
    # Some legacy checkpoints may store keys without the module prefix; normalize if needed
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {param_count:,} params")

    # ------------------------------------------------------------------
    # 3. Load dataset
    # ------------------------------------------------------------------
    print(f"  Loading dataset: {dataset_name}")
    epochs = load_and_process_dataset(dataset_name, config)
    if not epochs:
        raise RuntimeError(f"No epochs loaded for dataset: {dataset_name}")
    print(f"  Loaded {len(epochs)} epochs")

    dataset = GNSDataset(epochs, config)
    loader = DataLoader(
        dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=batch_collate_fn, pin_memory=False,
        drop_last=False,
    )
    print(f"  Dataset size: {len(dataset)} graph samples, {len(loader)} batches")

    # ------------------------------------------------------------------
    # 4. Full-dataset inference
    # ------------------------------------------------------------------
    # Per-sample storage (flattened across all nodes in all batches)
    all_p_los = []
    all_mu_nlos = []
    all_log_sigma_los = []
    all_log_sigma_nlos = []
    all_labels = []          # ground-truth NLOS label (0=LOS, 1=NLOS)
    all_elevations = []      # elevation in degrees
    all_cno = []             # CNO in dBHz
    all_pr_errors = []       # pseudorange error in km (already de-meaned)

    # Track per-batch counts for sanity checks
    total_nodes_inferred = 0

    for batch_idx, batch in enumerate(loader):
        node_features, edge_index, edge_attr, pseudorange_errors, nlos_labels = batch

        if node_features.size(0) == 0:
            continue
        if node_features.dim() == 3 and node_features.size(0) == 1:
            node_features = node_features.squeeze(0)
        if edge_index.dim() == 3 and edge_index.size(0) == 1:
            edge_index = edge_index.squeeze(0)

        node_features = node_features.to(device)
        edge_index = edge_index.to(device)

        if edge_index.size(1) == 0:
            N = node_features.size(0)
            edge_index = torch.tensor(
                [[i for i in range(N)], [i for i in range(N)]],
                device=device, dtype=torch.long,
            )

        with torch.no_grad():
            p_los, mu_nlos, log_sigma_los, log_sigma_nlos = model(
                node_features, edge_index
            )

        # Convert to numpy
        p_los_np = p_los.squeeze().cpu().numpy()
        mu_nlos_np = mu_nlos.squeeze().cpu().numpy()
        log_sigma_los_np = log_sigma_los.squeeze().cpu().numpy()
        log_sigma_nlos_np = log_sigma_nlos.squeeze().cpu().numpy()
        labels_np = nlos_labels.squeeze().cpu().numpy()
        pr_err_np = pseudorange_errors.squeeze().cpu().numpy()

        # Extract raw features from node_features (dim 0=elev/90, dim 2=cno/60, dim 5=prErr/100)
        node_np = node_features.cpu().numpy()
        elev_np = node_np[:, 0] * 90.0
        cno_np = node_np[:, 2] * 60.0

        # Flatten scalars to lists
        def _to_list(arr):
            if arr.ndim == 0:
                return [float(arr)]
            return arr.flatten().tolist()

        batch_size_actual = node_np.shape[0]
        all_p_los.extend(_to_list(p_los_np))
        all_mu_nlos.extend(_to_list(mu_nlos_np))
        all_log_sigma_los.extend(_to_list(log_sigma_los_np))
        all_log_sigma_nlos.extend(_to_list(log_sigma_nlos_np))
        all_labels.extend(_to_list(labels_np))
        all_elevations.extend(_to_list(elev_np))
        all_cno.extend(_to_list(cno_np))
        all_pr_errors.extend(_to_list(pr_err_np))
        total_nodes_inferred += batch_size_actual

    print(f"  Total nodes inferred: {total_nodes_inferred}")
    assert len(all_p_los) == total_nodes_inferred, "Node count mismatch!"

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    p_los_arr = np.array(all_p_los, dtype=np.float64)
    mu_nlos_arr = np.array(all_mu_nlos, dtype=np.float64)
    log_sigma_los_arr = np.array(all_log_sigma_los, dtype=np.float64)
    log_sigma_nlos_arr = np.array(all_log_sigma_nlos, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int32)
    elev_arr = np.array(all_elevations, dtype=np.float64)
    cno_arr = np.array(all_cno, dtype=np.float64)
    pr_err_arr = np.array(all_pr_errors, dtype=np.float64)

    sigma_los_arr = np.exp(log_sigma_los_arr)
    sigma_nlos_arr = np.exp(log_sigma_nlos_arr)

    # Binary predictions from p_los threshold
    preds_arr = (p_los_arr < 0.5).astype(np.int32)  # NLOS=1 if p_los < 0.5
    cls_metrics = compute_classification_metrics(labels_arr, preds_arr)

    # Per-class masks
    los_mask = labels_arr == 0
    nlos_mask = labels_arr == 1

    # ------------------------------------------------------------------
    # 5a. p_los distribution
    # ------------------------------------------------------------------
    p_los_dist = {
        "overall": compute_distribution_stats(p_los_arr),
        "LOS_samples": compute_distribution_stats(p_los_arr[los_mask]),
        "NLOS_samples": compute_distribution_stats(p_los_arr[nlos_mask]),
        "gap_LOS_minus_NLOS": _safe_float(
            np.mean(p_los_arr[los_mask]) - np.mean(p_los_arr[nlos_mask])
        ) if los_mask.any() and nlos_mask.any() else None,
    }

    # ------------------------------------------------------------------
    # 5b. mu_nlos distribution
    # ------------------------------------------------------------------
    mu_nlos_dist = {
        "overall": compute_distribution_stats(mu_nlos_arr),
        "LOS_samples": compute_distribution_stats(mu_nlos_arr[los_mask]),
        "NLOS_samples": compute_distribution_stats(mu_nlos_arr[nlos_mask]),
    }

    # ------------------------------------------------------------------
    # 5c. sigma stats
    # ------------------------------------------------------------------
    sigma_stats = {}
    for name, arr in [("sigma_los", sigma_los_arr), ("sigma_nlos", sigma_nlos_arr)]:
        sigma_stats[name] = {
            "overall": compute_distribution_stats(arr),
            "LOS_samples": compute_distribution_stats(arr[los_mask]),
            "NLOS_samples": compute_distribution_stats(arr[nlos_mask]),
        }

    # Log-sigma stats
    log_sigma_stats = {}
    for name, arr in [("log_sigma_los", log_sigma_los_arr), ("log_sigma_nlos", log_sigma_nlos_arr)]:
        log_sigma_stats[name] = {
            "overall": compute_distribution_stats(arr),
            "LOS_samples": compute_distribution_stats(arr[los_mask]),
            "NLOS_samples": compute_distribution_stats(arr[nlos_mask]),
        }

    # ------------------------------------------------------------------
    # 5d. Elevation-binned p_los analysis
    # ------------------------------------------------------------------
    elev_bins = [(-90, -15), (-15, 0), (0, 15), (15, 30), (30, 45), (45, 60), (60, 90)]
    elev_bin_analysis = []
    for low, high in elev_bins:
        in_bin = (elev_arr >= low) & (elev_arr < high)
        if in_bin.sum() == 0:
            continue
        pl = p_los_arr[in_bin]
        lb = labels_arr[in_bin]
        bin_preds = (pl < 0.5).astype(np.int32)
        bin_acc = (bin_preds == lb).mean()
        elev_bin_analysis.append({
            "elev_range": f"[{low}, {high})",
            "count": int(in_bin.sum()),
            "p_los_mean": _safe_float(pl.mean()),
            "p_los_std": _safe_float(pl.std()),
            "accuracy": _safe_float(bin_acc),
            "n_nlos": int((lb == 1).sum()),
            "n_los": int((lb == 0).sum()),
        })

    # ------------------------------------------------------------------
    # 5e. CNO-binned analysis
    # ------------------------------------------------------------------
    cno_bins = [(0, 20), (20, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 100)]
    cno_bin_analysis = []
    for low, high in cno_bins:
        in_bin = (cno_arr >= low) & (cno_arr < high)
        if in_bin.sum() == 0:
            continue
        pl = p_los_arr[in_bin]
        lb = labels_arr[in_bin]
        bin_preds = (pl < 0.5).astype(np.int32)
        bin_acc = (bin_preds == lb).mean()
        cno_bin_analysis.append({
            "cno_range": f"[{low}, {high})",
            "count": int(in_bin.sum()),
            "p_los_mean": _safe_float(pl.mean()),
            "accuracy": _safe_float(bin_acc),
            "n_nlos": int((lb == 1).sum()),
            "p_los_nlos_avg": _safe_float(pl[lb == 1].mean()) if (lb == 1).any() else None,
        })

    # ------------------------------------------------------------------
    # 5f. Error case analysis (FN / FP top-10)
    # ------------------------------------------------------------------
    # False Negatives: predicted LOS (p_los >= 0.5) but true NLOS (label=1)
    fn_indices = np.where((preds_arr == 0) & (labels_arr == 1))[0]
    fn_p_los_sorted = fn_indices[np.argsort(-p_los_arr[fn_indices])]  # closest to 0.5 first
    fn_entries = []
    for idx in fn_p_los_sorted[:10]:
        fn_entries.append({
            "index": int(idx),
            "p_los": _safe_float(p_los_arr[idx]),
            "elevation_deg": _safe_float(elev_arr[idx]),
            "cno_dBHz": _safe_float(cno_arr[idx]),
            "pr_error_km": _safe_float(pr_err_arr[idx]),
            "sigma_los": _safe_float(sigma_los_arr[idx]),
            "sigma_nlos": _safe_float(sigma_nlos_arr[idx]),
            "mu_nlos": _safe_float(mu_nlos_arr[idx]),
        })

    # False Positives: predicted NLOS (p_los < 0.5) but true LOS (label=0)
    fp_indices = np.where((preds_arr == 1) & (labels_arr == 0))[0]
    fp_p_los_sorted = fp_indices[np.argsort(p_los_arr[fp_indices])]  # lowest p_los first
    fp_entries = []
    for idx in fp_p_los_sorted[:10]:
        fp_entries.append({
            "index": int(idx),
            "p_los": _safe_float(p_los_arr[idx]),
            "elevation_deg": _safe_float(elev_arr[idx]),
            "cno_dBHz": _safe_float(cno_arr[idx]),
            "pr_error_km": _safe_float(pr_err_arr[idx]),
            "sigma_los": _safe_float(sigma_los_arr[idx]),
            "sigma_nlos": _safe_float(sigma_nlos_arr[idx]),
            "mu_nlos": _safe_float(mu_nlos_arr[idx]),
        })

    # ------------------------------------------------------------------
    # 6. Assemble results dict
    # ------------------------------------------------------------------
    results = {
        "experiment": exp_name,
        "dataset": dataset_name,
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_val_loss_stored": _safe_float(ckpt_val_loss) if ckpt_val_loss is not None else None,
        "total_nodes": total_nodes_inferred,
        "total_los": int(los_mask.sum()),
        "total_nlos": int(nlos_mask.sum()),
        "classification": cls_metrics,
        "p_los_distribution": p_los_dist,
        "mu_nlos_distribution": mu_nlos_dist,
        "sigma_statistics": sigma_stats,
        "log_sigma_statistics": log_sigma_stats,
        "elevation_bin_analysis": elev_bin_analysis,
        "cno_bin_analysis": cno_bin_analysis,
        "false_negatives_top10": fn_entries,
        "false_positives_top10": fp_entries,
        "summary": {
            "Accuracy": cls_metrics["accuracy"],
            "Precision": cls_metrics["precision"],
            "Recall": cls_metrics["recall"],
            "F1": cls_metrics["f1"],
            "p_los_LOS_avg": p_los_dist["LOS_samples"].get("mean"),
            "p_los_NLOS_avg": p_los_dist["NLOS_samples"].get("mean"),
            "p_los_gap": p_los_dist["gap_LOS_minus_NLOS"],
            "mu_nlos_NLOS_mean": mu_nlos_dist["NLOS_samples"].get("mean"),
            "mu_nlos_LOS_mean": mu_nlos_dist["LOS_samples"].get("mean"),
            "sigma_los_mean": sigma_stats["sigma_los"]["overall"].get("mean"),
            "sigma_nlos_mean": sigma_stats["sigma_nlos"]["overall"].get("mean"),
            "sigma_nlos_NLOS_mean": sigma_stats["sigma_nlos"]["NLOS_samples"].get("mean"),
            "sigma_nlos_LOS_mean": sigma_stats["sigma_nlos"]["LOS_samples"].get("mean"),
            "sigma_los_NLOS_mean": sigma_stats["sigma_los"]["NLOS_samples"].get("mean"),
            "sigma_los_LOS_mean": sigma_stats["sigma_los"]["LOS_samples"].get("mean"),
        },
    }

    # ------------------------------------------------------------------
    # 7. Save JSON
    # ------------------------------------------------------------------
    output_path = os.path.join(output_dir, f"analysis_{dataset_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MoG NLOS-GAT Model Analysis"
    )
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment name (e.g., exp_008)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., berlin1_potsdamer_platz)")
    parser.add_argument("--result-dir", type=str,
                        default=r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result",
                        help="Root result directory (default: <project_root>/model/part1_GAT/result)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for JSON (default: same as result_dir)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, cuda, or auto")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference")

    args = parser.parse_args()

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    if args.output_dir is None:
        output_dir = args.result_dir
    else:
        output_dir = args.output_dir

    print(f"=" * 70)
    print(f"MoG Analysis: {args.exp} / {args.dataset}")
    print(f"Device: {device_str}")
    print(f"Output dir: {output_dir}")
    print(f"=" * 70)

    results = analyze_experiment(
        exp_name=args.exp,
        dataset_name=args.dataset,
        result_dir=args.result_dir,
        output_dir=output_dir,
        device_str=device_str,
        batch_size=args.batch_size,
    )

    # Print compact summary
    s = results["summary"]
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {args.exp} / {args.dataset}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:   {s['Accuracy']:.4f}")
    print(f"  Precision:  {s['Precision']:.4f}")
    print(f"  Recall:     {s['Recall']:.4f}")
    print(f"  F1:         {s['F1']:.4f}")
    print(f"  p_los LOS  avg: {s['p_los_LOS_avg']:.4f}")
    print(f"  p_los NLOS avg: {s['p_los_NLOS_avg']:.4f}")
    print(f"  p_los gap:      {s['p_los_gap']:.4f}")
    print(f"  mu_nlos LOS:    {s['mu_nlos_LOS_mean']:.4f}")
    print(f"  mu_nlos NLOS:   {s['mu_nlos_NLOS_mean']:.4f}")
    print(f"  sigma_los:      {s['sigma_los_mean']:.4f}")
    print(f"  sigma_nlos:     {s['sigma_nlos_mean']:.4f}")
    print(f"  sigma_nlos (LOS):  {s['sigma_nlos_LOS_mean']:.4f}")
    print(f"  sigma_nlos (NLOS): {s['sigma_nlos_NLOS_mean']:.4f}")
    print(f"  sigma_los (LOS):   {s['sigma_los_LOS_mean']:.4f}")
    print(f"  sigma_los (NLOS):  {s['sigma_los_NLOS_mean']:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

"""
模型分析脚本 — 加载 exp_002 最佳模型，进行验证集错误案例分析
"""
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, get_config
from Data_read import load_and_process_dataset
from NodeFeature_Generate import extract_node_features, extract_labels, extract_pseudorange_errors, FEATURE_DIM
from Depth_Adj_Generate import build_azimuth_graph
from GAT_V2025 import NLOSGAT

# ─── 1. Load model ───
config = get_config()
device = config.get_device()
print(f"Device: {device}")

model = NLOSGAT(
    in_features=config.IN_FEATURES,
    hidden_features=config.HIDDEN_FEATURES,
    num_heads=config.NUM_HEADS,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
).to(device)

checkpoint = torch.load(
    os.path.join(config.RESULT_DIR, "exp_002", "best_model.pth"),
    map_location=device,
    weights_only=False,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded model from epoch {checkpoint['epoch']+1}, val_loss={checkpoint['val_loss']:.4f}")

# ─── 2. Load data with fixed train/val split ───
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

all_epochs = []
for ds_name in config.DATASETS:
    epochs = load_and_process_dataset(ds_name, config)
    if epochs:
        all_epochs.extend(epochs)
    print(f"  {ds_name}: {len(epochs)} epochs")

num_total = len(all_epochs)
indices = np.random.permutation(num_total)
split = int(num_total * (1 - config.VALIDATION_SPLIT))
val_indices = indices[split:]
val_epochs_data = [all_epochs[i] for i in val_indices]
print(f"Total: {num_total}, Val: {len(val_epochs_data)}")

# ─── 3. Run inference, collect per-sample data ───
records = []

for epoch in val_epochs_data:
    if len(epoch.observations) == 0:
        continue

    node_features = extract_node_features(epoch)
    edge_index, edge_attr = build_azimuth_graph(epoch, config.AZIMUTH_THRESHOLD)
    pr_errors = extract_pseudorange_errors(epoch)
    nlos_labels = extract_labels(epoch)

    x = torch.tensor(node_features, dtype=torch.float32).to(device)
    ei = torch.tensor(edge_index, dtype=torch.long).to(device)

    with torch.no_grad():
        p_los, log_sigma = model(x, ei)

    p_los_np = p_los.squeeze().cpu().numpy()
    log_sigma_np = log_sigma.squeeze().cpu().numpy()
    nlos_np = nlos_labels.squeeze()

    if p_los_np.ndim == 0:
        p_los_np = np.array([p_los_np.item()])
        log_sigma_np = np.array([log_sigma_np.item()])

    for i, obs in enumerate(epoch.observations):
        records.append({
            'elevation': obs.elevation,
            'azimuth': obs.azimuth,
            'cno': obs.cno,
            'pr_stdev': obs.pr_stdev,
            'pr_mes': obs.pr_mes,
            'pseudorange_error': abs(obs.pseudorange_error),
            'gnss_id': obs.gnss_id,
            'nlos_label': int(nlos_np[i]),
            'p_los': float(p_los_np[i]),
            'p_nlos': float(1.0 - p_los_np[i]),
            'log_sigma': float(log_sigma_np[i]),
            'sigma': float(np.exp(log_sigma_np[i])),
        })

print(f"\nTotal samples analyzed: {len(records)}")

# ─── 4. Compute stats ───
p_los_arr = np.array([r['p_los'] for r in records])
nlos_arr = np.array([r['nlos_label'] for r in records])
sigma_arr = np.array([r['sigma'] for r in records])

los_mask = nlos_arr == 0
nlos_mask = nlos_arr == 1

p_los_los = p_los_arr[los_mask]
p_los_nlos = p_los_arr[nlos_mask]

print(f"\n{'='*60}")
print(f"Final Model Analysis — exp_002 best_model")
print(f"{'='*60}")
print(f"Total samples: {len(records)}  (LOS={los_mask.sum()}, NLOS={nlos_mask.sum()})")
print(f"NLOS ratio: {nlos_mask.sum()/len(records):.3f}")
print(f"\np_LOS distribution:")
print(f"  All:  mean={p_los_arr.mean():.4f}  median={np.median(p_los_arr):.4f}  std={p_los_arr.std():.4f}")
print(f"  LOS:  mean={p_los_los.mean():.4f}  median={np.median(p_los_los):.4f}  std={p_los_los.std():.4f}  n={len(p_los_los)}")
print(f"  NLOS: mean={p_los_nlos.mean():.4f}  median={np.median(p_los_nlos):.4f}  std={p_los_nlos.std():.4f}  n={len(p_los_nlos)}")
print(f"  Gap (LOS_avg - NLOS_avg): {p_los_los.mean() - p_los_nlos.mean():.4f}")

# Classification (p_nlos > 0.5 as NLOS)
pred_nlos = (1.0 - p_los_arr) > 0.5
tp = np.sum(pred_nlos & nlos_mask)
fp = np.sum(pred_nlos & los_mask)
fn = np.sum(~pred_nlos & nlos_mask)
tn = np.sum(~pred_nlos & los_mask)

accuracy = (tp + tn) / len(records)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)

print(f"\nClassification metrics (threshold=0.5):")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

# Sigma stats
print(f"\nUncertainty (sigma) distribution:")
print(f"  mean={sigma_arr.mean():.2f}  median={np.median(sigma_arr):.2f}")
print(f"  min={sigma_arr.min():.2f}  max={sigma_arr.max():.2f}")
print(f"  p25={np.percentile(sigma_arr, 25):.2f}  p75={np.percentile(sigma_arr, 75):.2f}")

# Per-class sigma
sigma_los = sigma_arr[los_mask]
sigma_nlos = sigma_arr[nlos_mask]
print(f"  LOS sigma:  mean={sigma_los.mean():.2f}  median={np.median(sigma_los):.2f}")
print(f"  NLOS sigma: mean={sigma_nlos.mean():.2f}  median={np.median(sigma_nlos):.2f}")

# ─── 5. Error case analysis ───
print(f"\n{'='*60}")
print(f"Error Case Analysis")
print(f"{'='*60}")

# Type A: False Negative — NLOS sample predicted as LOS (p_los > 0.5, but label=NLOS)
fn_records = [r for r in records if r['nlos_label'] == 1 and r['p_los'] > 0.5]
fn_sorted = sorted(fn_records, key=lambda r: r['p_los'], reverse=True)
print(f"\n--- Type A: NLOS predicted as LOS (FN, p_los>0.5) ---")
print(f"  Count: {len(fn_records)} / {nlos_mask.sum()} = {len(fn_records)/nlos_mask.sum()*100:.1f}% of all NLOS")
if fn_sorted:
    print(f"  Top 10 worst cases (highest p_los on NLOS samples):")
    print(f"  {'elev':>6s} {'cno':>5s} {'prErr(km)':>10s} {'gnss':>8s} {'p_los':>7s} {'sigma':>8s}")
    for r in fn_sorted[:10]:
        print(f"  {r['elevation']:6.1f} {r['cno']:5.1f} {r['pseudorange_error']:10.2f} {r['gnss_id']:>8s} {r['p_los']:7.4f} {r['sigma']:8.2f}")

# Type B: False Positive — LOS sample predicted as NLOS (p_los < 0.5, but label=LOS)
fp_records = [r for r in records if r['nlos_label'] == 0 and r['p_los'] < 0.5]
fp_sorted = sorted(fp_records, key=lambda r: r['p_los'])
print(f"\n--- Type B: LOS predicted as NLOS (FP, p_los<0.5) ---")
print(f"  Count: {len(fp_records)} / {los_mask.sum()} = {len(fp_records)/los_mask.sum()*100:.1f}% of all LOS")
if fp_sorted:
    print(f"  Top 10 worst cases (lowest p_los on LOS samples):")
    print(f"  {'elev':>6s} {'cno':>5s} {'prErr(km)':>10s} {'gnss':>8s} {'p_los':>7s} {'sigma':>8s}")
    for r in fp_sorted[:10]:
        print(f"  {r['elevation']:6.1f} {r['cno']:5.1f} {r['pseudorange_error']:10.2f} {r['gnss_id']:>8s} {r['p_los']:7.4f} {r['sigma']:8.2f}")

# ─── 6. Feature analysis of error vs correct ───
print(f"\n{'='*60}")
print(f"Feature Comparison: Correct vs Error")
print(f"{'='*60}")

# NLOS samples: correct (p_los<=0.5) vs error (p_los>0.5)
nlos_correct = [r for r in records if r['nlos_label'] == 1 and r['p_los'] <= 0.5]
nlos_error = [r for r in records if r['nlos_label'] == 1 and r['p_los'] > 0.5]

print(f"\n--- NLOS samples: Correctly classified (p_los<=0.5) vs Missed (p_los>0.5) ---")
print(f"  Correct: {len(nlos_correct)}  Missed: {len(nlos_error)}")
if nlos_correct and nlos_error:
    features_to_compare = ['elevation', 'cno', 'pseudorange_error', 'pr_stdev']
    names = ['Elevation(deg)', 'CNO(dBHz)', '|prError|(km)', 'prStdev(m)']
    print(f"  {'Feature':<18s} {'Correct_mean':>12s} {'Error_mean':>12s} {'Diff':>10s}")
    for feat, name in zip(features_to_compare, names):
        corr_vals = [r[feat] for r in nlos_correct]
        err_vals = [r[feat] for r in nlos_error]
        diff = np.mean(err_vals) - np.mean(corr_vals)
        print(f"  {name:<18s} {np.mean(corr_vals):12.3f} {np.mean(err_vals):12.3f} {diff:+10.3f}")

# LOS samples: correct (p_los>=0.5) vs error (p_los<0.5)
los_correct = [r for r in records if r['nlos_label'] == 0 and r['p_los'] >= 0.5]
los_error = [r for r in records if r['nlos_label'] == 0 and r['p_los'] < 0.5]

print(f"\n--- LOS samples: Correctly classified (p_los>=0.5) vs Missed (p_los<0.5) ---")
print(f"  Correct: {len(los_correct)}  Missed: {len(los_error)}")
if los_correct and los_error:
    print(f"  {'Feature':<18s} {'Correct_mean':>12s} {'Error_mean':>12s} {'Diff':>10s}")
    for feat, name in zip(features_to_compare, names):
        corr_vals = [r[feat] for r in los_correct]
        err_vals = [r[feat] for r in los_error]
        diff = np.mean(err_vals) - np.mean(corr_vals)
        print(f"  {name:<18s} {np.mean(corr_vals):12.3f} {np.mean(err_vals):12.3f} {diff:+10.3f}")

# GNSS constellation breakdown
print(f"\n--- Per-constellation p_los stats ---")
for gnss in ['GPS', 'Glonass', 'Galileo', 'BeiDou']:
    gnss_recs = [r for r in records if r['gnss_id'] == gnss]
    if gnss_recs:
        p_vals = np.array([r['p_los'] for r in gnss_recs])
        n_vals = np.array([r['nlos_label'] for r in gnss_recs])
        los_p = p_vals[n_vals == 0]
        nlos_p = p_vals[n_vals == 1]
        print(f"  {gnss:>8s}: n={len(gnss_recs):4d}  p_los_mean={p_vals.mean():.4f}  "
              f"LOS_avg={los_p.mean():.4f}  NLOS_avg={nlos_p.mean():.4f}  "
              f"gap={(los_p.mean()-nlos_p.mean()):.4f}")

# p_los distribution histogram (text-based)
print(f"\n--- p_los distribution (LOS vs NLOS) ---")
bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
print(f"  {'Bin':>12s} {'LOS':>8s} {'NLOS':>8s} {'LOS%':>8s} {'NLOS%':>8s}")
for lo, hi in bins:
    los_count = np.sum((p_los_los >= lo) & (p_los_los < hi + 1e-9))
    nlos_count = np.sum((p_los_nlos >= lo) & (p_los_nlos < hi + 1e-9))
    print(f"  [{lo:.1f}-{hi:.1f}): {los_count:8d} {nlos_count:8d} {los_count/len(p_los_los)*100:7.1f}% {nlos_count/len(p_los_nlos)*100:7.1f}%")

# p_los vs pseudorange_error correlation
print(f"\n--- p_los vs |pseudorange_error| by true class ---")
for label, name in [(0, 'LOS'), (1, 'NLOS')]:
    label_recs = [r for r in records if r['nlos_label'] == label]
    low_err = [r for r in label_recs if r['pseudorange_error'] < 1.0]
    mid_err = [r for r in label_recs if 1.0 <= r['pseudorange_error'] < 10.0]
    high_err = [r for r in label_recs if r['pseudorange_error'] >= 10.0]
    print(f"  {name}:")
    for err_range, err_recs in [('<1km', low_err), ('1-10km', mid_err), ('>=10km', high_err)]:
        if err_recs:
            p_vals = [r['p_los'] for r in err_recs]
            print(f"    |err| {err_range:>6s}: n={len(err_recs):4d}  p_los_mean={np.mean(p_vals):.4f}  p_los_median={np.median(p_vals):.4f}")

print(f"\nAnalysis complete.")

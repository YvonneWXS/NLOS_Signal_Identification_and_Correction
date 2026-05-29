"""
analyze_experiment.py — 加载指定实验的最佳模型，进行全面分析
用法: python analyze_experiment.py --exp exp_006 [--dataset berlin1_potsdamer_platz]
"""
import os
import sys
import argparse
import pickle
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, get_config
from Data_read import load_and_process_dataset
from NodeFeature_Generate import extract_node_features, extract_labels, extract_pseudorange_errors, FEATURE_DIM
from Depth_Adj_Generate import build_azimuth_graph
from GAT_V2025 import NLOSGAT


def analyze_experiment(exp_name, dataset_name, config):
    """完整分析实验"""
    device = config.get_device()
    exp_dir = os.path.join(config.RESULT_DIR, exp_name)
    best_model_path = os.path.join(exp_dir, 'best_model.pth')

    if not os.path.exists(best_model_path):
        print(f"ERROR: {best_model_path} not found!")
        return None

    # 加载模型
    model = NLOSGAT(
        in_features=config.IN_FEATURES,
        hidden_features=config.HIDDEN_FEATURES,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Device: {device}")
    print(f"Loaded model from {exp_name}/best_model.pth")
    print(f"  epoch={checkpoint.get('epoch', '?')+1 if 'epoch' in checkpoint else '?'}, val_loss={checkpoint.get('val_loss', checkpoint.get('val_loss', '?')):.4f}")

    # 加载数据
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    all_epochs = []
    epochs_data = load_and_process_dataset(dataset_name, config)
    if epochs_data:
        all_epochs.extend(epochs_data)
    print(f"  {dataset_name}: {len(epochs_data)} epochs")

    if not all_epochs:
        print("ERROR: No data loaded!")
        return None

    # Train/val split (same as training)
    num_total = len(all_epochs)
    indices = np.random.permutation(num_total)
    split = int(num_total * (1 - config.VALIDATION_SPLIT))
    val_indices = indices[split:]
    val_epochs_data = [all_epochs[i] for i in val_indices]
    print(f"Total: {num_total}, Val: {len(val_epochs_data)}")

    # 推理
    records = []
    for epoch in val_epochs_data:
        if len(epoch.observations) == 0:
            continue

        node_features = extract_node_features(epoch)
        edge_index, edge_attr = build_azimuth_graph(epoch, config.AZIMUTH_THRESHOLD)
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

    # 基础统计
    p_los_arr = np.array([r['p_los'] for r in records])
    nlos_arr = np.array([r['nlos_label'] for r in records])
    sigma_arr = np.array([r['sigma'] for r in records])

    los_mask = nlos_arr == 0
    nlos_mask = nlos_arr == 1

    p_los_los = p_los_arr[los_mask]
    p_los_nlos = p_los_arr[nlos_mask]

    results = {
        'exp_name': exp_name,
        'dataset': dataset_name,
        'val_loss': checkpoint.get('val_loss', float('nan')),
        'num_samples': len(records),
        'num_los': int(los_mask.sum()),
        'num_nlos': int(nlos_mask.sum()),
        'nlos_ratio': float(nlos_mask.sum() / len(records)),
    }

    print(f"\n{'='*60}")
    print(f"Model Analysis — {exp_name} ({dataset_name})")
    print(f"{'='*60}")
    print(f"Total samples: {len(records)}  (LOS={los_mask.sum()}, NLOS={nlos_mask.sum()})")
    print(f"NLOS ratio: {nlos_mask.sum()/len(records):.3f}")

    # p_los distribution
    print(f"\np_LOS distribution:")
    print(f"  All:  mean={p_los_arr.mean():.4f}  median={np.median(p_los_arr):.4f}  std={p_los_arr.std():.4f}")
    print(f"  LOS:  mean={p_los_los.mean():.4f}  median={np.median(p_los_los):.4f}  std={p_los_los.std():.4f}  n={len(p_los_los)}")
    print(f"  NLOS: mean={p_los_nlos.mean():.4f}  median={np.median(p_los_nlos):.4f}  std={p_los_nlos.std():.4f}  n={len(p_los_nlos)}")
    gap = p_los_los.mean() - p_los_nlos.mean()
    print(f"  Gap (LOS_avg - NLOS_avg): {gap:.4f}")

    results['p_los_all_mean'] = float(p_los_arr.mean())
    results['p_los_los_avg'] = float(p_los_los.mean())
    results['p_los_nlos_avg'] = float(p_los_nlos.mean())
    results['p_los_gap'] = float(gap)

    # 分类指标
    pred_nlos = (1.0 - p_los_arr) > 0.5
    tp = int(np.sum(pred_nlos & nlos_mask))
    fp = int(np.sum(pred_nlos & los_mask))
    fn = int(np.sum(~pred_nlos & nlos_mask))
    tn = int(np.sum(~pred_nlos & los_mask))

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

    results['accuracy'] = float(accuracy)
    results['precision'] = float(precision)
    results['recall'] = float(recall)
    results['f1'] = float(f1)
    results['tp'] = tp
    results['fp'] = fp
    results['fn'] = fn
    results['tn'] = tn

    # Sigma stats
    sigma_los = sigma_arr[los_mask]
    sigma_nlos = sigma_arr[nlos_mask]
    print(f"\nUncertainty (sigma) distribution:")
    print(f"  mean={sigma_arr.mean():.2f}  median={np.median(sigma_arr):.2f}")
    print(f"  LOS sigma:  mean={sigma_los.mean():.2f}  median={np.median(sigma_los):.2f}")
    print(f"  NLOS sigma: mean={sigma_nlos.mean():.2f}  median={np.median(sigma_nlos):.2f}")

    results['sigma_mean'] = float(sigma_arr.mean())
    results['sigma_los_mean'] = float(sigma_los.mean())
    results['sigma_nlos_mean'] = float(sigma_nlos.mean())

    # ========== 错误案例分析 ==========
    print(f"\n{'='*60}")
    print(f"Error Case Analysis")
    print(f"{'='*60}")

    # Type A: FN (NLOS → predicted as LOS, p_los>0.5)
    fn_records = [r for r in records if r['nlos_label'] == 1 and r['p_los'] > 0.5]
    fn_sorted = sorted(fn_records, key=lambda r: r['p_los'], reverse=True)
    print(f"\n--- Type A: NLOS predicted as LOS (FN, p_los>0.5) ---")
    print(f"  Count: {len(fn_records)} / {nlos_mask.sum()} = {len(fn_records)/max(nlos_mask.sum(),1)*100:.1f}% of all NLOS")
    if fn_sorted:
        print(f"  Top 10 worst cases (highest p_los on NLOS samples):")
        print(f"  {'elev':>6s} {'cno':>5s} {'prErr(km)':>10s} {'gnss':>8s} {'p_los':>7s} {'sigma':>8s}")
        for r in fn_sorted[:10]:
            print(f"  {r['elevation']:6.1f} {r['cno']:5.1f} {r['pseudorange_error']:10.2f} {r['gnss_id']:>8s} {r['p_los']:7.4f} {r['sigma']:8.2f}")

    results['fn_count'] = len(fn_records)
    results['fn_pct'] = float(len(fn_records) / max(nlos_mask.sum(), 1) * 100)
    if fn_sorted:
        results['fn_top10_elev_mean'] = float(np.mean([r['elevation'] for r in fn_sorted[:10]]))
        results['fn_top10_cno_mean'] = float(np.mean([r['cno'] for r in fn_sorted[:10]]))
        results['fn_top10_prerr_mean'] = float(np.mean([r['pseudorange_error'] for r in fn_sorted[:10]]))
        results['fn_top10_prstdev_mean'] = float(np.mean([r['pr_stdev'] for r in fn_sorted[:10]]))

    # Type B: FP (LOS → predicted as NLOS, p_los<0.5)
    fp_records = [r for r in records if r['nlos_label'] == 0 and r['p_los'] < 0.5]
    fp_sorted = sorted(fp_records, key=lambda r: r['p_los'])
    print(f"\n--- Type B: LOS predicted as NLOS (FP, p_los<0.5) ---")
    print(f"  Count: {len(fp_records)} / {los_mask.sum()} = {len(fp_records)/max(los_mask.sum(),1)*100:.1f}% of all LOS")
    if fp_sorted:
        print(f"  Top 10 worst cases (lowest p_los on LOS samples):")
        print(f"  {'elev':>6s} {'cno':>5s} {'prErr(km)':>10s} {'gnss':>8s} {'p_los':>7s} {'sigma':>8s}")
        for r in fp_sorted[:10]:
            print(f"  {r['elevation']:6.1f} {r['cno']:5.1f} {r['pseudorange_error']:10.2f} {r['gnss_id']:>8s} {r['p_los']:7.4f} {r['sigma']:8.2f}")

    results['fp_count'] = len(fp_records)
    results['fp_pct'] = float(len(fp_records) / max(los_mask.sum(), 1) * 100)
    if fp_sorted:
        results['fp_top10_elev_mean'] = float(np.mean([r['elevation'] for r in fp_sorted[:10]]))
        results['fp_top10_cno_mean'] = float(np.mean([r['cno'] for r in fp_sorted[:10]]))
        results['fp_top10_prerr_mean'] = float(np.mean([r['pseudorange_error'] for r in fp_sorted[:10]]))
        results['fp_top10_prstdev_mean'] = float(np.mean([r['pr_stdev'] for r in fp_sorted[:10]]))

    # 错误 vs 正确 特征对比
    print(f"\n{'='*60}")
    print(f"Feature Comparison: Correct vs Error")
    print(f"{'='*60}")

    features_to_compare = ['elevation', 'cno', 'pseudorange_error', 'pr_stdev']
    names = ['Elevation(deg)', 'CNO(dBHz)', '|prError|(km)', 'prStdev(m)']

    # NLOS: correct vs error
    nlos_correct = [r for r in records if r['nlos_label'] == 1 and r['p_los'] <= 0.5]
    nlos_error = [r for r in records if r['nlos_label'] == 1 and r['p_los'] > 0.5]
    print(f"\n--- NLOS samples: Correctly classified vs Missed ---")
    print(f"  Correct: {len(nlos_correct)}  Missed: {len(nlos_error)}")
    if nlos_correct and nlos_error:
        print(f"  {'Feature':<18s} {'Correct_mean':>12s} {'Error_mean':>12s} {'Diff':>10s}")
        for feat, name in zip(features_to_compare, names):
            corr_vals = [r[feat] for r in nlos_correct]
            err_vals = [r[feat] for r in nlos_error]
            diff = np.mean(err_vals) - np.mean(corr_vals)
            print(f"  {name:<18s} {np.mean(corr_vals):12.3f} {np.mean(err_vals):12.3f} {diff:+10.3f}")
            results[f'nlos_{feat}_correct'] = float(np.mean(corr_vals))
            results[f'nlos_{feat}_error'] = float(np.mean(err_vals))

    # LOS: correct vs error
    los_correct = [r for r in records if r['nlos_label'] == 0 and r['p_los'] >= 0.5]
    los_error = [r for r in records if r['nlos_label'] == 0 and r['p_los'] < 0.5]
    print(f"\n--- LOS samples: Correctly classified vs Missed ---")
    print(f"  Correct: {len(los_correct)}  Missed: {len(los_error)}")
    if los_correct and los_error:
        print(f"  {'Feature':<18s} {'Correct_mean':>12s} {'Error_mean':>12s} {'Diff':>10s}")
        for feat, name in zip(features_to_compare, names):
            corr_vals = [r[feat] for r in los_correct]
            err_vals = [r[feat] for r in los_error]
            diff = np.mean(err_vals) - np.mean(corr_vals)
            print(f"  {name:<18s} {np.mean(corr_vals):12.3f} {np.mean(err_vals):12.3f} {diff:+10.3f}")
            results[f'los_{feat}_correct'] = float(np.mean(corr_vals))
            results[f'los_{feat}_error'] = float(np.mean(err_vals))

    # 星座分布
    print(f"\n--- Per-constellation p_los stats ---")
    for gnss in ['GPS', 'Glonass', 'Galileo', 'BeiDou']:
        gnss_recs = [r for r in records if r['gnss_id'] == gnss]
        if gnss_recs:
            p_vals = np.array([r['p_los'] for r in gnss_recs])
            n_vals = np.array([r['nlos_label'] for r in gnss_recs])
            los_p = p_vals[n_vals == 0]
            nlos_p = p_vals[n_vals == 1]
            gnss_gap = los_p.mean() - nlos_p.mean() if len(los_p) > 0 and len(nlos_p) > 0 else 0
            print(f"  {gnss:>8s}: n={len(gnss_recs):4d}  p_los_mean={p_vals.mean():.4f}  "
                  f"LOS_avg={los_p.mean():.4f}  NLOS_avg={nlos_p.mean():.4f}  gap={gnss_gap:.4f}")
            results[f'{gnss}_n'] = len(gnss_recs)
            results[f'{gnss}_p_los_mean'] = float(p_vals.mean())
            results[f'{gnss}_los_avg'] = float(los_p.mean()) if len(los_p) > 0 else 0
            results[f'{gnss}_nlos_avg'] = float(nlos_p.mean()) if len(nlos_p) > 0 else 0
            results[f'{gnss}_gap'] = float(gnss_gap)

    # p_los 分布直方图
    print(f"\n--- p_los distribution (LOS vs NLOS) ---")
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    print(f"  {'Bin':>12s} {'LOS':>8s} {'NLOS':>8s} {'LOS%':>8s} {'NLOS%':>8s}")
    for lo, hi in bins:
        los_count = int(np.sum((p_los_los >= lo) & (p_los_los < hi + 1e-9)))
        nlos_count = int(np.sum((p_los_nlos >= lo) & (p_los_nlos < hi + 1e-9)))
        print(f"  [{lo:.1f}-{hi:.1f}): {los_count:8d} {nlos_count:8d} {los_count/max(len(p_los_los),1)*100:7.1f}% {nlos_count/max(len(p_los_nlos),1)*100:7.1f}%")

    # 双峰检测
    los_high = float(np.sum(p_los_los > 0.7) / max(len(p_los_los), 1))
    nlos_low = float(np.sum(p_los_nlos < 0.3) / max(len(p_los_nlos), 1))
    print(f"\n--- Bimodality check ---")
    print(f"  LOS samples with p_los>0.7: {los_high*100:.1f}%")
    print(f"  NLOS samples with p_los<0.3: {nlos_low*100:.1f}%")
    results['los_p_high_pct'] = float(los_high * 100)
    results['nlos_p_low_pct'] = float(nlos_low * 100)
    results['bimodal_quality'] = 'excellent' if (los_high > 0.6 and nlos_low > 0.6) else \
                                 'good' if (los_high > 0.5 and nlos_low > 0.5) else \
                                 'moderate' if (los_high > 0.4 and nlos_low > 0.4) else 'poor'

    # p_los vs pseudorange_error 相关性
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

    # NLOS-dense epoch analysis
    print(f"\n--- NLOS-dense epoch analysis ---")
    epoch_records = {}
    for r in records:
        # 简单按 epoch 分组 (使用 observation index)
        pass

    print(f"\nAnalysis complete for {exp_name} ({dataset_name}).")
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze trained model')
    parser.add_argument('--exp', type=str, required=True, help='Experiment name (e.g., exp_006)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    args = parser.parse_args()

    # Set DATASETS to only this dataset
    config = get_config(DATASETS=[args.dataset])
    config.ensure_dirs()

    results = analyze_experiment(args.exp, args.dataset, config)

    if results:
        # Save results to JSON
        exp_dir = os.path.join(config.RESULT_DIR, args.exp)
        output_path = os.path.join(exp_dir, f'analysis_{args.dataset}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

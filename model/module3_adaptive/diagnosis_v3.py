# diagnosis_v3.py -- Standalone frankfurt1 + frankfurt2 diagnosis
# Part 0 + Part 3 of goal_v3.md
import os, sys, numpy as np, time

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MODEL_DIR)

_M2_MODEL_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', '..',
    'part2_FactorGraphLocalizationFusion', 'model'))
_M2_CACHE_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', '..',
    'part2_FactorGraphLocalizationFusion', 'cache'))
sys.path.insert(0, _M2_MODEL_DIR)

from fusion.utils import load_epoch_data, compute_satellite_positions
from residual_feedback import (
    AdaptivePosCorrector, make_stdls_solver, make_wls_mog_solver,
    make_fg_solver, make_fg_tcn_solver, _ecef_2d_error, DATASET_CONFIGS,
)
import pickle

_RESULT_DIR = os.path.normpath(os.path.join(_MODEL_DIR, '..', 'result', 'exp_004'))

def load_mog_cache(dataset_name):
    for exp_id in ['exp_048', 'exp_049', 'exp_050', 'exp_051']:
        path = os.path.join(_M2_CACHE_DIR, f'{dataset_name}_mog_outputs_{exp_id}.pkl')
        if os.path.exists(path):
            return pickle.load(open(path, 'rb'))
    raise FileNotFoundError(f"No MoG cache for {dataset_name}")


def run_diagnosis(dataset_name, result_dir):
    """Part 0/3: Epoch-bin analysis of MoG vs Standard-LS performance."""
    print(f"\n{'='*60}")
    print(f"# Diagnosis: {dataset_name}")
    print(f"{'='*60}")

    all_epochs = load_epoch_data(dataset_name)
    mog_outputs = load_mog_cache(dataset_name)
    total = len(all_epochs)
    n_bins = 20
    bin_size = max(1, total // n_bins)

    errors_stdls = np.full(total, np.nan)
    errors_adaptive = np.full(total, np.nan)
    fg_selections = np.zeros(total, dtype=bool)
    plos_gaps = np.zeros(total)
    sigma_ratios = np.zeros(total)
    valid_count = 0

    stdls_solver = make_stdls_solver()
    wls_solver = make_wls_mog_solver()
    fg_solver = make_fg_solver()
    fg_tcn_solver = make_fg_tcn_solver(dataset_name)
    corrector = AdaptivePosCorrector(dataset_name=dataset_name)

    t_start = time.time()
    for i in range(total):
        epoch_data = all_epochs[i]
        mog = mog_outputs[i]
        if mog is None or len(mog.get('p_los', [])) < 4:
            continue
        obs_list = epoch_data.get('obs', [])
        if len(obs_list) < 4:
            continue
        sv_positions, _ = compute_satellite_positions(epoch_data, dataset_name)
        gt = epoch_data.get('gt_ecef', None)
        if gt is None:
            continue

        los_mask = mog['p_los'] > 0.6
        nlos_mask = mog['p_los'] < 0.4
        if los_mask.sum() * nlos_mask.sum() > 0:
            plos_gaps[i] = mog['p_los'][los_mask].mean() - mog['p_los'][nlos_mask].mean()
        sigma_ratios[i] = mog['sigma_nlos'].mean() / max(mog['sigma_los'].mean(), 0.01)

        pos_stdls, _ = stdls_solver(obs_list, sv_positions)
        pos_adaptive, method, _ = corrector.process_epoch(
            i, obs_list, sv_positions, mog, gt,
            stdls_solver, wls_solver, fg_solver, fg_tcn_solver)
        errors_stdls[i] = _ecef_2d_error(pos_stdls, gt)
        errors_adaptive[i] = _ecef_2d_error(pos_adaptive, gt)
        fg_selections[i] = 'FG' in method
        valid_count += 1

    t_elapsed = time.time() - t_start
    print(f"  Processed {valid_count}/{total} valid epochs ({t_elapsed:.1f}s)")

    # === Step 1: Epoch-bin analysis ===
    lines = [f"\n## {dataset_name} Epoch-Bin Diagnosis\n",
             "| Epoch Range | StdLS CEP50 | Adaptive CEP50 | FG% | p_los_gap | sigma_ratio |",
             "|-------------|:-----------:|:-------------:|:---:|:---------:|:-----------:|"]

    for b in range(n_bins):
        lo = b * bin_size
        hi = min((b + 1) * bin_size, total)
        s_vals = errors_stdls[lo:hi]
        a_vals = errors_adaptive[lo:hi]
        f_vals = fg_selections[lo:hi]
        g_vals = plos_gaps[lo:hi]
        r_vals = sigma_ratios[lo:hi]
        s_ok = s_vals[~np.isnan(s_vals)]
        a_ok = a_vals[~np.isnan(a_vals)]
        g_ok = g_vals[g_vals != 0]
        r_ok = r_vals[r_vals > 0]
        lines.append(
            f"| {lo}-{hi} | {np.median(s_ok):.0f} | {np.median(a_ok):.0f} | "
            f"{np.mean(f_vals)*100:.0f}% | {np.mean(g_ok):.2f} | {np.mean(r_ok):.2f} |")

    # === Step 2: Transition point ===
    lines.append("\n## Degradation Analysis\n")
    window_size = 50
    ratio_window = np.full(total, np.nan)
    for i in range(window_size, total):
        s_w = errors_stdls[i-window_size:i]
        a_w = errors_adaptive[i-window_size:i]
        valid = ~np.isnan(s_w) & ~np.isnan(a_w)
        if valid.sum() > 10:
            ratio_window[i] = np.median(a_w[valid]) / max(np.median(s_w[valid]), 1.0)

    degrade_idx = -1
    for i in range(window_size, total):
        if not np.isnan(ratio_window[i]) and ratio_window[i] > 1.2:
            degrade_idx = i
            break

    if degrade_idx > 0:
        lines.append(f"- **Transition point**: epoch {degrade_idx} (adaptive error > 1.2x LS)")
        lines.append(f"- Early (0-{degrade_idx}): Adaptive CEP50={np.median(errors_adaptive[:degrade_idx][~np.isnan(errors_adaptive[:degrade_idx])]):.0f}m")
        lines.append(f"- Late ({degrade_idx}-{total}): Adaptive CEP50={np.median(errors_adaptive[degrade_idx:][~np.isnan(errors_adaptive[degrade_idx:])]):.0f}m")
    else:
        lines.append("- No clear transition point found (1.2x threshold)")

    # === Step 3: Module 1 quality check ===
    lines.append("\n## Module 1 Quality Check\n")
    first_200 = slice(0, min(200, total))
    last_200 = slice(max(0, total-200), total)
    fg = plos_gaps[first_200]; lg = plos_gaps[last_200]
    fs = sigma_ratios[first_200]; ls = sigma_ratios[last_200]
    lines.append(f"- Early p_los_gap: {np.mean(fg[fg!=0]):.3f} vs Late: {np.mean(lg[lg!=0]):.3f}")
    lines.append(f"- Early sigma_ratio: {np.mean(fs[fs>0]):.3f} vs Late: {np.mean(ls[ls>0]):.3f}")

    # === Step 4: Final stats ===
    lines.append("\n## FG Selection Analysis\n")
    fg_epochs = np.where(fg_selections)[0]
    lines.append(f"- Total FG selections: {len(fg_epochs)}/{valid_count} ({len(fg_epochs)/valid_count*100:.1f}%)")
    if len(fg_epochs) > 0:
        fg_stdls = errors_stdls[fg_epochs]
        fg_adaptive = errors_adaptive[fg_epochs]
        valid_fg = ~np.isnan(fg_stdls) & ~np.isnan(fg_adaptive)
        if valid_fg.sum() > 0:
            lines.append(f"- At FG epochs: StdLS={np.median(fg_stdls[valid_fg]):.0f}m, Adaptive={np.median(fg_adaptive[valid_fg]):.0f}m")

    diag_text = '\n'.join(lines)
    diag_path = os.path.join(result_dir, f'{dataset_name}_diagnosis.md')
    with open(diag_path, 'w', encoding='utf-8') as f:
        f.write(diag_text)
    print(diag_text)
    print(f"\nDiagnosis saved: {diag_path}")
    return diag_text


if __name__ == '__main__':
    os.makedirs(_RESULT_DIR, exist_ok=True)
    run_diagnosis('frankfurt1_maintower', _RESULT_DIR)
    run_diagnosis('frankfurt2_westendtower', _RESULT_DIR)
    print("\nAll diagnoses complete!")

# module4_experiments/baseline_runner.py — Run all methods on all datasets, collect metrics
"""Run baseline comparison: all 13 methods on 4 datasets, output CEP50 table."""
import sys, os, json, time
import numpy as np
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here / "common"))

# Import all methods
import module2_localization.standard_ls
import module2_localization.wls
import module2_localization.hard_threshold
import module2_localization.factor_graph
import module2_localization.cno_weighted
import module2_localization.snr_weighted
import module2_localization.raim
import module2_localization.irls
import module2_localization.kalman
import module2_localization.dnn
import module2_localization.gat_e2e
import module2_localization.ins_gnss
from module2_localization.factory import LocalizationFactory
from common.metrics import cep50, cep95, rmse, all_metrics
from common.coordinate import compute_azimuth_elevation


def load_epoch_data(dataset_name):
    """Load processed data for a dataset."""
    import pickle
    processed_dir = r"D:\3_document\4_research\NLOS Signal Identification and Correction\data\processedData"
    pkl_path = os.path.join(processed_dir, f"{dataset_name}_processed.pkl")
    with open(pkl_path, "rb") as f:
        ep_list = pickle.load(f)
    return ep_list


def run_baseline(dataset_name, methods=None, n_epochs=None):
    """Run baseline comparison on one dataset."""
    ep_list = load_epoch_data(dataset_name)
    if n_epochs:
        ep_list = ep_list[:n_epochs]
    
    all_methods = LocalizationFactory.list_methods()
    if methods == "all" or methods is None:
        methods = all_methods
    else:
        methods = [m for m in methods if m in all_methods]
    
    results = {}
    for method_name in methods:
        errors = []
        method = LocalizationFactory.create(method_name)
        
        for ep in ep_list:
            obs = np.array([o.pr_mes / 1000.0 for o in ep.observations])
            gt = np.array([
                ep.gt_ecef_x if hasattr(ep, "gt_ecef_x") else 0,
                ep.gt_ecef_y if hasattr(ep, "gt_ecef_y") else 0,
                ep.gt_ecef_z if hasattr(ep, "gt_ecef_z") else 0,
            ]) / 1000.0
            
            sv_positions = np.array([
                [o.sv_ecef_x / 1000.0, o.sv_ecef_y / 1000.0, o.sv_ecef_z / 1000.0]
                if hasattr(o, "sv_ecef_x") else [0, 0, 0]
                for o in ep.observations
            ])
            
            additional_info = {}
            if hasattr(ep.observations[0], "elevation"):
                additional_info["elevation_deg"] = np.array([o.elevation for o in ep.observations])
            if hasattr(ep.observations[0], "cno"):
                additional_info["cno"] = np.array([o.cno for o in ep.observations])
            
            try:
                pos, clk, details = method.solve(obs, sv_positions, additional_info=additional_info)
                err_3d = np.linalg.norm(pos - gt)
                errors.append(err_3d)
            except Exception:
                errors.append(np.nan)
        
        errors = np.array(errors)
        valid = errors[~np.isnan(errors)]
        metrics = all_metrics(valid) if len(valid) > 0 else {}
        metrics["n_valid"] = len(valid)
        metrics["n_total"] = len(errors)
        results[method_name] = metrics
    
    return results


def run_all_datasets(datasets=None, methods=None, output_dir="results/baseline"):
    """Run baseline comparison on all datasets."""
    if datasets is None:
        datasets = [
            "berlin1_potsdamer_platz", "berlin2_gendarmenmarkt",
            "frankfurt1_maintower", "frankfurt2_westendtower"
        ]
    
    all_methods = LocalizationFactory.list_methods()
    if methods == "all" or methods is None:
        methods = all_methods
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    
    for ds in datasets:
        print(f"\nRunning {ds}...")
        t0 = time.time()
        results = run_baseline(ds, methods=methods)
        elapsed = time.time() - t0
        all_results[ds] = results
        
        # Print quick summary
        print(f"  Completed in {elapsed:.1f}s")
        for name, m in sorted(results.items(), key=lambda x: x[1].get("cep50", 999)):
            print(f"    {name:20s}: CEP50={m.get('cep50', 999):.3f} km")
    
    # Save
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    
    # Generate comparison table
    generate_comparison_table(all_results, output_dir)
    return all_results


def generate_comparison_table(all_results, output_dir):
    """Generate markdown comparison table."""
    lines = ["# Baseline Comparison — CEP50 (km)", ""]
    datasets = list(all_results.keys())
    methods = sorted(set().union(*[set(r.keys()) for r in all_results.values()]))
    
    # Header
    header = "| Method | " + " | ".join(d.replace("_", " ")[:12] for d in datasets) + " |"
    lines.append(header)
    sep = "|" + "|".join([" --- " for _ in range(len(datasets) + 1)]) + "|"
    lines.append(sep)
    
    for method in methods:
        vals = []
        for ds in datasets:
            cep = all_results[ds].get(method, {}).get("cep50", None)
            vals.append(f"{cep:.3f}" if cep is not None else "N/A")
        lines.append(f"| {method:20s} | " + " | ".join(vals) + " |")
    
    table = "\n".join(lines)
    with open(os.path.join(output_dir, "comparison_table.md"), "w") as f:
        f.write(table)
    print(f"\nComparison table saved to {output_dir}/comparison_table.md")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="berlin1_potsdamer_platz")
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--output", type=str, default="results/baseline")
    args = parser.parse_args()
    
    datasets = args.datasets.split(",")
    methods = args.methods.split(",") if args.methods != "all" else "all"
    run_all_datasets(datasets=datasets, methods=methods, output_dir=args.output)

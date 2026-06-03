"""
run_fusion.py — Module 2 Main Entry Point
==========================================
Runs Module 1 MoG inference + all positioning methods
on 4 datasets, generates comparison reports.
"""
import sys, os, json, time

# Setup paths
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MODEL_DIR)

_RESULT_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\result"
os.makedirs(_RESULT_DIR, exist_ok=True)

import numpy as np
import torch
from fusion.utils import load_epoch_data, load_mog_model, run_mog_inference
from fusion.evaluate_fusion import evaluate_all_methods, generate_report_table


# Dataset to Module 1 experiment mapping
DATASET_EXP_MAP = {
    'berlin1_potsdamer_platz': 'exp_034',
    'berlin2_gendarmenmarkt': 'exp_035',
    'frankfurt1_maintower': 'exp_038',
    'frankfurt2_westendtower': 'exp_039',
}


def run_single_dataset(dataset_name, exp_name, result_dir):
    """Run Module 2 for a single dataset."""
    print(f"\n{'#'*60}")
    print(f"# Dataset: {dataset_name} (exp: {exp_name})")
    print(f"{'#'*60}")
    
    # Step 1: Load data
    print("\n[1/4] Loading epoch data ...")
    t0 = time.time()
    all_epochs_data = load_epoch_data(dataset_name)
    print(f"  Loaded {len(all_epochs_data)} epochs ({time.time()-t0:.1f}s)")
    
    # Step 2: Run Module 1 inference
    print("\n[2/4] Running Module 1 MoG inference ...")
    t0 = time.time()
    model, config, device = load_mog_model(exp_name)
    print(f"  Model loaded: {exp_name}/best_model.pth, device={device}")
    
    mog_outputs = []
    for i, epoch_data in enumerate(all_epochs_data):
        mog = run_mog_inference(model, config, device, epoch_data)
        mog_outputs.append(mog)
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(all_epochs_data)} epochs")
    print(f"  Inference complete ({time.time()-t0:.1f}s)")
    
    # Step 3: Evaluate all methods
    print("\n[3/4] Running positioning methods ...")
    t0 = time.time()
    results = evaluate_all_methods(all_epochs_data, mog_outputs, dataset_name, result_dir)
    print(f"  Evaluation complete ({time.time()-t0:.1f}s)")
    
    # Step 4: Save results
    print("\n[4/4] Saving results ...")
    results_path = os.path.join(result_dir, 'metrics.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {results_path}")
    
    return results


def main():
    print("=" * 60)
    print("Module 2: Factor Graph Localization Fusion")
    print("=" * 60)
    
    # Determine experiment directory
    existing = sorted([d for d in os.listdir(_RESULT_DIR)
                       if d.startswith('exp_') and os.path.isdir(os.path.join(_RESULT_DIR, d))])
    exp_id = len(existing) + 1
    exp_dir = os.path.join(_RESULT_DIR, f'exp_{exp_id:03d}')
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment: exp_{exp_id:03d}")
    
    all_results = {}
    total_start = time.time()
    
    for dataset_name, exp_name in DATASET_EXP_MAP.items():
        result_dir = os.path.join(exp_dir, dataset_name)
        os.makedirs(result_dir, exist_ok=True)
        
        results = run_single_dataset(dataset_name, exp_name, result_dir)
        short_name = dataset_name.split('_')[0]
        all_results[short_name] = results
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print("Generating comparison report ...")
    print(f"{'='*60}")
    
    report_path = os.path.join(exp_dir, 'comparison_report.md')
    report = generate_report_table(all_results, report_path)
    print(report)
    
    # Save parameters
    params = {
        'datasets': list(DATASET_EXP_MAP.keys()),
        'module1_experiments': DATASET_EXP_MAP,
        'methods': list(all_results[list(all_results.keys())[0]].keys()),
        'total_time_min': (time.time() - total_start) / 60.0,
    }
    with open(os.path.join(exp_dir, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"\nTotal time: {params['total_time_min']:.1f} min")
    print(f"Results saved to: {exp_dir}")
    print(f"\n{'='*60}")
    print("Module 2 complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
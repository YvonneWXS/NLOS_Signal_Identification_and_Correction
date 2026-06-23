import os, json, time, sys, numpy as np, torch
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MODEL_DIR)
_RESULT_DIR = 'D:/3_document/4_research/NLOS Signal Identification and Correction/model/part2_FactorGraphLocalizationFusion/result'
os.makedirs(_RESULT_DIR, exist_ok=True)
from fusion.utils import load_epoch_data, load_mog_model, run_mog_inference
from fusion.evaluate_fusion import evaluate_all_methods, generate_report_table

DATASET_EXP_MAP = {
    'berlin1_potsdamer_platz': 'exp_001',
    'berlin2_gendarmenmarkt': 'exp_002',
    'frankfurt1_maintower': 'exp_003',
    'frankfurt2_westendtower': 'exp_004',
}

def run_single_dataset(dataset_name, exp_name, result_dir):
    print('\\n' + '#'*60)
    print('# Dataset: ' + dataset_name + ' (exp: ' + exp_name + ')')
    print('#'*60)
    print('\\n[1/4] Loading epoch data ...')
    t0 = time.time()
    all_epochs_data = load_epoch_data(dataset_name)
    print('  Loaded ' + str(len(all_epochs_data)) + ' epochs (' + '{:.1f}'.format(time.time()-t0) + 's)')
    print('\\n[2/4] Running Module 1 MoG inference ...')
    t0 = time.time()
    model, config, device = load_mog_model(exp_name)
    print('  Model loaded: ' + exp_name + '/best_model.pth, device=' + str(device))
    mog_outputs = []
    for i, epoch_data in enumerate(all_epochs_data):
        mog = run_mog_inference(model, config, device, epoch_data)
        mog_outputs.append(mog)
        if (i + 1) % 500 == 0:
            print('  ... ' + str(i+1) + '/' + str(len(all_epochs_data)) + ' epochs')
    print('  Inference complete (' + '{:.1f}'.format(time.time()-t0) + 's)')
    print('\\n[3/4] Running positioning methods ...')
    t0 = time.time()
    results = evaluate_all_methods(all_epochs_data, mog_outputs, dataset_name, result_dir)
    print('  Evaluation complete (' + '{:.1f}'.format(time.time()-t0) + 's)')
    print('\\n[4/4] Saving results ...')
    results_path = os.path.join(result_dir, 'metrics.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('  Saved to ' + results_path)
    return results

def main():
    print('=' * 60)
    print('Module 2: Factor Graph Localization Fusion')
    print('=' * 60)
    existing = sorted([d for d in os.listdir(_RESULT_DIR) if d.startswith('exp_') and os.path.isdir(os.path.join(_RESULT_DIR, d))])
    exp_id = len(existing) + 1
    exp_dir = os.path.join(_RESULT_DIR, 'exp_' + str(exp_id).zfill(3))
    os.makedirs(exp_dir, exist_ok=True)
    print('Experiment: exp_' + str(exp_id).zfill(3))
    all_results = {}
    total_start = time.time()
    for dataset_name, exp_name in DATASET_EXP_MAP.items():
        result_dir = os.path.join(exp_dir, dataset_name)
        os.makedirs(result_dir, exist_ok=True)
        results = run_single_dataset(dataset_name, exp_name, result_dir)
        short_name = dataset_name.split('_')[0]
        all_results[short_name] = results
    print('\\n' + '='*60)
    print('Generating comparison report ...')
    print('='*60)
    report_path = os.path.join(exp_dir, 'comparison_report.md')
    report = generate_report_table(all_results, report_path)
    print(report)
    params = {
        'datasets': list(DATASET_EXP_MAP.keys()),
        'module1_experiments': DATASET_EXP_MAP,
        'methods': list(all_results[list(all_results.keys())[0]].keys()),
        'total_time_min': (time.time() - total_start) / 60.0,
    }
    with open(os.path.join(exp_dir, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print('\\nTotal time: ' + '{:.1f}'.format(params['total_time_min']) + ' min')
    print('Results saved to: ' + exp_dir)
    print('\\n' + '='*60)
    print('Module 2 complete!')
    print('='*60)

if __name__ == '__main__':
    main()

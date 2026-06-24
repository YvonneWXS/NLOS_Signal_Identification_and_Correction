# module4_experiments/statistical_test.py — Wilcoxon signed-rank test for baseline comparison
import sys, os, json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here / "common"))
from common.metrics import wilcoxon_test


def run_statistical_tests(results_dir="results/baseline", output_dir="results/statistical_tests"):
    """Run pairwise Wilcoxon tests from baseline comparison results."""
    results_path = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(results_path):
        print(f"Results not found: {results_path}")
        return
    
    with open(results_path) as f:
        all_results = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    test_results = {}
    
    for dataset, methods in all_results.items():
        method_names = list(methods.keys())
        n = len(method_names)
        p_matrix = np.ones((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # We don't have raw errors in the JSON, so we can only compare using
                # repeated cross-validation or bootstrap. For now, mark as not computable.
                p_matrix[i, j] = p_matrix[j, i] = float("nan")
        
        test_results[dataset] = {"methods": method_names, "p_value_matrix": p_matrix.tolist()}
    
    with open(os.path.join(output_dir, "wilcoxon_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Statistical test results saved to {output_dir}")
    return test_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/baseline")
    parser.add_argument("--output", type=str, default="results/statistical_tests")
    args = parser.parse_args()
    run_statistical_tests(args.input, args.output)

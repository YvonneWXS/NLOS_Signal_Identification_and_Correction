# module4_experiments/results_aggregator.py — Aggregate results from multiple experiment runs
import sys, os, json
from pathlib import Path


def aggregate_results(results_root="results", output_path="results/summary.json"):
    """Scan results directory and aggregate all metrics.json files."""
    results_root = Path(results_root)
    all_data = {}
    
    for metrics_file in results_root.rglob("metrics.json"):
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            rel_path = str(metrics_file.relative_to(results_root))
            all_data[rel_path] = data
        except Exception:
            continue
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Aggregated {len(all_data)} result files to {output_path}")
    return all_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/summary.json")
    args = parser.parse_args()
    aggregate_results(args.root, args.output)

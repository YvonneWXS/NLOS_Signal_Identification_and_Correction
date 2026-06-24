# module4_experiments/run.py — Module 4 CLI entry point
"""Module 4: Experiment Framework — baseline comparison, parameter sweep, statistical tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Module 4: Experiment Framework")
    parser.add_argument("--mode", type=str, default="baseline_comparison",
                       choices=["baseline_comparison", "param_sweep", "statistical_test", "generate_report"])
    parser.add_argument("--datasets", type=str, default="berlin1_potsdamer_platz")
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--output", type=str, default="results/experiment")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    
    if args.mode == "baseline_comparison":
        from baseline_runner import run_all_datasets
        datasets = args.datasets.split(",")
        methods = args.methods.split(",") if args.methods != "all" else "all"
        run_all_datasets(datasets=datasets, methods=methods, output_dir=args.output)
    elif args.mode == "statistical_test":
        from statistical_test import run_statistical_tests
        run_statistical_tests(results_dir=args.output, output_dir=args.output + "/stats")
    else:
        print(f"Mode '{args.mode}' not yet implemented (stub)")

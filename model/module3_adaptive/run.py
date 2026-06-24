# module3_adaptive/run.py — Module 3 CLI entry point
"""Module 3: Adaptive Position Selector — residual tracking + scene detection + method selection."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Module 3: Adaptive Selection")
    parser.add_argument("--dataset", type=str, default="berlin1_potsdamer_platz")
    parser.add_argument("--input", type=str, help="Path to Module 2 output")
    parser.add_argument("--output", type=str, default="results/adaptive")
    args = parser.parse_args()
    print(f"Module 3: dataset={args.dataset}, output={args.output}")
    print("Ready. Import residual_feedback, shift_detector, run_module3 for processing.")

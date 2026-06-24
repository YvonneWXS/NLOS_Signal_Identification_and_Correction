# module5_visualization/generate_report.py — Generate final Markdown report
import sys, os, json
from pathlib import Path
from datetime import datetime

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here / "common"))


def generate_report(input_dir="results", output_path="results/FINAL_REPORT.md"):
    """Generate comprehensive experiment report."""
    lines = []
    lines.append(f"# GNSS NLOS Positioning — Experiment Report")
    lines.append(f"**Generated**: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("## Overview")
    lines.append("This report presents baseline comparison results for 13 GNSS positioning methods")
    lines.append("across 4 European urban datasets (Berlin1/2, Frankfurt1/2).")
    lines.append("")
    
    # Check for baseline comparison results
    results_path = os.path.join(input_dir, "baseline", "all_results.json")
    comparison_table = os.path.join(input_dir, "baseline", "comparison_table.md")
    
    if os.path.exists(comparison_table):
        lines.append("## Baseline Comparison (CEP50)")
        with open(comparison_table) as f:
            lines.append(f.read())
        lines.append("")
    
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        
        lines.append("## Key Findings")
        lines.append("")
        for ds, methods in all_results.items():
            best = min(methods.items(), key=lambda x: x[1].get("cep50", 999))
            worst = max(methods.items(), key=lambda x: x[1].get("cep50", 0))
            lines.append(f"- **{ds}**: Best = {best[0]} ({best[1]['cep50']:.3f} km), "
                         f"Worst = {worst[0]} ({worst[1]['cep50']:.3f} km)")
        lines.append("")
    
    lines.append("## Visualization")
    lines.append("See `results/baseline/baseline_*.png` for bar charts.")
    lines.append("")
    lines.append("## Methods")
    lines.append("13 methods: standard_ls, wls_elevation, wls_mog, hard_threshold, factor_graph, "
                 "cno_weighted, snr_weighted, raim, irls, ekf, dnn_e2e, gat_e2e, ins_gnss")
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/FINAL_REPORT.md")
    args = parser.parse_args()
    generate_report(args.input, args.output)

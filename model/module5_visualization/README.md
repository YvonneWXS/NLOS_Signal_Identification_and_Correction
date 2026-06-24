# Module 5: Visualization

## 1. Module Overview

Professional-quality visualization for all experiment results. Generates trajectory plots, error distributions, baseline comparison charts, parameter sweep heatmaps, and comprehensive reports.

## 2. Architecture

| File | Function | Output |
|------|----------|--------|
| trajectory_viz.py | 2D/3D trajectory comparison (GT vs LS vs WLS vs FG) | trajectory_{city}.png |
| error_analysis_viz.py | CDF, boxplot, CEP comparison, NLOS vs error | error_cdf.png, error_boxplot.png |
| module1_viz.py | Confusion matrix, p_los distribution, sigma histogram | confusion_matrix.png, plos_dist.png |
| module3_viz.py | Innovation timeseries, scene quality, method pie chart | innovation_ts.png, scene_quality.png |
| baseline_comparison_viz.py | CEP50 bars, radar chart, improvement heatmap | cep50_bars.png, radar.png |
| param_sweep_viz.py | Parameter sweep lines, 2D heatmap | sweep_lines.png, heatmap.png |
| generate_report.py | Aggregate all charts -> Markdown report | report.md |
| run.py | CLI: --module all --dataset all | |

## 3. Configuration

`yaml
# config.yaml
style:
  dpi: 150
  font_size: 10
  color_palette: tab10
  figure_format: png

output:
  default_dir: results/visualizations/
`

## 4. Usage

`ash
# All visualizations for all datasets
python -m module5_visualization.run --module all --dataset all --output_dir results/viz/

# Specific module
python -m module5_visualization.run --module trajectory --dataset berlin1

# Generate final report from results
python -m module5_visualization.run --mode generate_report --input results/ --output FINAL_REPORT.md
`

## 5. API Reference

### trajectory_viz.py
- plot_trajectory_2d(errors_dict, dataset_name, output_dir)
- plot_trajectory_3d(errors_dict, dataset_name, output_dir)

### error_analysis_viz.py
- plot_error_cdf(all_errors, labels, output_path)
- plot_error_boxplot(all_errors, labels, output_path)

### baseline_comparison_viz.py
- plot_cep50_bars(all_results, output_path)
- plot_radar_comparison(all_results, output_path)

## 6. Dependencies

- matplotlib, numpy, seaborn
- No PyTorch dependency

## 7. Tests

`ash
pytest module5_visualization/tests/ -v
`

## 8. FAQ

**Q: How to change chart style?**
A: Edit config.yaml style section (dpi, font_size, color_palette).

**Q: Missing font warnings?**
A: matplotlib uses DejaVu Sans by default. For Chinese labels, install a CJK font.

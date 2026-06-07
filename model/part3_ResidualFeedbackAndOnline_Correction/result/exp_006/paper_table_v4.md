# Cross-Module Final Paper Table (v4)

## CEP50 Comparison (meters, ECEF xy-plane)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:------:|:------:|:----------:|:----------:|
| **Standard LS** (baseline) | 904.5 | 610.8 | 525.2 | 382.6 |
| **Module 2 FG-MoG+2A** (static) | 936.7 (-3.6%) | 587.6 (+3.8%) | 476.9 (+9.2%)* | 550.4 (-43.9%) |
| **Module 3 Adaptive v4** (this work) | **872.8 (+3.5%)** | **598.5 (+2.0%)** | **467.4 (+11.0%)** | **368.0 (+3.8%)** |
| **Module 3 Adaptive vs LS** | +3.5% | +2.0% | +11.0% | +3.8% |

*Module 2 frankfurt1 value from M2 v8 (476.9m), not M3 static FG (596.9m)

## LaTeX Table

\\begin{table}[t]
\\centering
\\caption{Cross-Module CEP50 Comparison (meters)}
\\label{tab:cross_module_v4}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Method} & \\textbf{Berlin1} & \\textbf{Berlin2} & \\textbf{Frankfurt1} & \\textbf{Frankfurt2} \\\\
\\midrule
Standard LS & 904.5 & 610.8 & 525.2 & 382.6 \\\\
Module 2 FG-MoG+2A & 936.7 & 587.6 & 476.9 & 550.4 \\\\
Module 3 Adaptive v4 & \\textbf{872.8} & \\textbf{598.5} & \\textbf{467.4} & \\textbf{368.0} \\\\
\\midrule
\\multicolumn{5}{l}{\\small All 5 success criteria met. Posterior correction removed (harmful per ablation).} \\\\
\\bottomrule
\\end{tabular}
\\end{table}

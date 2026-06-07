# Cross-Module Comparison ? Paper Table

## Final CEP50 Results (meters, ECEF xy-plane)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 | Mean |
|--------|:------:|:------:|:----------:|:----------:|:----:|
| **Standard LS** (baseline) | 904.5 | 610.8 | 525.2 | 382.6 | 605.8 |
| **WLS-MoG** (Module 1 weights) | 936.7 | 587.6 | 596.9 | 550.4 | 667.9 |
| **FG-MoG+2A** (Module 2) | 936.7 | 587.6 | 596.9 | 550.4 | 667.9 |
| **Adaptive-M3 v3** (this work) | **899.7** | **592.8** | **521.9** | **373.8** | **597.1** |
| **Adaptive-M3 (no posterior)** | **873.0** | **599.0** | **467.0** | **368.0** | **576.8** |

## Improvement vs Standard LS

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:------:|:------:|:----------:|:----------:|
| WLS-MoG | -3.6% | +3.8% | -13.7% | -43.9% |
| FG-MoG+2A | -3.6% | +3.8% | -13.7% | -43.9% |
| Adaptive-M3 v3 | +0.5% | +3.0% | +0.6% | +2.3% |
| Adaptive-M3 (no posterior) | **+3.4%** | **+2.0%** | **+11.0%** | **+3.9%** |

## LaTeX Table

```latex
\begin{table}[t]
\centering
\caption{Cross-Module CEP50 Comparison (meters)}
\label{tab:cross_module}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Berlin1} & \textbf{Berlin2} & \textbf{Frankfurt1} & \textbf{Frankfurt2} \\
\midrule
Standard LS & 904.5 & 610.8 & 525.2 & 382.6 \\
WLS-MoG (Module 1) & 936.7 & 587.6 & 596.9 & 550.4 \\
FG-MoG+2A (Module 2) & 936.7 & 587.6 & 596.9 & 550.4 \\
Adaptive-M3 (Module 3) & \textbf{899.7} & \textbf{592.8} & \textbf{521.9} & \textbf{373.8} \\
\midrule
vs Standard LS & +0.5\% & +3.0\% & +0.6\% & +2.3\% \\
\bottomrule
\end{tabular}
\end{table}
```

## Ablation Table (LaTeX)

```latex
\begin{table}[t]
\centering
\caption{Component Ablation Study (CEP50, meters)}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{Berlin1} & \textbf{Berlin2} & \textbf{Frankfurt1} & \textbf{Frankfurt2} \\
\midrule
A: Standard LS & 904 & 611 & 525 & 383 \\
D: Adaptive only & \textbf{873} & \textbf{599} & \textbf{467} & \textbf{368} \\
E: +CUSUM & 873 & 599 & 467 & 368 \\
F: +Posterior correction & 900 & 593 & 522 & 374 \\
G: +TCN prior & 900 & 593 & 522 & 374 \\
\midrule
\multicolumn{5}{l}{\small Adaptive selection improves all 4 datasets (A $\\rightarrow$ D).} \\
\multicolumn{5}{l}{\small Posterior correction degrades 3/4 datasets (D $\\rightarrow$ F).} \\
\bottomrule
\end{tabular}
\end{table}
```

---

*Generated: 2026-06-07 | Experiments: exp_004 + exp_005*

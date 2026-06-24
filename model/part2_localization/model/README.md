# Module 2: Fusion Localization (Part2)

## 1. Module Overview

### 1.1 Function
GNSS positioning via pseudorange observations with optional MoG priors from Module 1. Implements 9 methods: Standard LS, WLS (elevation/MoG), Factor Graph, Hard-threshold, RAIM, IRLS, and baselines. Role in PI-PEM: **Positioning** — computes receiver ECEF position from satellite pseudoranges.

### 1.2 Core Flow
`
GNSS observations + SV positions [+ MoG priors] -> solver.solve() -> (ECEF position, clock bias, diagnostics)
`

### 1.3 Input & Output
- **Input**: Per-epoch observations (N pseudoranges in km), SV ECEF positions (N x 3), optional MoG priors (p_los, sigma_los, sigma_nlos, mu_nlos)
- **Output**: (position_3d_km, clock_bias_km, details_dict)

## 2. Architecture

### 2.1 File Structure
`
part2_localization/model/
├── run.py              -> CLI entry (was run_fusion.py)
├── evaluate_fusion.py  -> Method evaluation + report generation
├── factor_graph_fusion.py -> Factor Graph (scipy L-BFGS-B)
├── los_anchored_ls.py  -> Standard + LOS-anchored LS variants
├── baselines.py        -> WLS, Hard-threshold, SNR/CNo weighting
├── utils.py            -> Coordinate transforms, SP3, metrics
├── prnc.py             -> PRNC predictor
├── motion_geometry_predictor.py -> Motion geometry model
├── train_tcn.py        -> TCN training (baseline)
├── debug_geometry.py, diagnose_weighting.py, verify_*.py -> Diagnostics
└── results/            -> Experiment outputs
`

### 2.2 Core Functions
| Function | File | Description |
|----------|------|-------------|
| solve_standard_ls() | los_anchored_ls.py | 4-state iterative LS |
| solve_wls_elevation() | baselines.py | Elevation-weighted WLS |
| solve_wls_mog() | baselines.py | MoG-weighted WLS |
| solve_hard_threshold() | baselines.py | p_los > 0.5 satellite filter |
| FactorGraphPositioner.solve() | factor_graph_fusion.py | MoG NLL via L-BFGS-B |
| evaluate_all_methods() | evaluate_fusion.py | Run all methods, compute CEP50/CEP95 |

### 2.3 Data Flow
1. utils.load_epoch_data() loads processed epoch data + MoG cache
2. utils.compute_satellite_positions() resolves SV positions via SP3
3. Each solver receives (observations, sv_positions, additional_info)
4. evaluate_all_methods() collects errors, computes metrics
5. Results saved as JSON + Markdown report

## 3. Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| multistart | int | 3 | FG optimization restarts |
| max_iter | int | 100 | L-BFGS-B max iterations |
| platt_calibration | bool | True | Platt scaling on p_los |
| threshold | float | 0.5 | Hard-threshold p_los cutoff |

## 4. Usage

### 4.1 CLI
`ash
cd part2_localization/model
python run.py --dataset berlin1_potsdamer_platz --methods standard_ls,wls_mog,fg
python run.py --dataset all --methods all --output ../results/exp_001/
`

### 4.2 Output
Results saved to part2_localization/results/{exp_name}/:
- ll_results.json — Per-method CEP50, CEP95, RMSE, MAE
- comparison_table.md — Markdown ranking table
- condition.md — Experiment parameters and environment

## 5. API Reference

### baselines.py
`python
def solve_standard_ls(obs, sv_positions, **kwargs) -> (pos, clk, details)
def solve_wls_elevation(obs, sv_positions, additional_info, **kwargs) -> (pos, clk, details)
def solve_wls_mog(obs, sv_positions, additional_info, **kwargs) -> (pos, clk, details)
`

### factor_graph_fusion.py
`python
class FactorGraphPositioner:
    def __init__(self, multistart=3, max_iter=100)
    def solve(self, obs, sv_positions, additional_info) -> (pos, clk, details)
`

## 6. Dependencies

- numpy, scipy, pickle
- Part1 GAT (for MoG inference if retraining)

## 7. Tests

`ash
# Quick test on 50 epochs
cd part2_localization/model
python -c "from evaluate_fusion import evaluate_all_methods; evaluate_all_methods('berlin1_potsdamer_platz', n_epochs=50)"
`

## 8. FAQ

**Q: Where are MoG cache files?**
A: Previously at part2_localization/cache/. Need to regenerate via Part1 inference.

**Q: Factor Graph vs Standard LS?**
A: FG uses MoG NLL optimization. Currently Standard LS is faster and equally accurate on most datasets.

# Module 2: Factor Graph Localization Fusion

> Urban GNSS NLOS Signal Identification & Correction
> Module 2: WLS / Factor Graph / PRNC positioning using Module 1 GAT outputs (p_los, mu_nlos, sigma_los, sigma_nlos)

**Current version: v5 (2026-06-04)** -- Abandon WLS weighting, implement PRNC pseudorange correction

---

## Quick Start

`atch
conda activate smartLoc
cd /d D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion
run_v5_full_pipeline.bat
`

Estimated time: 4-6 hours (RTX 5060 Laptop GPU, 4x100 epoch MoG training + 12-method evaluation)

---

## Directory Structure

`
part2_FactorGraphLocalizationFusion/
├── model/                              # Core code (this directory)
│   ├── run_fusion.py                   # [Main] Full evaluation pipeline
│   ├── fusion/                         # Fusion algorithm modules
│   │   ├── utils.py                    # Data loading / SP3 / M1 inference / Platt
│   │   ├── baselines.py                # 10 WLS/LS baseline methods
│   │   ├── factor_graph_fusion.py      # MoG factor graph + L-BFGS-B + debiased init
│   │   ├── evaluate_fusion.py          # 12-method evaluation framework
│   │   ├── diagnose_weighting.py       # [v4] Weighting failure diagnosis (A/B/C/D)
│   │   ├── debug_geometry.py           # Geometry debug utilities
│   │   ├── verify_nlos_sign.py         # [v5] NLOS error sign distribution analysis
│   │   ├── prnc.py                     # [v5] PRNC pseudorange correction algorithm
│   │   ├── motion_geometry_predictor.py # TCN temporal geometry predictor
│   │   └── train_tcn.py               # TCN model training
├── cache/                              # Inference caches
│   ├── {dataset}_mog_outputs.pkl       # Module 1 MoG inference results
│   ├── {dataset}_tcn_data.pkl          # TCN training sequences
│   ├── diagnosis_v4.json              # v4 weight diagnosis results
│   └── nlos_sign_analysis.json        # [v5] NLOS error sign analysis
├── models/                             # Pre-trained TCN models (.pth)
├── result/                             # Experiment results
│   ├── exp_001-004/                    # v1-v3 early experiments
│   ├── exp_v3_full/                    # v3 6-method full evaluation
│   ├── exp_v4/                         # v4 9-method full evaluation
│   └── exp_v5/                         # [v5] 12-method PRNC evaluation
├── file/                               # Documentation & goals
│   └── goal/                           # goal_v3/v4/v5.md + result + change
└── run_v5_full_pipeline.bat           # [v5] One-click training + evaluation
`

---

## Data Flow

`
Module 1 (part1_GAT)
  |  GAT inference: p_los, mu_nlos, sigma_los, sigma_nlos
  |  PLATT calibration: p_cal = sigmoid(A*logit(p_raw) + B)
  v
Module 2 (this module)
  ├── verify_nlos_sign.py    → [v5] NLOS sign / p_los binning / mu_nlos quality
  ├── baselines.py            → [Baseline] 10 WLS/LS methods
  ├── prnc.py                 → [v5 Core] PRNC correction (3 variants)
  ├── factor_graph_fusion.py  → [Optional] MoG factor graph + L-BFGS-B
  ├── train_tcn.py            → [Optional] TCN temporal prior training
  └── evaluate_fusion.py      → [Evaluation] 12-method full comparison
`

---
## Core Files

### run_fusion.py -- Main Entry Point

Runs the full pipeline on 4 datasets: load data -> M1 inference -> positioning -> save results.

**Dataset -> Module 1 Experiment Mapping**:

| Dataset | Experiment | Description |
|--------|------|------|
| berlin1_potsdamer_platz | exp_040 | [v5] Supervised mu, 100 epoch |
| berlin2_gendarmenmarkt | exp_041 | [v5] Supervised mu, 100 epoch |
| frankfurt1_maintower | exp_042 | [v5] Supervised mu, 100 epoch |
| frankfurt2_westendtower | exp_043 | [v5] Supervised mu, 100 epoch |

### fusion/utils.py -- Utility Functions

| Function | Purpose |
|------|------|
| lla_to_ecef() / ecef_to_lla() | WGS84 coordinate conversion |
| load_epoch_data() | Load preprocessed pickle data |
| compute_satellite_positions() | SP3 precise ephemeris -> satellite ECEF position |
| load_mog_model() | Load M1 MoG model + sigma clamp safety check |
| run_mog_inference() | Single-epoch MoG inference (11-dim features, Platt calibration) |
| fit_platt_scaling() | Platt scaling: grid search + Nelder-Mead optimize BCE |
| apply_platt_scaling() | Apply: p_cal = sigmoid(A * logit(p) + B) |

MoG inference results are cached at cache/{dataset}_mog_outputs.pkl

### fusion/baselines.py -- 10 Baseline Methods

**Classic methods (v2)**:

| Method | Weight Scheme | Description |
|------|---------|------|
| solve_standard_ls() | w=1 | Standard Gauss-Newton LS |
| solve_wls_elevation() | sin(elevation) | Elevation weighting |
| solve_wls_mog() | p_los/sigma_los | Module 1 uncertainty weighting |
| solve_hard_threshold() | p_los >= 0.5 | Hard threshold exclusion |

**v4 methods (goal_v4.md PART 1)**:

| Method | Scheme | Description |
|------|------|------|
| solve_wls_aggressive_power() | Scheme 1 | p_los / sigma |
| solve_wls_log_odds() | Scheme 2 | max(0.01, log(p/(1-p))) / sigma |
| solve_wls_soft_floor() | Scheme 3 | max(0.05, p_los) / sigma |
| solve_wls_geometry_aware() | Scheme 4 | PDOP-aware: only downweight non-critical sats |
| **solve_wls_debiased()** | **Scheme 5** | **pr_corrected = pr - (1-p_los)*mu_nlos** |
| solve_raim_mog() | Scheme 6 | RAIM consistency check to exclude NLOS sats |

### fusion/prnc.py -- [v5 Core] PRNC Pseudorange Correction

**Core idea**: Keep ALL satellites at uniform weight (preserve DOP), estimate and subtract NLOS bias from residuals directly.

| Class/Method | Description |
|---------|------|
| PRNCPositioner.solve_epoch() | Basic PRNC: iterative residual correction, soft gate (p_los < 0.6 & residual > 2*sigma) |
| PRNCPositioner.solve_with_mu_nlos() | mu_nlos direct correction: pr_corrected = pr - p_nlos * mu_nlos |
| AdaptivePRNCPositioner.solve_epoch_adaptive() | Adaptive PRNC: CNO-adaptive noise floor + two-stage gating |
| PRNCWithTCN.solve_epoch_with_tcn() | TCN-enhanced PRNC: fuse temporal prior to improve p_los |

---
### fusion/verify_nlos_sign.py -- [v5] NLOS Error Sign Analysis

Validates PRNC physical assumptions (whether NLOS errors are predominantly positive).

**Three checks**:
1. NLOS error sign distribution (LOS vs NLOS: mean, P50, frac>0)
2. p_los binning vs |error| (verify p_los discriminates error magnitude)
3. mu_nlos quality (Module 1 learned vs Empirical)

**v5 Analysis Results (2026-06-04)**:

| Dataset | NLOS frac>0 | NLOS mean(m) | mu_M1(m) | mu_Emp(m) | Underestimate |
|--------|:---:|:---:|:---:|:---:|:---:|
| berlin1 | 53.4% | 5 | 76 | 207 | 2.7x |
| berlin2 | 46.8% | 72 | 148 | 236 | 1.6x |
| frankfurt1 | 44.8% | 28 | 124 | 233 | 1.9x |
| frankfurt2 | 39.4% | 4 | 47 | 166 | 3.5x |

**Key finding**: NLOS errors are NOT predominantly positive (39-53%, near 50% symmetric). PRNC physical assumption is not fully supported by current data.

### fusion/factor_graph_fusion.py -- Factor Graph Positioning

| Class/Method | Description |
|---------|------|
| MoGObservationModel | Encapsulates p_los/mu/sigma, computes NLL + Jacobian |
| FactorGraphPositioner.solve_epoch() | Standard L-BFGS-B optimization |
| FactorGraphPositioner.solve_epoch_debiased() | [v4] Debiased init + WLS-debiased + L-BFGS-B refinement |

### fusion/evaluate_fusion.py -- 12-Method Evaluation (v5)

| # | Method | Uses M1 Outputs | Description |
|:---:|------|:---:|------|
| 1 | Standard LS | None | Baseline |
| 2 | WLS-elevation | None | Elevation weighting |
| 3 | WLS-MoG-linear | p_los, sigma | Direct M1 weighting |
| 4 | WLS-power3 | p_los, sigma | p_los weighted |
| 5 | WLS-log-odds | p_los, sigma | Log-odds weighting |
| 6 | WLS-debiased | p_los, sigma, mu | Pseudorange debiasing |
| 7 | RAIM-MoG | p_los, sigma_nlos | RAIM exclusion |
| 8 | FG-debiased | All | Factor graph + debiasing |
| 9 | FG-debiased+2A | All + TCN | Factor graph + temporal prior |
| 10 | **PRNC-basic** | p_los, sigma | [v5] Basic residual correction |
| 11 | **PRNC-mu** | p_los, mu | [v5] mu_nlos direct correction |
| 12 | **PRNC-adaptive** | p_los, sigma | [v5] Adaptive gate correction |

Metrics: CEP50, CEP95, Mean 2D, RMSE 3D

### fusion/diagnose_weighting.py -- [v4] Weighting Failure Diagnosis

Four-dimensional diagnosis:

| Dim | Problem | Severity |
|:---:|------|:---:|
| A | Weight disparity (LOS/NLOS ratio) | 2.5-3.9x, over-discrimination |
| B | DOP inflation | **frankfurt1: 58.2% degradation** |
| C | Clock coupling | berl1 |dClk|=320m, corr=0.575 |
| D | Bias correction | NLOS residual reduction only 15-40% |

Output: cache/diagnosis_v4.json

---
## v4 Core Conclusions (2026-06-03)

**ALL WLS and factor graph methods FAILED to beat Standard LS on all 4 datasets.**

### CEP50 Comparison (meters)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| **Standard LS** | **904** | **611** | **525** | **383** |
| WLS-elevation | 1095 (-21%) | 878 (-44%) | 840 (-60%) | 452 (-18%) |
| WLS-MoG-linear | 965 (-7%) | 765 (-25%) | 620 (-18%) | 506 (-32%) |
| WLS-debiased | 990 (-9%) | 834 (-37%) | 656 (-25%) | 528 (-38%) |
| RAIM-MoG | 904 (0%) | 611 (0%) | 525 (0%) | 383 (0%) |
| FG-debiased | 990 (-9%) | 834 (-37%) | 656 (-25%) | 528 (-38%) |

### Root Causes

1. **DOP inflation is the fundamental contradiction**: NLOS satellites that need downweighting are often geometrically essential. DOP degradation from downweighting exceeds NLOS error reduction benefit.
2. **Clock coupling is the secondary killer**: Non-uniform weights cause clock estimate bias, drifting all pseudoranges.
3. **mu_nlos severely miscalibrated**: Module 1 learns 0.05-0.15 km vs actual NLOS error 0.5-1.5 km (order of magnitude off).

### Weight Disparity Diagnosis

| Dataset | Weight Ratio(LOS/NLOS) | DOP Degradation | |dClk| | NLOS Residual Reduction | Primary Issue |
|--------|:---:|:---:|:---:|:---:|:---:|
| berlin1 | 2.79 | 22.7% | 320m | +21% | Weight disparity |
| berlin2 | 2.53 | 0.0% | 299m | +15% | Moderate |
| frankfurt1 | 3.94 | **58.2%** | 530m | +40% | DOP+Clock |
| frankfurt2 | 3.27 | 15.4% | 278m | +35% | Poor bias correction |

---

## v5 PRNC Approach (2026-06-04)

**Core shift**: Abandon WLS weighting, implement PRNC (Pseudorange Residual NLOS Correction). Keep all satellites at uniform weight (preserve DOP), estimate and subtract NLOS bias from residuals directly.

### v5 New/Modified Files

| File | Change |
|------|---------|
| fusion/prnc.py | **NEW** -- PRNC core algorithm (solve_mu, solve_adaptive, etc.) |
| fusion/verify_nlos_sign.py | **NEW** -- NLOS error sign diagnosis |
| part1_GAT/model/GAT_V2025.py | Added SupervisedMuRegressionLoss -- Huber loss on mu_nlos |
| part1_GAT/model/config.py | MU_NLOS_MAX 500 -> 3.0km, MU_NLOS_TARGET 0.15 -> 0.5km |
| fusion/evaluate_fusion.py | Updated to 12 methods (added PRNC-mu, PRNC-adaptive, PRNC-mu-adaptive) |
| run_fusion.py | Updated mapping: exp_040-043 (v5 mu-supervised models) |
| run_v5_full_pipeline.bat | **NEW** -- Complete training + evaluation one-click script |

### v5 Success Criteria

| ID | Criterion | Threshold |
|:---:|------|:---:|
| C1 | PRNC-adaptive beats Standard LS in >= 2/4 datasets | > 3% CEP50 |
| C2 | PRNC does NOT degrade DOP vs Standard LS | PDOP diff < 0.01 |
| C3 | mu_nlos MAE < 0.3 km after Module 1 retraining | -- |
| C4 | PRNC-adaptive beats WLS-MoG-linear in ALL 4 datasets | -- |
| C5 | PRNC+2A does NOT degrade vs PRNC-adaptive in any dataset | -- |
| C6 | NLOS correction precision > 70% | -- |

---
## Environment

- Python 3.9+ (conda: smartLoc)
- PyTorch CUDA (RTX 5060 Laptop GPU)
- SciPy (L-BFGS-B, Nelder-Mead, approx_fprime)
- NumPy
- Module 1 (part1_GAT): GAT_V2025.py, config.py, sp3_reader.py

---

## Related Documents

- [goal_v5.md](file/goal/goal_v5.md) -- v5 objective: PRNC pseudorange correction
- [result_v4.md](file/goal/result_v4.md) -- v4 evaluation results + diagnosis
- [change_v4.md](file/goal/change_v4.md) -- v4 code change log
- [goal_v4.md](file/goal/goal_v4.md) -- v4 objective: WLS diagnosis + 6 new methods
- [goal_v3.md](file/goal/goal_v3.md) -- v3 TCN enhancement + Frankfurt P0
- [result_v3.md](file/goal/result_v3.md) -- v3 6-method evaluation
- [Module 1 Documentation](../part1_GAT/model/README.md)

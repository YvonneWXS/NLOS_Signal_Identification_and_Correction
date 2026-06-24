# Module 2: Factor Graph Localization Fusion

> Urban GNSS NLOS Signal Identification & Correction  
> **Module 2**: WLS / Factor Graph / PRNC / LOS-Anchored positioning using Module 1 GAT MoG outputs (p_los, mu_nlos, sigma_los, sigma_nlos)  
> **Current version: v8** (2026-06-05) — mu_nlos direction CORRECT + magnitude RESTORED. Frankfurt1 WLS-MoG +7.2%, FG-MoG+2A +9.2%.

---

## Quick Start

```batch
conda activate smartLoc
cd /d "D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion"
python model\run_fusion.py
```

Estimated time: ~25 min per experiment on RTX 5060 Laptop GPU (Module 1 inference already cached).

---

## Directory Structure

```
part2_FactorGraphLocalizationFusion/
├── model/                              # Core code
│   ├── run_fusion.py                   # [Main] Full evaluation pipeline entry point
│   └── fusion/                         # Fusion algorithm modules
│       ├── __init__.py                 # Package init
│       ├── utils.py                    # Data loading / SP3 / M1 inference / Platt calibration / coordinate transforms
│       ├── baselines.py                # 10 WLS/LS baseline positioning methods
│       ├── factor_graph_fusion.py      # MoG factor graph + L-BFGS-B + debiased initialization
│       ├── evaluate_fusion.py          # 22-method evaluation framework + report generation
│       ├── prnc.py                     # [v5] PRNC pseudorange correction (3 variants)
│       ├── los_anchored_ls.py          # [v6] 5 LOS-anchored positioning methods
│       ├── diagnose_weighting.py       # [v4] Weighting failure diagnosis (DOP / Clock / Bias)
│       ├── verify_clock_contamination.py # [v6] Clock contamination hypothesis testing
│       ├── verify_nlos_sign.py         # [v5] NLOS error sign distribution analysis
│       ├── debug_geometry.py           # [P0] Jacobian sign verification + SP3 clock decision
│       ├── motion_geometry_predictor.py # [2A] TCN temporal geometry predictor architecture
│       └── train_tcn.py                # [2A] TCN model training script
├── cache/                              # Inference caches + diagnostics
│   ├── {dataset}_mog_outputs_exp0XX.pkl # Module 1 MoG inference results (rebuilt per experiment)
│   ├── clock_contamination_analysis.json
│   ├── diagnosis_v4.json
│   └── nlos_sign_analysis.json
├── models/                             # Pre-trained TCN models (.pth)
├── result/                             # Experiment results
│   ├── exp_001-004/                    # v1-v2 early experiments
│   ├── exp_010/                        # v4 diagnostic evaluation
│   ├── exp_011/                        # v5 (PRNC)
│   ├── exp_012/                        # v6 16-method full evaluation
│   ├── exp_013/                        # v7 quick check
│   ├── exp_014/                        # v7 full evaluation
│   ├── exp_015/                        # v8 final evaluation (CURRENT)
│   └── exp_v3_*/ exp_v4*/              # Legacy intermediate experiments
├── project/                            # Reference projects (DO NOT MODIFY)
│   ├── g2o/                            # g2o graph optimization library
│   └── gtsam/                          # GTSAM factor graph library
└── file/                               # Documentation
    ├── 参考项目.md                       # Reference project notes
    └── goal/                           # Versioned goals, results, and change logs
        ├── goal_v1.md through goal_v8.md
        ├── result_v1.md through result_v8.md
        └── change_v1.md through change_v8.md
```

---

## Data Flow

```
Module 1 (part1_GAT)
  │  GAT MoG inference → p_los, mu_nlos, sigma_los, sigma_nlos
  │  Platt calibration → p_cal = sigmoid(A * logit(p_raw) + B)
  ▼
Module 2 (this module)
  │
  ├── debug_geometry.py              → [P0] Verify Jacobian sign, SP3 clock decision
  ├── verify_nlos_sign.py            → [v5] NLOS error sign distribution, mu_nlos quality
  ├── verify_clock_contamination.py  → [v6] Clock contamination hypothesis testing
  ├── diagnose_weighting.py          → [v4] Weighting failure root cause analysis (A/B/C/D)
  │
  ├── baselines.py                   → [Base] 10 WLS/LS methods
  ├── prnc.py                        → [v5] PRNC correction (basic / mu / adaptive)
  ├── los_anchored_ls.py             → [v6] LOS-anchored positioning methods
  ├── factor_graph_fusion.py         → [v3] MoG factor graph + L-BFGS-B optimization
  │
  ├── train_tcn.py                   → [2A] Train TCN temporal prior predictor
  ├── motion_geometry_predictor.py   → [2A] TCN architecture
  │
  └── evaluate_fusion.py             → [Eval] 22-method full comparison → metrics.json + comparison_report.md
```

---

## Core Files: Detailed Description

### run_fusion.py — Main Entry Point

Runs the full evaluation pipeline on all 4 datasets: load data → Module 1 MoG inference → 22-method positioning → save results.

**Dataset → Module 1 Experiment Mapping (v8)**:

| Dataset | Experiment | Model Status |
|---------|-----------|-------------|
| berlin1_potsdamer_platz | exp_048 | v8: mu dir OK, F1=0.854 |
| berlin2_gendarmenmarkt | exp_049 | v8: mu dir OK, F1=0.892 |
| frankfurt1_maintower | exp_050 | v8: mu dir OK, F1=0.843 |
| frankfurt2_westendtower | exp_051 | v8: mu dir OK, F1=0.906 |

Auto-increments experiment ID (exp_001, exp_002, …) based on existing `result/` directories.

---

### fusion/utils.py — Utility Functions

| Function | Purpose |
|----------|---------|
| `lla_to_ecef()` / `ecef_to_lla()` | WGS84 coordinate conversion (iterative 5-pass refinement) |
| `load_epoch_data()` | Load preprocessed pickle from `data/processedData/` |
| `compute_satellite_positions()` | SP3 precise ephemeris → satellite ECEF position, with LRU cache |
| `load_mog_model()` | Load Module 1 MoG model from `part1_GAT/result/{exp_name}/best_model.pth` |
| `run_mog_inference()` | Single-epoch MoG inference (11-dim features, Platt calibration) |
| `fit_platt_scaling()` | Platt scaling: grid search + Nelder-Mead optimize BCE on p_los logits |
| `apply_platt_scaling()` | Apply: p_cal = sigmoid(A * logit(p_raw) + B) |

**Feature extraction** (11 dimensions): elevation, azimuth, CNO, prStdev, GNSS one-hot (GPS=7, Glonass=8, Galileo=9, BeiDou=10), prError, pseudorange residual, cycle slip flag, half-cycle slip flag.

MoG inference results cached at `cache/{dataset}_mog_outputs.pkl` and rebuilt automatically when the Module 1 experiment mapping changes.

---

### fusion/baselines.py — 10 Baseline Methods

All methods use iterative Gauss-Newton WLS with Jacobian H[:,:3] = -LOS, H[:,3] = +1.0 (verified in debug_geometry.py).

| Method | Weight Scheme | Uses M1 | Description |
|--------|---------------|:------:|-------------|
| `solve_standard_ls()` | w = 1 | — | Standard equal-weight LS, primary baseline |
| `solve_wls_elevation()` | sin²(elevation) | — | Traditional elevation weighting |
| `solve_wls_mog()` | p_los / sigma_los² | p, σ | Module 1 uncertainty weighting |
| `solve_hard_threshold()` | p_los ≥ 0.5 mask | p | Hard threshold exclusion (fallback to all if <4 sats) |
| `solve_wls_aggressive_power()` | p_los³ / sigma_los | p, σ | Aggressive NLOS suppression (power-3) |
| `solve_wls_log_odds()` | max(0.01, log(p/(1-p))) / sigma_los | p, σ | Log-odds weight transformation |
| `solve_wls_soft_floor()` | max(0.05, p_los) / sigma_los | p, σ | Soft floor prevents near-zero weights |
| `solve_wls_geometry_aware()` | PDOP-aware selective downweighting | p, σ | Only downweight non-critical NLOS sats |
| `solve_wls_debiased()` | pr_corrected = pr - p_nlos * mu_nlos, then WLS with p_los/sigma_los² | All | Debiased pseudorange + uncertainty weighting |
| `solve_raim_mog()` | RAIM consistency check → exclude outlier NLOS sats | p, σ_NLOS | RAIM-based NLOS exclusion |

---

### fusion/prnc.py — [v5] PRNC Pseudorange Correction

**Core idea**: Keep ALL satellites at uniform weight (preserve DOP), estimate and subtract NLOS bias from residuals directly.

| Method | Description |
|--------|-------------|
| `PRNCPositioner.solve_basic()` | Iterative residual correction, gate: p_los < 0.6 & residual > 2×sigma |
| `PRNCPositioner.solve_with_mu()` | Direct correction: pr_corrected = pr - p_nlos * mu_nlos |
| `AdaptivePRNCPositioner.solve_adaptively()` | CNO-adaptive noise floor + two-stage gating |
| `PRNCWithTCN.solve_with_tcn()` | Adaptive PRNC + TCN temporal prior blending |

---

### fusion/los_anchored_ls.py — [v6] LOS-Anchored Positioning

**Core idea**: Use high-confidence LOS satellites (p_los > 0.7) for clock estimation, decoupling clock from NLOS contamination.

| Method | Description |
|--------|-------------|
| `estimate_clock_los_anchored()` | Median of residuals from p_los > 0.7 sats; fallback to top-N if <4 |
| `run_standard_ls()` | Standard Gauss-Newton LS (position + clock, iterative) |
| `solve_los_anchored_ls()` | Standard LS with LOS-only clock estimate |
| `solve_los_anchored_wls_mog()` | WLS-MoG with LOS-only clock estimate |
| `solve_los_anchored_prnc()` | PRNC-mu correction with LOS-only clock estimate |
| `solve_los_anchored_debiased_wls()` | Debiased WLS with LOS-only clock estimate |
| `select_satellites_geometry_aware()` | Satellite selection guaranteeing PDOP ≤ 1.2× baseline |
| `solve_los_anchored_combined()` | Geometry-aware selection + LOS clock + debiased + WLS |

**v6 Conclusion**: Hypothesis REJECTED. LOS-Anchored-LS = Standard LS (0.0% difference in all 4 datasets). Iterative LS self-corrects clock bias during convergence.

---

### fusion/factor_graph_fusion.py — [v3] MoG Factor Graph

Implements a Mixture-of-Gaussians factor graph for soft-information fusion:

| Component | Description |
|-----------|-------------|
| `MoGObservationModel` | Per-satellite MoG likelihood: log p(r) = logsumexp([log(p_los) + log N(r|0, σ_los²), log(1-p_los) + log N(r|μ_nlos, σ_nlos²)]) |
| `FactorGraphPositioner` | Positions using MoG negative log-likelihood via L-BFGS-B |
| `FactorGraphPositioner.solve_standard()` | Standard FG with MoG observation model |
| `FactorGraphPositioner.solve_with_debiasing()` | Pr_corrected = pr - (1-p_los) * mu_nlos, then FG |
| `FactorGraphPositioner.solve_with_tcn_prior()` | FG + TCN temporal prior blending (FG-MoG+2A) |

Smooth approximations used throughout (smooth max/min/clip) to ensure gradient continuity for L-BFGS-B.

---

### fusion/evaluate_fusion.py — 22-Method Evaluation

| # | Method | Uses M1 | Category |
|:--:|--------|:------:|----------|
| 1 | Standard LS | — | Baseline |
| 2 | WLS-elevation | — | Baseline |
| 3 | WLS-MoG-linear | p, σ | v3 |
| 4 | WLS-MoG-power3 | p, σ | v4 |
| 5 | WLS-log-odds | p, σ | v4 |
| 6 | WLS-debiased | p, σ, μ | v4 |
| 7 | RAIM-MoG | p, σ_NLOS | v4 |
| 8 | FactorGraph-MoG | p, μ, σ_LOS, σ_NLOS | v3 |
| 9 | FactorGraph-debiased | All | v4 |
| 10 | FactorGraph-MoG+2A | All + TCN prior | 2A |
| 11 | Hard-threshold | p | v3 |
| 12 | PRNC-basic | p, σ | v5 |
| 13 | PRNC-mu | p, μ | v5 |
| 14 | PRNC-adaptive | p, σ | v5 |
| 15 | PRNC-mu-adaptive | p, σ, μ | v5 |
| 16 | PRNC-mu-corrected | p, μ | v7 |
| 17 | LOS-Anchored-LS | p | v6 |
| 18 | LOS-Anchored-WLS-MoG | p, σ | v6 |
| 19 | LOS-Anchored-PRNC | p, σ, μ | v6 |
| 20 | LOS-Anchored-Debiased-WLS | p, σ, μ | v6 |
| 21 | LOS-Anchored-Combined | All | v6 |
| 22 | Debiased-WLS-v2 | p, μ | v7 |
| 23 | Geometry-Aware-Debiased-WLS | p, μ | v7 |

**Metrics**: CEP50, CEP95, Mean 2D, RMSE 3D, %<5m, %<10m, %<20m, %<50m, %<100m.

Platt scaling calibration applied to all p_los values before evaluation.

---

### fusion/motion_geometry_predictor.py + train_tcn.py — [2A] TCN Temporal Prior

A Temporal Convolutional Network (TCN) that predicts next-epoch NLOS prior probabilities from historical trajectory and satellite geometry sequences. Uses dilated causal 1D convolutions with residual connections.

**Config**: MAX_SV=20, SEQ_LEN=10, HIDDEN_DIM=64, BATCH_SIZE=128, EPOCHS=50, LR=1e-3, EARLY_STOP=10.

The TCN prior is blended with Module 1 p_los via Bayesian fusion in the factor graph (FG-MoG+2A method).

---

### Diagnostic & Verification Scripts

| Script | Purpose | Version |
|--------|---------|:------:|
| `debug_geometry.py` | Jacobian sign verification, SP3 clock correction decision, single-epoch sanity checks | P0 |
| `diagnose_weighting.py` | Root cause analysis of WLS failure: DOP inflation, clock coupling, bias miscalibration | v4 |
| `verify_clock_contamination.py` | Tests whether clock estimate absorbs NLOS positive bias, making residuals symmetric | v6 |
| `verify_nlos_sign.py` | NLOS error sign distribution analysis, p_los binning, mu_nlos quality diagnostics | v5 |

---

## Version History Summary

### v8 (2026-06-05) — Pure Pairwise Ranking mu_nlos Direction Fix (CURRENT)

**Problem**: v7's MuDirectionLoss LOS suppression (2.0×) caused mu_nlos magnitude collapse (181–223 m), ruining debiasing effectiveness despite correct direction.

**Fix**: Replaced suppression-based loss with pure pairwise ranking:
- Removed LOS suppression term entirely
- Kept mean ordering (mu_NLOS > mu_LOS + 0.15 km) + pairwise sampling (0.10 km margin per pair)
- Restored LAMBDA_MU_REG from 0.05 → 0.20, MU_NLOS_TARGET from 0.50 → 0.30 km

**mu_nlos Direction — v5 vs v7 vs v8**:

| Dataset | v5 Dir | v7 Dir | v8 Dir | v8 mu_LOS | v8 mu_NLOS | Margin |
|---------|:------:|:------:|:------:|:---------:|:----------:|:------:|
| berlin1 | WRONG (-22m) | OK (+196m, collapsed) | **OK** | 191m | 308m | +117m |
| berlin2 | WRONG (-136m) | OK (+147m, collapsed) | **OK** | 73m | 216m | +143m |
| frankfurt1 | WRONG (-70m) | OK (+173m, collapsed) | **OK** | 117m | 237m | +121m |
| frankfurt2 | WRONG (-191m) | OK (+158m, collapsed) | **OK** | 141m | 260m | +119m |

**Positioning — v6 vs v7 vs v8 CEP50 (m)**:

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:------:|:------:|:----------:|:----------:|
| Standard LS | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-MoG (v6) | 965.3 | 771.6 | 459.4 | 508.5 |
| WLS-MoG (v7) | 940.5 | 800.2 | 623.6 | 510.4 |
| **WLS-MoG (v8)** | 964.7 | 721.4 | **487.2 (+7.2%)** | 515.2 |
| FG-MoG+2A (v6) | 959.6 | 778.8 | **445.7 (+15.1%)** | 494.5 |
| FG-MoG+2A (v7) | 946.4 | 784.9 | 582.8 | 497.7 |
| **FG-MoG+2A (v8)** | 981.6 | 760.4 | **476.9 (+9.2%)** | 500.1 |

---

### v7 (2026-06-04) — MuDirectionLoss Direction Fix (Partially Successful)

Fixed mu_nlos direction in all 4 datasets (replaced v5's SupervisedMuRegression which produced wrong direction). Removed LOS suppression penalty. **But**: mu_nlos magnitude collapsed to 181–223 m, causing positioning regression vs v6. FG-MoG+2A frankfurt1 degraded from +15.1% to -11.0%.

### v6 (2026-06-04) — LOS-Anchored Clock Fix (Hypothesis Rejected)

16-method evaluation. Hypothesis: clock estimate absorbs NLOS positive bias → LOS-anchored clock should fix. Result: **Rejected** — LOS-Anchored-LS = Standard LS (0.0% difference). Iterative LS self-corrects.

### v5 (2026-06-04) — PRNC Pseudorange Correction

Shifted from WLS weighting to PRNC: uniform weights + residual-based NLOS correction. All success criteria FAILED.

### v4 (2026-06-03) — WLS Diagnosis + 6 New Methods

Root cause diagnosis: DOP inflation, clock coupling, mu_nlos miscalibration. ALL methods failed to beat Standard LS on all 4 datasets.

### v1-v3 — Initial GAT + WLS Baseline

Established the basic pipeline. Frankfurt1 was the only dataset where WLS-MoG beat Standard LS.

---

## Key Scientific Findings

1. **Soft information works, but only with favorable geometry**: Frankfurt1 proves MoG-based positioning can beat Standard LS when NLOS satellites are geometrically redundant. The other 3 datasets fail because downweighting NLOS degrades DOP more than debiasing helps.
2. **mu_nlos direction is learnable**: v8 demonstrates that a pure pairwise ranking loss reliably produces mu_NLOS > mu_LOS in all 4 urban scenarios, with healthy magnitude (216–308 m).
3. **DOP inflation is the fundamental bottleneck**: Weight-based methods cannot overcome the geometry penalty in Berlin and Frankfurt-2. The invariant across v1-v8 is that NLOS satellites that need downweighting are often geometrically essential.
4. **Clock estimation is not the bottleneck**: v6 conclusively proved that clock contamination is self-corrected by iterative LS. Clock-based approaches are unnecessary.
5. **Module 1 has reached practical ceiling**: Classification (F1 0.84–0.91) and mu direction (all correct) are sufficient for downstream use. Further improvements require Module 3 residual feedback.
6. **Block-diagonal batching is safe**: 32-graph batching gives 2.7× speedup with zero classification degradation (<0.01% F1 difference).

---

## Environment

| Item | Value |
|------|-------|
| Python | 3.9+ (conda: smartLoc) |
| PyTorch | CUDA (RTX 5060 Laptop GPU, 8 GB) |
| SciPy | L-BFGS-B, Nelder-Mead, least-squares |
| NumPy | 1.x |
| Module 1 | part1_GAT/model/GAT_V2025.py, config.py, sp3_reader.py |

---

## Relevant Documents

- [goal_v8.md](file/goal/goal_v8.md) — v8 objective: pure pairwise ranking mu fix
- [result_v8.md](file/goal/result_v8.md) — v8 evaluation results (direction OK, positioning frankfurt1 +9.2%)
- [change_v8.md](file/goal/change_v8.md) — v8 code change log
- [goal_v7.md](file/goal/goal_v7.md) — v7 objective: fix mu_nlos direction
- [result_v7.md](file/goal/result_v7.md) — v7 results: direction fixed, magnitude collapsed
- [goal_v6.md](file/goal/goal_v6.md) — v6 objective: LOS-anchored clock
- [goal_v5.md](file/goal/goal_v5.md) — v5 objective: PRNC pseudorange correction
- [goal_v4.md](file/goal/goal_v4.md) — v4 objective: WLS diagnosis + 6 new methods
- [goal_v3.md](file/goal/goal_v3.md) — v3 objective: TCN enhancement + Frankfurt P0
- [Module 1 Documentation](../part1_GAT/model/README.md)
- [Module 3 Documentation](../../part3_ResidualFeedbackAndOnline_Correction/model/README.md)
- [Main Project README](../../../README.md)

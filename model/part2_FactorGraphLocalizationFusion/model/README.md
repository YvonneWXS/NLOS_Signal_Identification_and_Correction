# Module 2: Factor Graph Localization Fusion

> Urban GNSS NLOS Signal Identification & Correction
> Module 2: WLS / Factor Graph / PRNC / LOS-Anchored positioning using Module 1 GAT outputs (p_los, mu_nlos, sigma_los, sigma_nlos)

**Current version: v7 (2026-06-05)** -- mu_nlos direction fix (direction CORRECT, positioning regressed), 20-method evaluation -- LOS-anchored clock fix (hypothesis rejected), 16-method evaluation

---

## Quick Start

```batch
conda activate smartLoc
cd /d D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion
python model\run_fusion.py
```

Estimated time: ~15 min for 16-method evaluation on 4 datasets (RTX 5060 Laptop GPU, Module 1 inference already cached)

---

## Directory Structure

```
part2_FactorGraphLocalizationFusion/
+-- model/                              # Core code (this directory)
|   +-- run_fusion.py                   # [Main] Full evaluation pipeline
|   +-- fusion/                         # Fusion algorithm modules
|   |   +-- utils.py                    # Data loading / SP3 / M1 inference / Platt
|   |   +-- baselines.py                # 10 WLS/LS baseline methods
|   |   +-- factor_graph_fusion.py      # MoG factor graph + L-BFGS-B + debiased init
|   |   +-- evaluate_fusion.py          # 16-method evaluation framework
|   |   +-- los_anchored_ls.py          # [v6] 5 LOS-anchored positioning methods
|   |   +-- verify_clock_contamination.py # [v6] Clock contamination verification
|   |   +-- diagnose_weighting.py       # [v4] Weighting failure diagnosis (A/B/C/D)
|   |   +-- verify_nlos_sign.py         # [v5] NLOS error sign distribution analysis
|   |   +-- prnc.py                     # [v5] PRNC pseudorange correction algorithm
|   |   +-- motion_geometry_predictor.py # TCN temporal geometry predictor
|   |   +-- train_tcn.py               # TCN model training
+-- cache/                              # Inference caches
|   +-- {dataset}_mog_outputs.pkl       # Module 1 MoG inference results
|   +-- clock_contamination_analysis.json # [v6] Clock contamination diagnostic
|   +-- diagnosis_v4.json              # v4 weight diagnosis results
|   +-- nlos_sign_analysis.json        # [v5] NLOS error sign analysis
+-- models/                             # Pre-trained TCN models (.pth)
+-- result/                             # Experiment results
|   +-- exp_001-004/                    # v1-v3 early experiments
|   +-- exp_012/                        # [v6] 16-method full evaluation
|   +-- exp_v5/                         # [v5] 12-method PRNC evaluation
|   +-- exp_v4/                         # v4 9-method full evaluation
|   +-- exp_v3_full/                    # v3 6-method full evaluation
+-- file/                               # Documentation & goals
|   +-- goal/                           # goal_v3/v4/v5/v6.md + result + change
+-- run_v5_full_pipeline.bat           # [v5] One-click training + evaluation
```

---

## Data Flow

```
Module 1 (part1_GAT)
  |  GAT inference: p_los, mu_nlos, sigma_los, sigma_nlos
  |  PLATT calibration: p_cal = sigmoid(A*logit(p_raw) + B)
  v
Module 2 (this module)
  +-- verify_nlos_sign.py            -> [v5] NLOS sign / p_los binning / mu_nlos quality
  +-- verify_clock_contamination.py  -> [v6] Clock contamination hypothesis testing
  +-- baselines.py                   -> [Baseline] 10 WLS/LS methods
  +-- prnc.py                        -> [v5 Core] PRNC correction (3 variants)
  +-- los_anchored_ls.py             -> [v6] 5 LOS-anchored + geometry-aware methods
  +-- factor_graph_fusion.py         -> [Optional] MoG factor graph + L-BFGS-B
  +-- train_tcn.py                   -> [Optional] TCN temporal prior training
  +-- evaluate_fusion.py             -> [Evaluation] 16-method full comparison
```

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

### fusion/los_anchored_ls.py -- [v6] LOS-Anchored Positioning Methods

**Core idea**: Use only high-confidence LOS satellites (p_los > 0.7) for clock estimation, decoupling clock from NLOS contamination.

| Method | Description |
|---------|------|
| estimate_clock_los_anchored() | Clock estimation using only confident LOS sats; fallback to sigma-clip if <4 LOS sats |
| run_standard_ls() | Standard Gauss-Newton LS (position + clock) |
| solve_los_anchored_ls() | Method 1: Standard LS with LOS-only clock estimate |
| solve_los_anchored_wls_mog() | Method 2: WLS-MoG with LOS-only clock estimate |
| solve_los_anchored_prnc() | Method 3: PRNC correction with LOS-only clock estimate |
| solve_los_anchored_debiased_wls() | Method 4: Debiased WLS with LOS-only clock estimate |
| select_satellites_geometry_aware() | Geometry-aware satellite selection (guarantees PDOP <= 1.2x baseline) |
| solve_los_anchored_combined() | Method 5: Geometry-aware selection + LOS clock + debiased + WLS |

### fusion/verify_clock_contamination.py -- [v6] Clock Contamination Diagnosis

Validates the v6 hypothesis: whether clock estimate absorbs NLOS positive bias, making residuals appear symmetric.

**Analysis steps**:
1. Compare contaminated clock (median of ALL satellites) vs LOS-anchored clock (median of high-p_los sats only)
2. Measure delta_clk = clk_los - clk_contaminated
3. Analyze NLOS residuals under both clock estimates
4. Determine if clock contamination is the root cause of WLS failure

**v6 Result**: Hypothesis REJECTED. LOS-Anchored-LS = Standard LS (0.0% difference in all 4 datasets). Iterative LS self-corrects any clock bias during convergence.

---

### fusion/evaluate_fusion.py -- 20-Method Evaluation (v7)

| # | Method | Uses M1 Outputs | Description |
|:---:|------|:---:|------|
| 1 | Standard LS | None | Baseline |
| 2 | WLS-elevation | None | Elevation weighting |
| 3 | WLS-MoG-linear | p_los, sigma | Direct M1 weighting |
| 4 | WLS-power3 | p_los, sigma | p_los^3 weighted |
| 5 | WLS-log-odds | p_los, sigma | Log-odds weighting |
| 6 | WLS-debiased | p_los, sigma, mu | Pseudorange debiasing |
| 7 | RAIM-MoG | p_los, sigma_nlos | RAIM exclusion |
| 8 | FG-debiased | All | Factor graph + debiasing |
| 9 | FG-debiased+2A | All + TCN | Factor graph + temporal prior |
| 10 | PRNC-basic | p_los, sigma | [v5] Basic residual correction |
| 11 | PRNC-mu | p_los, mu | [v5] mu_nlos direct correction |
| 12 | **LOS-Anchored-LS** | p_los | [v6] Standard LS + LOS-only clock |
| 13 | **LOS-Anchored-WLS-MoG** | p_los, sigma | [v6] WLS-MoG + LOS-only clock |
| 14 | **LOS-Anchored-PRNC** | p_los, sigma, mu | [v6] PRNC + LOS-only clock |
| 15 | **LOS-Anchored-Debiased-WLS** | p_los, sigma, mu | [v6] Debiased WLS + LOS-only clock |
| 16 | **LOS-Anchored-Combined** | All | [v6] Geometry-aware + LOS clock + debiased + WLS |

Metrics: CEP50, CEP95, Mean 2D, RMSE 3D

---


---

## v7 Core Conclusions (2026-06-05)

**mu_nlos direction FIXED in all 4 datasets. Positioning REGRESSED due to mu_nlos magnitude collapse.**

### mu_nlos Direction: v5 vs v7

| Dataset | v5 Direction | v7 Direction | v7 Margin |
|--------|:---:|:---:|:---:|
| berlin1 | WRONG (-22m) | **OK** | +196m |
| berlin2 | WRONG (-136m) | **OK** | +147m |
| frankfurt1 | WRONG (-70m) | **OK** | +173m |
| frankfurt2 | WRONG (-191m) | **OK** | +158m |

### CEP50 Comparison (m) - v6 vs v7 (Key Methods)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| **Standard LS** | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-MoG (v7) | 940.5 | 800.2 | 623.6 | 510.4 |
| WLS-debiased (v7) | 979.5 | 863.1 | 628.6 | 547.5 |
| Debiased-WLS-v2 (v7) | 987.5 | 863.3 | 606.9 | 556.3 |
| PRNC-mu-corrected (v7) | 965.8 | 740.4 | 580.1 | 477.9 |
| **FG-MoG+2A (v6 best)** | 959.6 | 778.8 | **445.7 (+15.1%)** | 494.5 |
| FG-MoG+2A (v7) | 946.4 | 784.9 | 582.8 (-11.0%) | 497.7 |

### Why v7 Regressed

1. **mu_nlos magnitude collapsed**: LAMBDA_MU_REG weakened from 0.30 to 0.05 + aggressive MuDirectionLoss weights (2.0x + 3.0x) caused all mu values to shrink. v7 mu_NLOS = 181-223m vs v5 mu_NLOS = 162-395m (but v5 had wrong direction).
2. **Direction fix is necessary but NOT sufficient**: Correct direction alone doesn't improve positioning when magnitude is too small for meaningful correction.
3. **Frankfurt1 regression is the clearest signal**: v6 FG-MoG+2A achieved +15.1% (445.7m) on frankfurt1. v7 degraded to -11.0% (582.8m). The wrong-direction v6 model had better sigma calibration by coincidence.

### Scientific Contribution

The v7 experiment proves:
1. MuDirectionLoss can reliably fix mu_nlos direction in all datasets
2. Direction correction alone does not improve downstream positioning
3. mu_nlos magnitude and direction are coupled through the loss function
4. A pure ranking loss (without magnitude suppression) is needed for v8

## v6 Core Conclusions (2026-06-05)

**Hypothesis REJECTED: Clock contamination is NOT the root cause of WLS failure.**

### v6 16-Method CEP50 Results (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| **Standard LS** | **904.5** | **610.8** | **525.2** | **382.6** |
| LOS-Anchored-LS | 904.5 (0.0%) | 610.8 (0.0%) | 525.2 (0.0%) | 382.6 (0.0%) |
| LOS-Anchored-WLS-MoG | 966.8 (-6.9%) | 774.8 (-26.8%) | 474.7 (+9.6%) | 510.2 (-33.4%) |
| LOS-Anchored-Debiased-WLS | 1028.5 (-13.7%) | 858.7 (-40.6%) | 466.5 (+11.2%) | 558.1 (-45.9%) |
| WLS-MoG (original) | 965.3 (-6.7%) | 771.6 (-26.3%) | 459.4 (+12.5%) | 508.5 (-32.9%) |
| **FG-MoG+2A** | 959.6 (-6.1%) | 778.8 (-27.5%) | **445.7 (+15.1%)** | 494.5 (-29.2%) |

### Why v6 Failed

1. **Iterative LS self-corrects**: Gauss-Newton adjusts position AND clock simultaneously. Any initial clock bias is corrected during convergence regardless of which satellites were used for the initial estimate.
2. **Clock is a free parameter**: In Hx = pr - clk, the clock term is solved jointly with position. "Contamination" is absorbed into the clock parameter without affecting position.

### The Real Bottleneck (Synthesized from v1-v6)

| Version | Hypothesis | Result |
|--------|-----------|--------|
| v3 | WLS weighting with p_los/sigma | Works only in frankfurt1 (NLOS sats redundant) |
| v4 | 6 new WLS schemes | All fail: DOP inflation + clock coupling |
| v5 | PRNC pseudorange correction | Fails: NLOS errors symmetric after clock absorption |
| v6 | LOS-anchored clock | Fails: iterative LS self-corrects clock |

**The invariant**: mu_nlos is directionally wrong (mu_LOS > mu_NLOS in all datasets). Until Module 1 learns to output higher mu for NLOS than LOS, no Module 2 method using mu_nlos can work reliably.

**Best method**: FactorGraph-MoG+2A on frankfurt1 (+15.1% CEP50 vs Standard LS). Works because:
1. Frankfurt1 has favorable geometry where NLOS sats can be deweighted without DOP penalty
2. Factor graph + TCN temporal prior provides better smoothing
3. Does NOT rely on mu_nlos for correction (uses p_los/sigma weighting only)

---

## v5 PRNC Approach (2026-06-04)

**Core shift**: Abandon WLS weighting, implement PRNC (Pseudorange Residual NLOS Correction). Keep all satellites at uniform weight (preserve DOP), estimate and subtract NLOS bias from residuals directly.

### v5 Success Criteria (All FAILED)

| ID | Criterion | Threshold | Result |
|:---:|------|:---:|:---:|
| C1 | PRNC-adaptive beats Standard LS in >= 2/4 datasets | > 3% CEP50 | FAIL |
| C2 | PRNC does NOT degrade DOP vs Standard LS | PDOP diff < 0.01 | PASS |
| C3 | mu_nlos MAE < 0.3 km after Module 1 retraining | -- | FAIL (0.05-0.15 vs actual 0.5-1.5 km) |
| C4 | PRNC-adaptive beats WLS-MoG-linear in ALL 4 datasets | -- | FAIL |
| C5 | PRNC+2A does NOT degrade vs PRNC-adaptive in any dataset | -- | PASS |
| C6 | NLOS correction precision > 70% | -- | FAIL |

---

## v4 Core Conclusions (2026-06-03)

**ALL WLS and factor graph methods FAILED to beat Standard LS on all 4 datasets.**

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

## Next Steps (v7)

1. **Fix mu_nlos direction in Module 1**: Add explicit loss penalty for mu_nlos_LOS > mu_nlos_NLOS. This is the invariant bottleneck across v1-v6.
2. **Geometry-aware WLS**: Only apply weights when DOP impact is acceptable (generalize frankfurt1 success).
3. **Abandon clock-based approaches**: v6 conclusively proves clock estimation is not the bottleneck.
4. **Consider end-to-end training**: Jointly optimize Module 1 for downstream positioning, not just classification.

---

## Environment

- Python 3.9+ (conda: smartLoc)
- PyTorch CUDA (RTX 5060 Laptop GPU)
- SciPy (L-BFGS-B, Nelder-Mead, approx_fprime)
- NumPy
- Module 1 (part1_GAT): GAT_V2025.py, config.py, sp3_reader.py

---

## Related Documents

- [goal_v7.md](file/goal/goal_v7.md) -- v7 objective: fix mu_nlos direction
- [result_v7.md](file/goal/result_v7.md) -- v7 evaluation results + regression analysis
- [change_v7.md](file/goal/change_v7.md) -- v7 code change log
- [goal_v6.md](file/goal/goal_v6.md) -- v6 objective: LOS-anchored clock contamination fix
- [result_v6.md](file/goal/result_v6.md) -- v6 evaluation results + hypothesis rejection
- [change_v6.md](file/goal/change_v6.md) -- v6 code change log
- [result_v6.md](file/goal/result_v6.md) -- v6 evaluation results + hypothesis rejection
- [change_v6.md](file/goal/change_v6.md) -- v6 code change log
- [goal_v5.md](file/goal/goal_v5.md) -- v5 objective: PRNC pseudorange correction
- [result_v4.md](file/goal/result_v4.md) -- v4 evaluation results + diagnosis
- [change_v4.md](file/goal/change_v4.md) -- v4 code change log
- [goal_v4.md](file/goal/goal_v4.md) -- v4 objective: WLS diagnosis + 6 new methods
- [goal_v3.md](file/goal/goal_v3.md) -- v3 TCN enhancement + Frankfurt P0
- [result_v3.md](file/goal/result_v3.md) -- v3 6-method evaluation
- [Module 1 Documentation](../part1_GAT/model/README.md)

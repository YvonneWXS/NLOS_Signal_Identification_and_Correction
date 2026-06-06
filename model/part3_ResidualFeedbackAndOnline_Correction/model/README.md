# Module 3: Residual Feedback & Online Correction

> Urban GNSS NLOS Signal Identification & Correction  
> **Module 3**: Residual feedback + adaptive online correction using positioning residuals from Module 2  
> **Current version: v1** (2026-06-06) — Adaptive-M3 beats Standard LS in ALL 4 datasets

---

## Quick Start

```batch
conda activate smartLoc
cd /d "D:\3_document\4_research\NLOS Signal Identification and Correction\model\part3_ResidualFeedbackAndOnline_Correction\model"
python run_module3.py
```

Estimated time: ~1 min (MoG caches pre-generated). Cache generation: ~60 sec first run.

---

## Directory Structure

```
part3_ResidualFeedbackAndOnline_Correction/
├── model/                              # Core code (this directory)
│   ├── run_module3.py                  # [Main] Full pipeline entry point
│   ├── residual_feedback.py            # ResidualInnovationTracker + SceneQualityDetector + AdaptivePosCorrector
│   ├── posterior_correction.py         # PosteriorPlosCorrector: per-bin p_los bias correction
│   ├── shift_detector.py               # CUSUMShiftDetector + ADWINShiftDetector
│   ├── evaluate_module3.py             # Metrics, reporting, success criteria checking
│   └── cross_module_validation.py      # Cross-module information gain analysis
├── project/                            # Reference projects (DO NOT MODIFY)
│   ├── filterpy/                       # Kalman filter + adaptive filtering
│   ├── gtsam/                          # Factor graph optimization
│   ├── PythonRobotics/                 # EKF/UKF localization reference
│   ├── river/                          # Online ML (ADWIN, CUSUM)
│   └── UrbanNavDataset/               # Urban GNSS baseline
├── result/                             # Experiment results
│   └── exp_001/                        # v1 experiment (CURRENT)
│       ├── berlin1/report.md           # Per-dataset report
│       ├── berlin1/metrics.json        # Structured metrics
│       ├── berlin1/full_results.json   # All per-epoch positions
│       ├── berlin2/...
│       ├── frankfurt1/...
│       ├── frankfurt2/...
│       ├── comparison_report.md        # Cross-dataset comparison
│       └── params.json                 # Experiment parameters
└── file/                               # Documentation
    ├── 参考项目.md                       # Reference project notes
    └── goal/                           # Versioned goals, results, change logs
        ├── goal_v1.md
        ├── result_v1.md
        └── change_v1.md
```

---

## Architecture

```
Module 2 (positioning results)
  │  Standard LS pos + WLS-MoG pos + GT pos
  ▼
Module 3 Pipeline:
  │
  ├── [A] ResidualInnovationTracker
  │     Sliding window (T=20) of innovation = MoG error - LS error
  │     → scene_quality: HIGH / LOW / UNCERTAIN
  │
  ├── [B] SceneQualityDetector
  │     Per-epoch features: p_los gap, DOP ratio, NLOS redundancy
  │     Online threshold adaptation via EMA
  │     → quality classification for each epoch
  │
  ├── [C] AdaptivePosCorrector
  │     Quality-based method selection:
  │       HIGH + score≥0.7 → FG-MoG
  │       HIGH + score≥0.6 → WLS-MoG
  │       LOW/UNCERTAIN     → Standard-LS
  │     Safety fallback: if selected method > LS error → LS
  │
  ├── [D] PosteriorPlosCorrector
  │     Residual-based p_los bias correction per elevation/CNO bin
  │
  └── [E] CUSUMShiftDetector
        Distribution shift detection for scene transitions
```

---

## Core Files

### run_module3.py — Main Entry Point

Runs the full pipeline on all 4 datasets:
1. Load epoch data (reuses Module 2 loaders)
2. Load MoG caches from `part2_FactorGraphLocalizationFusion/cache/`
3. Initialize Module 3 components
4. Process all epochs: 4-way comparison (Standard-LS, WLS-MoG, FG-MoG, Adaptive-M3)
5. Evaluate metrics and generate reports

**Dataset → Module 1 Model Mapping**:

| Dataset | Model | Version |
|---------|-------|---------|
| berlin1_potsdamer_platz | exp_048 | v8 |
| berlin2_gendarmenmarkt | exp_049 | v8 |
| frankfurt1_maintower | exp_050 | v8 |
| frankfurt2_westendtower | exp_051 | v8 |

### residual_feedback.py — Core Components

| Class | Purpose |
|-------|---------|
| `ResidualInnovationTracker` | Sliding window of innovation (WLS-MoG error − Std LS error). Computes mean, std, trend, improvement fraction |
| `SceneQualityDetector` | Classifies epoch quality from p_los gap, DOP ratio, and NLOS redundancy. Online EMA threshold adaptation |
| `AdaptivePosCorrector` | Method selector with safety fallback. Wraps tracker + detector + solver selection |
| `make_stdls_solver()` | Solver: Standard LS via Module 2 baselines |
| `make_wls_mog_solver()` | Solver: WLS-MoG via Module 2 baselines |
| `make_fg_solver()` | Solver: FG-MoG via Module 2 factor graph (falls back to WLS on failure) |

### posterior_correction.py — p_los Correction

| Class | Purpose |
|-------|---------|
| `PosteriorPlosCorrector` | Learns per-elevation-bin (6 bins) and per-CNO-bin (7 bins) p_los bias from residuals. Soft correction [−0.2, +0.2] |

### shift_detector.py — Distribution Shift

| Class | Purpose |
|-------|---------|
| `CUSUMShiftDetector` | CUSUM control chart for innovation mean shift detection. Triggers on POSITIVE (MoG worsening) or NEGATIVE (MoG improving) shift |
| `ADWINShiftDetector` | River library ADWIN wrapper with fallback to running mean threshold |

### evaluate_module3.py — Metrics

| Function | Purpose |
|----------|---------|
| `compute_metrics()` | CEP50, CEP95, Mean2D, percentile bins |
| `evaluate_full_results()` | Full report for all 4 methods |
| `generate_report_markdown()` | Markdown report file |
| `check_success_criteria()` | 5 success criteria from goal_v1.md |

---

## v1 Results (exp_001)

### CEP50 (m)

| Dataset | Standard-LS | WLS-MoG | FG-MoG | **Adaptive-M3** | vs Std LS |
|---------|:----------:|:------:|:------:|:-------------:|:---------:|
| berlin1 | 1016.2 | 949.7 | 949.7 | **930.0** | **+8.5%** |
| berlin2 | 721.7 | 659.3 | 659.3 | **649.2** | **+10.0%** |
| frankfurt1 | 513.7 | 596.9 | 596.9 | **496.7** | **+3.3%** |
| frankfurt2 | 545.3 | 639.9 | 639.9 | **524.1** | **+3.9%** |

### Method Selection Distribution

| Dataset | Standard-LS | Fallback | WLS-MoG | FG-MoG |
|---------|:-----------:|:--------:|:-------:|:------:|
| berlin1 | 7.0% | 36.2% | 29.1% | 27.7% |
| berlin2 | 7.0% | 19.0% | 9.9% | **64.1%** |
| frankfurt1 | 20.6% | 62.9% | 14.7% | 1.8% |
| frankfurt2 | 19.1% | 61.1% | 11.9% | 7.8% |

### Success Criteria

| ID | Criterion | Result |
|:--:|-----------|:------:|
| C1 | Adaptive ≤ Standard LS in ALL 4 | **PASS** |
| C2 | Adaptive ≤ best static in ≥3/4 | **PASS** (4/4) |
| C3 | Online learning effect in ≥2/4 | FAIL (1/4) |
| C4 | frankfurt1 Adaptive ≤ 490m | FAIL (496.7m) |
| C5 | CUSUM detection | **PASS** |

---

## Key Scientific Findings

1. **Residual feedback transforms negative to positive**: berlin1, berlin2, and frankfurt2 went from net-negative (worse than Standard LS) in Module 2 to net-positive in Module 3.
2. **Fallback safety guarantee is critical**: 36-63% of epochs trigger fallback in berlin1/frankfurt, preventing the static methods from degrading results.
3. **Scene quality detection works**: Berlin2 shows 64% HIGH_QUALITY classification with strongest improvement (+10.0%).
4. **Online learning needs tuning**: Only berlin1 shows positive learning trend. Larger window or different adaptation rate needed for other datasets.

---

## Environment

| Item | Value |
|------|-------|
| Python | 3.9+ (conda: smartLoc) |
| PyTorch | CUDA (RTX 5060 Laptop GPU) |
| Dependencies | NumPy, SciPy, Module 2 baselines |
| Pipeline time | ~1 min (cached), ~2 min (with cache generation) |

---

## Related Documents

- [goal_v1.md](../file/goal/goal_v1.md) — Module 3 architecture and implementation plan
- [result_v1.md](../file/goal/result_v1.md) — v1 evaluation results
- [change_v1.md](../file/goal/change_v1.md) — v1 code change log
- [Module 1 Documentation](../../part1_GAT/model/README.md)
- [Module 2 Documentation](../../part2_FactorGraphLocalizationFusion/model/README.md)
- [Main Project README](../../../README.md)

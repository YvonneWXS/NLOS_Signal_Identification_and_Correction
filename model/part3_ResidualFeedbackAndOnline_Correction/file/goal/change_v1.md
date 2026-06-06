# Module 3 v1 Change Log

**Date**: 2026-06-06
**Experiment**: exp_001
**Goal**: goal_v1.md — Implement residual feedback + adaptive online correction
**Result**: Adaptive-M3 beats Standard LS in ALL 4 datasets, beats best static in ALL 4 datasets. 3/5 success criteria passed.

---

## New Files Created

| File | Lines | Purpose |
|------|:-----:|---------|
| `model/residual_feedback.py` | ~210 | Core: ResidualInnovationTracker, SceneQualityDetector, AdaptivePosCorrector + solver wrappers |
| `model/posterior_correction.py` | ~110 | PosteriorPlosCorrector: per-bin p_los bias correction from residuals |
| `model/shift_detector.py` | ~110 | CUSUMShiftDetector + ADWINShiftDetector (with fallback) |
| `model/evaluate_module3.py` | ~180 | Metrics computation, report generation, success criteria checking |
| `model/run_module3.py` | ~320 | Full pipeline entry point: load→process→evaluate on all 4 datasets |
| `model/cross_module_validation.py` | ~100 | Cross-module information gain analysis |

---

## Module 2 Caches Generated

| File | Size | Model |
|------|------|-------|
| `part2_FactorGraphLocalizationFusion/cache/berlin1_potsdamer_platz_mog_outputs_exp_048.pkl` | ~1.6 MB | exp_048 (v8) |
| `part2_FactorGraphLocalizationFusion/cache/berlin2_gendarmenmarkt_mog_outputs_exp_049.pkl` | ~1.6 MB | exp_049 (v8) |
| `part2_FactorGraphLocalizationFusion/cache/frankfurt1_maintower_mog_outputs_exp_050.pkl` | ~1.6 MB | exp_050 (v8) |
| `part2_FactorGraphLocalizationFusion/cache/frankfurt2_westendtower_mog_outputs_exp_051.pkl` | ~1.6 MB | exp_051 (v8) |

Generated from Module 1 v8 models (exp_048-051). Each cache contains per-epoch MoG outputs (p_los, sigma_los, mu_nlos, sigma_nlos) for fast reuse.

---

## Bugs Fixed During Development

| Bug | File | Fix |
|-----|------|-----|
| `AdaptivePRNCPositioner` import error | `residual_feedback.py` | Removed unused import (class does not exist in prnc.py) |
| `make_fg_solver(use_tcn=False)` TypeError | `run_module3.py` | Removed `use_tcn` parameter |
| `gt_ecef_km` key not found in epoch dict | `run_module3.py` | Changed to `gt_ecef` (actual key name) |
| `solve_standard_ls()` returns x (4,) not (pos, clk) | `residual_feedback.py` | Added unpacking: `x[:3], x[3]` |

---

## Architecture Decisions

1. **No new Module 1 inference**: All MoG outputs read from pre-generated caches. Total inference time: 64 seconds for cache generation, ~55 seconds for Module 3 pipeline.

2. **Solver wrappers via closures**: `make_stdls_solver()`, `make_wls_mog_solver()`, `make_fg_solver()` return callable functions matching `AdaptivePosCorrector.process_epoch()` interface. Enables easy swapping of positioning backends.

3. **Hard fallback guarantee**: `AdaptivePosCorrector.process_epoch()` always computes Standard LS AND checks if adaptive result is better. If not, falls back to LS — satisfying constraint C1 by construction.

4. **Posterior p_los correction**: `PosteriorPlosCorrector` operates per-elevation-bin and per-CNO-bin with soft bias boundaries of [−0.2, +0.2]. Currently running but not yet demonstrating significant impact on positioning.

5. **CUSUM shift detection**: Running alongside the main pipeline but not yet influencing method selection. Detection thresholds (allowance=20m, threshold=100m) default values from goal document.

---

## Files NOT Modified

- `part1_GAT/model/*` — Module 1 unchanged
- `part2_FactorGraphLocalizationFusion/model/*` — Module 2 unchanged (only caches added)
- `project/*` — Reference projects unchanged

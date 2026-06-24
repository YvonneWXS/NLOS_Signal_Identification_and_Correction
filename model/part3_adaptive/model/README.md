# Module 3: Adaptive Selection (Part3)

## 1. Module Overview

### 1.1 Function
Online adaptive positioning method selection based on residual innovation tracking and scene quality detection. Role in PI-PEM: **Adaptation** — dynamically selects between LS, WLS, or FG per epoch.

### 1.2 Core Flow
`
Per-epoch positioning results -> ResidualInnovationTracker -> SceneQualityDetector -> AdaptivePositionSelector -> best method per epoch
`

### 1.3 Input & Output
- **Input**: Per-epoch positioning results from Module 2 (multiple methods)
- **Output**: Selected method per epoch + aggregated CEP50 metrics

## 2. Architecture

### 2.1 File Structure
`
part3_adaptive/model/
├── run.py                -> CLI entry (was run_module3.py)
├── residual_feedback.py  -> ResidualInnovationTracker: sliding window statistics
├── shift_detector.py     -> SceneQualityDetector: 3-metric scoring
├── run_module3.py        -> AdaptivePositionSelector: threshold-based selection
├── posterior_correction.py -> Posterior correction module
├── evaluate_module3.py   -> Evaluation and metrics
├── cross_module_validation.py -> Cross-module validation
├── diagnosis_v3.py       -> Diagnostic tools
├── reproduce_paper_results.py -> Paper result reproduction
├── run_ablation.py       -> Ablation experiment runner
└── results/              -> Experiment outputs
`

### 2.2 Core Classes
| Class | File | Description |
|-------|------|-------------|
| ResidualInnovationTracker | residual_feedback.py | Tracks residual stats over sliding window |
| SceneQualityDetector | shift_detector.py | Scores scene via p_los gap + PDOP + NLOS redundancy |
| AdaptivePositionSelector | run_module3.py | Selects FG/WLS/LS based on quality score |

### 2.3 Selection Logic
`
scene_score >= 0.65 -> Factor Graph
0.50 <= scene_score < 0.65 -> WLS
scene_score < 0.50 -> Standard LS
Fallback: if selected method > 5% worse than LS, revert to LS
`

## 3. Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| window_size | int | 50 | Sliding window for residual tracking |
| fg_threshold | float | 0.65 | Minimum scene score for FG |
| wls_threshold | float | 0.50 | Minimum scene score for WLS |
| w_plos_gap | float | 0.33 | p_los gap weight |
| w_pdop | float | 0.33 | PDOP ratio weight |
| w_nlos_red | float | 0.34 | NLOS redundancy weight |

## 4. Usage

### 4.1 CLI
`ash
cd part3_adaptive/model
python run.py --dataset berlin1_potsdamer_platz --input ../results/exp_001/
python run.py --dataset all --input ../results/baseline/ --output ../results/adaptive/
`

### 4.2 Output
Results saved to part3_adaptive/results/{exp_name}/:
- daptive_metrics.json — Per-epoch method selections + CEP50
- selection_stats.json — Method selection frequency distribution
- condition.md — Experiment parameters

## 5. API Reference

### run_module3.py
`python
def run_adaptive_selection(dataset_name, positioning_results, config) -> dict
`

### residual_feedback.py
`python
class ResidualInnovationTracker:
    def update(self, residuals) -> float  # innovation score
    def get_statistics() -> dict
`

## 6. Dependencies

- numpy, scipy
- Part2 localization results
- Internal: residual_feedback, shift_detector, posterior_correction

## 7. Tests

`ash
cd part3_adaptive/model
python run.py --dataset berlin1_potsdamer_platz --input ../results/test/
`

## 8. FAQ

**Q: Does adaptive selection always improve over Standard LS?**
A: Not always. On low-NLOS datasets (frankfurt2, 26.6% NLOS), LS alone often suffices.

**Q: What happens if no method meets thresholds?**
A: Falls back to Standard LS as safe default.

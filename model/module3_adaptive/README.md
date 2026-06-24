# Module 3: Adaptive Selection

## 1. Module Overview

### 1.1 Function
Residual innovation tracking + scene quality detection -> adaptive method selection (LS/WLS/FG) with safety fallback to LS.

### 1.2 Core Flow
Per-epoch residuals -> ResidualInnovationTracker (sliding window) + SceneQualityDetector (3-metric weighted) -> AdaptivePositionSelector (threshold-based) -> selected method or LS fallback

### 1.3 Input/Output
- Input: Module 2 outputs (per-epoch positions, residuals), MoG outputs\n- Output: selected_method, corrected_position, scene_quality_score

## 2. Architecture

### 2.1 Internal Structure
`
module3_adaptive/
+-- tracker.py        -> ResidualInnovationTracker (window=50)
+-- detector.py       -> SceneQualityDetector (p_los_gap, pdop_ratio, nlos_redundancy)
+-- selector.py       -> AdaptivePositionSelector (FG>=0.65, WLS>=0.50, else LS)
+-- posterior_correction.py -> Posterior correction module
+-- run.py            -> CLI entry point
`

### 2.2 Core Dependencies
- Consumes module1_nlos and module2_localization outputs\n- Feeds into module4_experiments for evaluation

## 3. Configuration

### 3.1 config.yaml Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| window_size | int | 50 | 20-200 | Sliding window for residual tracking |
| fg_threshold | float | 0.65 | 0.50-0.80 | Score threshold for FG selection |
| wls_threshold | float | 0.50 | 0.40-0.65 | Score threshold for WLS selection |
| fallback.enabled | bool | true | - | Safety fallback to LS if relative error > 5% |

## 4. Usage

### 4.1 Command Line
`ash
# Run adaptive selection on one dataset
python -m model.module3_adaptive.run --dataset berlin1 --input results/module2_outputs/
`

### 4.2 Python API
`python
from module3_adaptive.selector import AdaptivePositionSelector
selector = AdaptivePositionSelector(config)
method, position, details = selector.select(epoch_errors, mog_outputs)
`

## 5. Core API Reference

### ResidualInnovationTracker (tracker.py)
Tracks residual statistics (mean, variance, autocorrelation) in sliding window.

### SceneQualityDetector (detector.py)
Weighted scoring: p_los_gap (33%) + pdop_ratio (33%) + nlos_redundancy (34%).

### AdaptivePositionSelector (selector.py)
Score >= 0.65 -> FG, >= 0.50 -> WLS, < 0.50 -> LS. Safety fallback to LS.

## 6. Dependencies

### 6.1 Internal
- module1_nlos (MoG outputs)\n- module2_localization (positioning methods)

### 6.2 External
- numpy, scipy

## 7. Testing

`ash
pytest model/module3_adaptive/tests/ -v
`

## 8. FAQ

### Q: Does adaptive selection help?
On European datasets, standard LS is already optimal. Adaptive selection cannot improve when all alternatives are worse.

### Q: What triggers FG selection?
High scene quality score (>=0.65): clear LOS/NLOS separation, good DOP, sufficient NLOS redundancy.

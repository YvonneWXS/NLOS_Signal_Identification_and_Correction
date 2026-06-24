# Module 2: Fusion Localization

## 1. Module Overview

### 1.1 Function
13 localization methods (5 existing + 8 new baselines) with unified interface via LocalizationBase ABC and factory pattern.

### 1.2 Core Flow
Pseudorange observations + SV positions + MoG outputs -> LocalizationBase.solve() -> (ECEF position, clock bias, details)

### 1.3 Input/Output
- Input: observations (N,) km, sv_positions (N,3) km, additional_info (p_los, sigma, elevation)\n- Output: position (3,) ECEF km, clock_bias (m), details dict

## 2. Architecture

### 2.1 Internal Structure
`
module2_localization/
+-- base.py           -> LocalizationBase ABC (unified interface)
+-- factory.py        -> LocalizationFactory (registry pattern)
+-- standard_ls.py    -> Standard iterative LS
+-- wls.py            -> WLS-elevation, WLS-MoG
+-- hard_threshold.py -> p_los > threshold LS
+-- factor_graph.py   -> L-BFGS-B MoG optimization
+-- raim.py           -> RAIM residual detection
+-- irls.py           -> Iterative Reweighted LS (Huber)
+-- kalman.py         -> EKF positioning
+-- cno_weighted.py   -> C/N0-weighted LS
+-- snr_weighted.py   -> SNR-weighted LS
+-- dnn.py            -> DNN end-to-end (stub)
+-- gat_e2e.py        -> GAT end-to-end (stub)
+-- ins_gnss.py       -> INS/GNSS coupling (stub)
+-- run.py            -> CLI entry point
`

### 2.2 Core Dependencies
- Uses common/coordinate.py\n- Consumes module1_nlos MoG outputs\n- Feeds into module3_adaptive

## 3. Configuration

### 3.1 config.yaml Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| hard_threshold.threshold | float | 0.5 | 0.3-0.8 | p_los exclusion threshold |
| factor_graph.multistart | int | 3 | 1-10 | Multi-start optimization count |
| factor_graph.max_iter | int | 100 | 50-500 | L-BFGS-B max iterations |
| ekf.process_noise | float | 0.1 | 0.01-1.0 | EKF process noise |

## 4. Usage

### 4.1 Command Line
`ash
# Run all methods on one dataset
python -m model.module2_localization.run --dataset berlin1 --methods all

# Run specific method
python -m model.module2_localization.run --dataset berlin1 --methods standard_ls,wls_elevation
`

### 4.2 Python API
`python
from module2_localization.factory import LocalizationFactory
method = LocalizationFactory.create('standard_ls')
pos, clk, details = method.solve(obs, sv_positions, additional_info={'elevation_deg': elev})
`

## 5. Core API Reference

### LocalizationBase (base.py)
`python
class LocalizationBase(ABC):
    @abstractmethod
    def solve(observations, sv_positions, sv_systems=None, additional_info=None) -> (position, clock_bias, details)
`
### LocalizationFactory (factory.py)
`python
LocalizationFactory.register('method_name')(MethodClass)
LocalizationFactory.create('method_name', config={})
LocalizationFactory.list_methods()  # -> ['standard_ls', ...]
`

## 6. Dependencies

### 6.1 Internal
- common.coordinate\n- module1_nlos (for MoG priors)

### 6.2 External
- numpy, scipy\n- torch (for DNN/GAT stubs)

## 7. Testing

`ash
pytest model/module2_localization/tests/ -v
`

## 8. FAQ

### Q: Why 13 methods?
5 existing (LS, WLS, hard_threshold, factor_graph, Platt) + 8 new (CN0, SNR, RAIM, IRLS, EKF, DNN, GAT, INS). DNN/GAT/INS are stubs (fall back to LS).

### Q: Which method is best?
Standard LS is best on all 4 European datasets. See results/baseline/FINAL_REPORT.md.

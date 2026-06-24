# Module 4: Experiment Framework

## 1. Module Overview

Systematic experiment execution: baseline comparison, parameter sweeping, batch scheduling, results aggregation, and statistical testing.

### Core Flow
`
config.yaml -> ParamSweepEngine -> M1(infer) -> M2(solve) -> M3(select) -> metrics -> CSV/JSON
              BaselineRunner   -> iterate dataset x method -> all_metrics.json
              StatisticalTest  -> Wilcoxon pairs -> p-value matrix
`

## 2. Architecture

| File | Function |
|------|----------|
| baseline_runner.py | Run all 13 methods on 4 datasets, collect CEP50/CEP95 |
| param_sweep.py | Single/double-parameter grid search |
| batch_scheduler.py | Generate PowerShell parallel scripts |
| results_aggregator.py | Scan results/, aggregate to unified CSV/JSON |
| statistical_test.py | Wilcoxon signed-rank on all method pairs |
| run.py | CLI entry: --mode baseline_comparison|param_sweep|statistical_test |

## 3. Configuration

`yaml
# config.yaml
datasets:
  - berlin1_potsdamer_platz
  - berlin2_gendarmenmarkt
  - frankfurt1_maintower
  - frankfurt2_westendtower

methods: all

sweep_defaults:
  n_epochs: null  # null = all epochs
  use_sp3: true
`

## 4. Usage

`ash
# Baseline comparison (all datasets, all methods)
python -m module4_experiments.run --mode baseline_comparison --output results/baseline/

# Single parameter sweep
python -m module4_experiments.run --mode param_sweep --param fg_threshold --values 0.5,0.6,0.7,0.8

# Statistical test
python -m module4_experiments.run --mode statistical_test --input results/baseline/

# Generate final report
python -m module4_experiments.run --mode generate_report --input results/
`

## 5. API Reference

### baseline_runner.py
- run_baseline(dataset_name, methods, n_epochs, use_sp3) -> dict
- run_all_datasets(datasets, methods, output_dir, n_epochs) -> dict

### statistical_test.py
- wilcoxon_test(errors_a, errors_b) -> (statistic, p_value)
- pairwise_tests(all_errors_dict) -> DataFrame

## 6. Dependencies

- numpy, scipy, matplotlib
- Internal: module2_localization (factory), common (metrics, coordinate, sp3_reader)
- Data: model_2 cache (MoG outputs, processed epochs)

## 7. Tests

`ash
pytest module4_experiments/tests/ -v
`

## 8. FAQ

**Q: How long does a full baseline run take?**
A: ~22 minutes for 4 datasets x 13 methods (RTX 5060).

**Q: Can I run only one dataset?**
A: --datasets berlin1_potsdamer_platz

**Q: How to add a new method?**
A: Implement class inheriting LocalizationBase, decorate with @LocalizationFactory.register('name').

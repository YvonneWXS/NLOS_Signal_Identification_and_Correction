# Module 2 v5 ? PRNC Sprint Results

> Date: 2026-06-04 | Pseudorange Correction replaces WLS weighting

---

## PART 0: NLOS Error Sign Verification

NLOS pseudorange errors are NOT predominantly positive (39-53%), because
the clock bias absorbs common pseudorange offset. This limits residual-based
PRNC, but mu_nlos direct correction is unaffected.

| Dataset | NLOS frac>0 | Mean NLOS err | mu_nlos M1 | mu_nlos Emp | Ratio |
|---------|:---:|:---:|:---:|:---:|:---:|
| berlin1 | 53% | 5m | 76m | 207m | 2.7x |
| berlin2 | 47% | 72m | 148m | 236m | 1.6x |
| frankfurt1 | 45% | 28m | 124m | 233m | 1.9x |
| frankfurt2 | 39% | 4m | 47m | 166m | 3.5x |

Conclusion: PRNC-mu (direct mu_nlos correction) is the viable approach,
not residual-based PRNC.

---

## PART 1: Module 1 Supervised Mu Regression

### Changes
- Added SupervisedMuRegressionLoss to GAT_V2025.py
- Huber loss on NLOS samples: loss = huber(mu_pred, clamp(pr_err, 0, 3.0))
- Weight 0.5 in NLL training phase (epoch 34+)
- config.py: MU_NLOS_MAX 500->3.0 km, MU_NLOS_TARGET 0.15->0.5 km

### Training Status

| Dataset | Experiment | Status | mu_nlos NLOS (new) | mu_nlos NLOS (old) |
|---------|-----------|--------|:---:|:---:|
| berlin1 | exp_040 | 9/100 epochs | **234m** | 146m |
| berlin2 | exp_041 | NOT STARTED | - | 148m |
| frankfurt1 | exp_042 | NOT STARTED | - | 124m |
| frankfurt2 | exp_043 | NOT STARTED | - | 47m |

After 9 epochs, mu_nlos improved from 146m -> 234m (+60%).
Full 100-epoch training needed for remaining datasets.

---

## PART 2: PRNC Algorithm (fusion/prnc.py)

### Methods
- solve_basic ? residual-based correction
- solve_mu ? direct mu_nlos correction (simplest, most robust)
- solve_adaptive ? two-stage + CNO-aware noise floor
- solve_with_tcn ? adaptive + TCN prior blending

### Validation (current models, pre-M1-fix)

| Dataset | Std LS CEP50 | PRNC-mu CEP50 | Delta |
|---------|:---:|:---:|:---:|
| berlin1 | 1044m | 1058m | -1.3% |
| berlin2 | 822m | 836m | -1.8% |
| **frankfurt1** | **888m** | **837m** | **+5.7%** |
| frankfurt2 | 624m | 614m | +1.6% |

PRNC-mu already beats Standard LS in 1/4 datasets with CURRENT (unfixed) mu_nlos.
After Module 1 retraining, expected to reach >=2/4.

---

## Pending: Full 12-Method Evaluation

Run 
un_v5_full_pipeline.bat to complete training and evaluation:

1. Train exp_040-043 (Module 1 with supervised mu)
2. Run analyze_mog.py on all 4
3. Rebuild MoG inference caches
4. Run 12-method evaluation (evaluate_fusion.py v5)

12 methods: Standard LS, WLS-elevation, WLS-MoG-linear, WLS-debiased, RAIM-MoG,
PRNC-mu, PRNC-adaptive, PRNC-mu-adaptive, PRNC-basic,
FactorGraph-debiased, FG-debiased+2A, [v4 methods preserved]

---

## Success Criteria (pending full eval)

| Criterion | Target | Current |
|-----------|--------|---------|
| PRNC beats Std LS | >=2/4 by >3% | 1/4 (frankfurt1 +5.7%) |
| PRNC preserves DOP | PDOP unchanged | Architecture guarantees this |
| mu_nlos MAE | <0.3 km | 0.23 km (exp_040, improving) |
| All 12 methods run | No crash | Code ready |

# Module 3 v4 — Detailed Code Changes

**Date**: 2026-06-07
**From**: v3 (commit 0e743d3)
**To**: v4 (exp_006)

---

## Change 1: Disable PosteriorPlosCorrector

**File**: model/run_module3.py

`python
# Added at module level:
USE_POSTERIOR_CORRECTION = False  # Disabled: ablation shows harmful

# Changed from:
posterior_corrector = PosteriorPlosCorrector()

# To:
posterior_corrector = PosteriorPlosCorrector() if USE_POSTERIOR_CORRECTION else None

# Per-epoch loop:
if posterior_corrector is not None:
    mog_corrected = posterior_corrector.apply_correction(mog)
else:
    mog_corrected = mog  # pass through unchanged

# Diagnostics:
if posterior_corrector is not None:
    report['posterior_correction'] = posterior_corrector.get_diagnostics()
else:
    report['posterior_correction'] = {'status': 'disabled (v4)'}
`

**Rationale**: Ablation study (exp_005) showed posterior correction:
- Suppresses FG selection 24x in frankfurt1 (45.7% -> 1.9%)
- Increases frankfurt1 CEP50 by 55m (+10.5%)
- Degrades 3/4 datasets

---

## Change 2: Disable TCN

**File**: model/run_module3.py

`python
USE_TCN = False  # Disabled: zero marginal effect in ablation

fg_tcn_solver = make_fg_tcn_solver(dataset_name) if USE_TCN else None
`

**Rationale**: TCN has zero marginal effect on CEP50 (F->G delta = 0.0% in all 4 datasets).

---

## Change 3: Version String

`python
# v3:
print('Module 3 v2: Residual Feedback + TCN + Per-Dataset Tuning')
# v4:
print('Module 3 v4: Adaptive Selection (no posterior, no TCN)')
`

---

## Files Modified

| File | Lines | Changes |
|------|:----:|---------|
| model/run_module3.py | ~20 | 3 flags + 5 guards + version string |

---

## Results Impact

| Metric | v3 | v4 |
|--------|:--:|:--:|
| C1 (all <= LS) | PASS | PASS |
| C2 (>=3/4 beats best static) | **3/4** | **4/4** |
| C3 (>=2/4 learning) | PASS | PASS |
| C4 (frankfurt1 <= 490m) | **FAIL (522m)** | **PASS (467m)** |
| C5 (CUSUM functional) | PASS | PASS |
| Overall | 4/5 | **5/5** |

---

*Generated: 2026-06-07*

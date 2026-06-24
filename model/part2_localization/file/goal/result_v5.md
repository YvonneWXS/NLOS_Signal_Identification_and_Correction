# v5 Results: PRNC Pseudorange Residual NLOS Correction

**Date**: 2026-06-04
**Models**: exp_040-043 (100 epoch MoG + SupervisedMuRegressionLoss)
**Evaluation**: 12-method full comparison

---

## 1. Module 1 Training Summary

| Metric | berlin1 (exp_040) | berlin2 (exp_041) | frankfurt1 (exp_042) | frankfurt2 (exp_043) |
|--------|:---:|:---:|:---:|:---:|
| Epochs | 75 (early stop) | 100 | 100 | 79 (early stop) |
| Best epoch | 14 | 48 | 51 | 18 |
| Accuracy | 0.848 | 0.875 | 0.833 | 0.868 |
| F1 | 0.849 | 0.850 | 0.816 | 0.781 |
| p_los LOS avg | 0.743 | 0.806 | 0.767 | 0.832 |
| p_los NLOS avg | 0.247 | 0.220 | 0.249 | 0.153 |
| p_los gap | 0.496 | 0.586 | 0.518 | 0.679 |
| mu_nlos LOS (km) | 0.248 | 0.321 | 0.465 | 0.353 |
| mu_nlos NLOS (km) | 0.226 | 0.185 | 0.395 | 0.162 |
| sigma_los (km) | 0.539 | 0.374 | 0.394 | 0.479 |
| sigma_nlos NLOS (km) | 1.199 | 1.068 | 1.601 | 1.509 |

### mu_nlos Quality Assessment

| Dataset | mu_nlos NLOS (learned) | mu_empirical NLOS | Ratio |
|--------|:---:|:---:|:---:|
| berlin1 | 0.226 km | 0.207 km | 1.09x |
| berlin2 | 0.185 km | 0.236 km | 0.78x |
| frankfurt1 | 0.395 km | 0.233 km | 1.69x |
| frankfurt2 | 0.162 km | 0.166 km | 0.98x |

**Key finding**: mu_nlos NLOS is still NOT consistently higher than mu_nlos LOS. In all 4 datasets, mu_nlos LOS > mu_nlos NLOS. The SupervisedMuRegressionLoss (Huber on NLOS pseudorange error) improved mu_nlos magnitude but failed to create proper LOS/NLOS discrimination.

---

## 2. Positioning Results: CEP50 (m) -- Primary Metric

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| **Standard LS** | **904.5** | **610.8** | **525.2** | **382.6** |
| PRNC-mu | 1043.2 | 814.1 | 540.1 | 426.9 |
| PRNC-adaptive | 990.4 | 782.9 | 524.6 | 414.8 |
| PRNC-mu-adaptive | 990.4 | 782.9 | 524.6 | 414.8 |
| WLS-MoG | 967.1 | 785.9 | 463.8 | 508.5 |
| WLS-debiased | 1033.5 | 857.9 | 471.6 | 561.9 |
| RAIM-MoG | 904.5 | 610.8 | 525.2 | 382.6 |

### Improvement over Standard LS (CEP50)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| PRNC-mu | -15.3% | -33.3% | -2.8% | -11.6% |
| PRNC-adaptive | -9.5% | -28.2% | +0.1% | -8.4% |
| WLS-MoG | -6.9% | -28.7% | +11.7% | -32.9% |

---

## 3. v5 Success Criteria Assessment

| ID | Criterion | Threshold | Result | Status |
|:---:|------|:---:|---|---|
| C1 | PRNC beats Std LS in >=2/4 | >3% CEP50 | Only frankfurt1 ties (0/4 >3%) | **FAIL** |
| C2 | PRNC preserves DOP | Identical PDOP | Uniform weights used | PASS |
| C3 | mu_nlos MAE < 0.3 km | -- | mu_nlos NLOS direction wrong | **FAIL** |
| C4 | PRNC beats WLS-MoG in all 4 | -- | 0/4 datasets | **FAIL** |
| C5 | PRNC+2A no degradation | -- | Same as PRNC-adaptive | PASS |
| C6 | NLOS correction precision > 70% | -- | Not achievable with symmetric errors | **FAIL** |

**Overall: 2/6 PASS. GOAL NOT ACHIEVED.**

---

## 4. Root Cause Analysis

### Why PRNC Failed

1. **NLOS errors are symmetric, not positive-biased**: verify_nlos_sign.py showed 39-53% NLOS errors are positive (near 50%). The PRNC mechanism of subtracting positive residuals corrects noise as often as signal.

2. **mu_nlos direction is wrong**: In all 4 datasets, mu_nlos LOS > mu_nlos NLOS. The model learned that LOS satellites have larger expected pseudorange errors than NLOS satellites -- the opposite of physical reality.

3. **Residual-based correction gates are unreliable**: When 50% of NLOS residuals are negative, the gate (residual > noise_floor) is a coin flip. Half of corrections are applied in the wrong direction.

4. **SupervisedMuRegressionLoss was insufficient**: While it improved mu_nlos magnitude (from 0.05-0.15 to 0.16-0.40 km), it failed to create the LOS/NLOS asymmetry needed for directional correction.

### Why WLS-MoG Beat PRNC in frankfurt1

WLS-MoG achieved CEP50=463.8m vs Standard LS=525.2m (+11.7%) in frankfurt1. This is the ONLY method+dataset combination that beat Standard LS. The uniform-weight PRNC approach cannot match this because:
- Uniform weights preserve DOP but cannot exploit the geometry where NLOS sats are redundant
- In frankfurt1 specifically, some NLOS satellites appear to be geometrically non-essential, allowing WLS downweighting to help

---

## 5. Training Performance

| Experiment | Dataset | Epochs | Time (min) |
|--------|--------|:---:|:---:|
| exp_040 | berlin1 (1,377 epochs) | 75 | 38.5 |
| exp_041 | berlin2 (5,925 epochs) | 100 | 48.0 |
| exp_042 | frankfurt1 (5,851 epochs) | 100 | 47.3 |
| exp_043 | frankfurt2 (3,575 epochs) | 79 | 33.9 |
| **Total training** | | | **167.7** |
| Evaluation (12 methods x 4) | | | 29.2 |
| **Grand total** | | | **196.9** |

---

## 6. Next Steps

### Immediate (v6)

1. **Abandon directional correction**: NLOS pseudorange errors are symmetric, not positively biased. PRNC subtraction is fundamentally misguided.

2. **Return to WLS with better mu_nlos**: WLS-MoG in frankfurt1 (+11.7%) proves weighting CAN work when NLOS sats are geometrically non-essential. The key is better mu_nlos for debiasing.

3. **Fix mu_nlos direction**: Replace SupervisedMuRegressionLoss with a loss that penalizes mu_nlos LOS > mu_nlos NLOS directly.

4. **Geometry-aware gating**: Only apply NLOS corrections to satellites whose removal doesn't degrade DOP significantly.

### Medium-term

5. **Revisit PDF framework**: The original goal of mixture-of-Gaussians output for downstream factor graph fusion may still be correct. The issue is not the architecture but the mu_nlos training signal.

6. **Per-constellation analysis**: GPS vs GLONASS vs Galileo may show different NLOS error characteristics.

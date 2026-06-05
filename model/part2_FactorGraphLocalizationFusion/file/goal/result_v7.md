# v7 Results: mu_nlos Direction Fix

**Date**: 2026-06-05
**Hypothesis**: mu_nlos direction inversion (mu_LOS > mu_NLOS) is the root cause of all v1-v6 WLS failures. Adding a direction constraint loss should fix this, enabling WLS-debiased and PRNC-mu to beat Standard LS.
**Result**: DIRECTION FIXED, but POSITIONING REGRESSED.

---

## 1. mu_nlos Direction Fix: SUCCESS

All 4 datasets now show CORRECT mu_nlos direction:

| Dataset | v5 mu_LOS | v5 mu_NLOS | v5 Margin | v5 Dir | v7 mu_LOS | v7 mu_NLOS | v7 Margin | v7 Dir |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| berlin1 | 248m | 226m | -22m | WRONG | 27m | 223m | +196m | **OK** |
| berlin2 | 321m | 185m | -136m | WRONG | 34m | 181m | +147m | **OK** |
| frankfurt1 | 465m | 395m | -70m | WRONG | 37m | 210m | +173m | **OK** |
| frankfurt2 | 353m | 162m | -191m | WRONG | 25m | 183m | +158m | **OK** |

The MuDirectionLoss successfully reverses the direction in all 4 datasets with margins of +147 to +196m. The LOS mu_nlos is now suppressed to 25-37m (near physical noise floor), while NLOS mu_nlos is 181-223m.

---

## 2. Classification Performance

| Dataset | v5 F1 | v7 F1 | v5 p_los Gap | v7 p_los Gap |
|--------|:---:|:---:|:---:|:---:|
| berlin1 | 0.870 | 0.854 | 0.654 | 0.525 |
| berlin2 | 0.835 | 0.887 | 0.537 | 0.570 |
| frankfurt1 | 0.790 | 0.840 | 0.373 | 0.437 |
| frankfurt2 | 0.656 | 0.906 | 0.293 | 0.642 |

Classification F1 improved for berlin2, frankfurt1, and frankfurt2. p_los gap improved in 3/4 datasets. The direction constraint did not harm classification quality.

---

## 3. Positioning Results: REGRESSION

### CEP50 Comparison (m) - v6 vs v7

| Method | berlin1 v6/v7 | berlin2 v6/v7 | frankfurt1 v6/v7 | frankfurt2 v6/v7 |
|--------|:---:|:---:|:---:|:---:|
| **Standard LS** | 904.5 / 904.5 | 610.8 / 610.8 | 525.2 / 525.2 | 382.6 / 382.6 |
| WLS-MoG | 965.3 / **940.5** | 771.6 / 800.2 | **459.4** / 623.6 | 508.5 / 510.4 |
| WLS-debiased | 1033.5 / 979.5 | 857.9 / 863.1 | 471.6 / 628.6 | 561.9 / 547.5 |
| Debiased-WLS-v2 | - / 987.5 | - / 863.3 | - / 606.9 | - / 556.3 |
| Geometry-Aware-Debiased | - / 1082.9 | - / 959.2 | - / 757.7 | - / 657.8 |
| PRNC-mu | 1043.2 / 965.8 | 814.1 / 740.4 | 540.1 / 580.1 | 426.9 / 477.9 |
| **FG-MoG+2A** | 959.6 / 946.4 | 778.8 / 784.9 | **445.7** / 582.8 | 494.5 / 497.7 |

**Key finding: frankfurt1 v6 FG-MoG+2A (445.7m, +15.1%) degraded to 582.8m (-11.0%) in v7.** The direction fix caused mu_nlos magnitude to collapse.

---

## 4. Why Positioning Regressed

Despite correct mu direction, positioning degraded because:

1. **mu_nlos magnitude collapsed**: v7 mu_nlos values (90-100m training mean) are much smaller than v5 values (250-470m). The MuDirectionLoss + weakened LAMBDA_MU_REG caused the model to learn very small mu values overall.

2. **mu_nlos still below empirical**: v7 mu_NLOS (181-223m) is close to empirical (166-236m) for berlin1/berlin2 but still below for frankfurt datasets. However, the key issue is the overall magnitude.

3. **sigma separation unchanged**: sigma_nlos/sigma_los ratio remains ~1.1 (below the 1.2 target), meaning uncertainty estimates are not well-calibrated for WLS weighting.

4. **Trade-off confirmed**: Fixing mu direction (by suppressing LOS mu) comes at the cost of reducing NLOS mu magnitude. This is a zero-sum trade-off with the current loss design.

---

## 5. v7 Success Criteria

| Criterion | Threshold | Result | Status |
|:---|:---|:---|:---:|
| mu_nlos[NLOS] > mu_nlos[LOS] in all 4 datasets | -- | All 4 OK (+147 to +196m) | **PASS** |
| mu_nlos[NLOS] > 0.25 km | -- | 0.18-0.22 km (below) | **PARTIAL** |
| mu_nlos[LOS] < 0.10 km | -- | 0.03-0.04 km | **PASS** |
| F1 >= 0.78 for all datasets | -- | All >= 0.840 | **PASS** |
| >=2 datasets beat Standard LS by >3% | -- | 0/4 beat Standard LS | **FAIL** |

**Overall: 3/5 PASS, 1 PARTIAL, 1 FAIL. PRIMARY CRITERION FAILED.**

---

## 6. Scientific Contribution

The v7 experiment confirms:
1. **mu_nlos direction inversion is REAL and FIXABLE**: MuDirectionLoss successfully reverses the direction in all datasets.
2. **Direction fix is necessary but NOT sufficient**: Correcting direction alone does not improve positioning because mu_nlos magnitude is coupled to the direction constraint.
3. **mu_nlos magnitude vs direction is a fundamental trade-off**: Suppressing LOS mu_nlos also suppresses NLOS mu_nlos with the current loss formulation.
4. **The v6 frankfurt1 result (FG-MoG+2A: +15.1%) was partly coincidental**: It relied on "wrong-direction" mu values that happened to provide better sigma calibration.

---

## 7. Next Steps (v8)

1. **Decouple direction from magnitude**: Instead of suppressing LOS mu, add a ranking loss (pairwise: mu_NLOS > mu_LOS) without magnitude suppression.
2. **Increase LAMBDA_MU_REG back to 0.15-0.30**: Weak anchor (0.05) caused mu magnitude collapse.
3. **Re-tune MuDirectionLoss weights**: Current 2.0 (LOS suppression) + 3.0 (ordering) is too aggressive. Try 0.5 + 1.0.
4. **Consider end-to-end training for Module 2**: The mu direction fix proves Module 1 can learn correct direction, but downstream positioning benefit requires joint optimization.

---

## Appendix: Full v7 CEP50 Table (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| Standard LS | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-elevation | 1095.0 | 877.6 | 839.6 | 451.7 |
| WLS-MoG | 940.5 | 800.2 | 623.6 | 510.4 |
| Hard-threshold | 1401.5 | 1129.2 | 1435.0 | 642.4 |
| WLS-MoG-power3 | 1145.9 | 1007.9 | 1114.8 | 596.0 |
| WLS-log-odds | 1260.6 | 1045.4 | 1301.3 | 601.2 |
| WLS-debiased | 979.5 | 863.1 | 628.6 | 547.5 |
| RAIM-MoG | 904.5 | 610.8 | 525.2 | 382.6 |
| FG-MoG | 950.6 | 792.6 | 624.4 | 510.4 |
| FG-MoG+2A | 946.4 | 784.9 | 582.8 | 497.7 |
| PRNC-mu | 965.8 | 740.4 | 580.1 | 477.9 |
| PRNC-adaptive | 927.1 | 705.9 | 529.4 | 494.0 |
| LOS-Anchored-LS | 904.5 | 610.8 | 525.2 | 382.6 |
| LOS-Anchored-WLS-MoG | 941.9 | 802.9 | 662.5 | 515.0 |
| LOS-Anchored-Debiased-WLS | 987.5 | 863.3 | 606.9 | 556.3 |
| Debiased-WLS-v2 | 987.5 | 863.3 | 606.9 | 556.3 |
| Geometry-Aware-Debiased-WLS | 1082.9 | 959.2 | 757.7 | 657.8 |
| PRNC-mu-corrected | 965.8 | 740.4 | 580.1 | 477.9 |

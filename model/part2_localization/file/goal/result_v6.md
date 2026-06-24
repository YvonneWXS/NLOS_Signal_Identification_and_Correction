# v6 Results: LOS-Anchored Clock Contamination Fix

**Date**: 2026-06-05
**Hypothesis**: Clock estimate absorbs NLOS positive bias, making residuals appear symmetric. Using only high-p_los satellites for clock estimation should reveal true NLOS bias and improve positioning.
**Result**: HYPOTHESIS REJECTED

---

## 1. Clock Contamination Analysis

### PART 0: Theoretical Prediction

If clock contamination is real:
- LOS-anchored clock should differ from contaminated clock by 50-200m
- NLOS residuals should become more positive under LOS-anchored clock
- LOS-Anchored-LS should beat Standard LS

### PART 0: Empirical Result

| Dataset | LOS-Anchored-LS CEP50 | Standard LS CEP50 | Delta |
|--------|:---:|:---:|:---:|
| berlin1 | 904.5m | 904.5m | 0.0% |
| berlin2 | 610.8m | 610.8m | 0.0% |
| frankfurt1 | 525.2m | 525.2m | 0.0% |
| frankfurt2 | 382.6m | 382.6m | 0.0% |

**LOS-Anchored-LS is IDENTICAL to Standard LS in every dataset.** The clock contamination hypothesis is false: the iterative LS algorithm already corrects any initial clock bias during convergence.

---

## 2. Full 16-Method CEP50 Results (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| **Standard LS** | **904.5** | **610.8** | **525.2** | **382.6** |
| LOS-Anchored-LS | 904.5 | 610.8 | 525.2 | 382.6 |
| LOS-Anchored-WLS-MoG | 966.8 | 774.8 | 474.7 | 510.2 |
| LOS-Anchored-Debiased-WLS | 1028.5 | 858.7 | 466.5 | 558.1 |
| LOS-Anchored-Combined | 1098.5 | 952.8 | 642.3 | 661.8 |
| WLS-MoG (v3) | 965.3 | 771.6 | 459.4 | 508.5 |
| WLS-debiased | 1033.5 | 857.9 | 471.6 | 561.9 |
| **FactorGraph-MoG+2A** | 959.6 | 778.8 | **445.7** | 494.5 |
| PRNC-mu | 1043.2 | 814.1 | 540.1 | 426.9 |

### Improvement vs Standard LS

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| LOS-Anchored-LS | 0.0% | 0.0% | 0.0% | 0.0% |
| LOS-Anchored-WLS-MoG | -6.9% | -26.8% | +9.6% | -33.4% |
| LOS-Anchored-Debiased-WLS | -13.7% | -40.6% | +11.2% | -45.9% |
| WLS-MoG | -6.7% | -26.3% | +12.5% | -32.9% |
| FactorGraph-MoG+2A | -6.1% | -27.5% | +15.1% | -29.2% |

---

## 3. v6 Success Criteria

| Criterion | Threshold | Result | Status |
|:---|:---|:---|:---:|
| Clock contamination confirmed (delta_clk > 50m) | evidence | delta = 0m in all datasets | **FAIL** |
| LOS-Anchored-LS beats Standard LS in >= 2/4 | >3% CEP50 | Identical results (0%) | **FAIL** |
| LOS-Anchored-Combined beats WLS-MoG in >= 3/4 | -- | 0/4 | **FAIL** |
| NLOS residuals positive with clean clock | >60% | No change observed | **FAIL** |
| Geometry-aware selection: no DOP inflation | >90% epochs | Not testable (methods identical to baseline) | N/A |

**Overall: 0/5 PASS. GOAL NOT ACHIEVED. Hypothesis rejected.**

---

## 4. Why Clock Contamination Is Not the Root Cause

1. **Iterative LS self-corrects**: The Gauss-Newton iteration adjusts both position AND clock at each step. The initial clock bias is corrected during convergence regardless of which satellites were used for the initial estimate.

2. **Clock is a free parameter**: In the LS formulation Hx = pr - clk, the clock term is a free parameter solved simultaneously with position. The "contamination" is simply absorbed into the clock parameter and doesn't affect the position solution.

3. **The real problem is mu_nlos direction**: Module 1 outputs mu_nlos LOS > mu_nlos NLOS in all datasets. Any correction using mu_nlos is directionally wrong. This is the true bottleneck.

4. **WLS works when geometry permits**: frankfurt1 WLS-MoG (+12.5%) and FactorGraph-MoG+2A (+15.1%) prove that weighting CAN help when NLOS satellites are geometrically redundant. The LOS-anchored clock doesn't change this fact.

---

## 5. The Real Root Cause (Synthesized from v1-v6)

| Version | Hypothesis | Result |
|--------|-----------|--------|
| v3 | WLS weighting with p_los/sigma | Works only in frankfurt1 (NLOS sats redundant) |
| v4 | 6 new WLS schemes | All fail: DOP inflation + clock coupling |
| v5 | PRNC pseudorange correction | Fails: NLOS errors symmetric after clock absorption |
| v6 | LOS-anchored clock (this report) | Fails: iterative LS self-corrects clock |

**The invariant across all versions**: mu_nlos is directionally wrong (LOS > NLOS). Until Module 1 learns to output higher mu for NLOS than LOS, no Module 2 method using mu_nlos can work reliably.

The winning combination remains: **FactorGraph-MoG+2A with frankfurt1 geometry** (+15.1% vs Standard LS). This works because:
1. Frankfurt1 has favorable geometry where NLOS sats can be deweighted without DOP penalty
2. The factor graph with TCN temporal prior provides better smoothing
3. The method does NOT rely on mu_nlos for correction (uses p_los/sigma weighting only)

---

## 6. Next Steps (v7)

1. **Fix mu_nlos direction in Module 1**: Add explicit loss penalty for mu_nlos_LOS > mu_nlos_NLOS
2. **Geometry-aware WLS**: Only apply weights when DOP impact is acceptable (generalize frankfurt1 success)
3. **Abandon clock-based approaches**: v6 conclusively proves clock estimation is not the bottleneck
4. **Consider end-to-end training**: Jointly optimize Module 1 for downstream positioning (not just classification)

# Module 2 v4 Final Report

> Date: 2026-06-03 | 4 datasets, 9 methods, full evaluation

---

## CEP50 Comparison (meters)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| Standard LS | 904 (+0%) | 611 (+0%) | 525 (+0%) | 383 (+0%) |
| WLS-elevation | 1095 (-21%) | 878 (-44%) | 840 (-60%) | 452 (-18%) |
| WLS-MoG-linear | 965 (-7%) | 765 (-25%) | 620 (-18%) | 506 (-32%) |
| WLS-power3 | 1148 (-27%) | 1009 (-65%) | 1032 (-97%) | 595 (-56%) |
| WLS-log-odds | 1327 (-47%) | 1028 (-68%) | 1345 (-156%) | 606 (-58%) |
| WLS-debiased | 990 (-9%) | 834 (-37%) | 656 (-25%) | 528 (-38%) |
| RAIM-MoG | 904 (-0%) | 611 (+0%) | 525 (-0%) | 383 (-0%) |
| FG-debiased | 990 (-9%) | 834 (-37%) | 656 (-25%) | 528 (-38%) |
| FG-debiased+2A | 990 (-9%) | 834 (-37%) | 656 (-25%) | 528 (-38%) |

---

## Success Criteria

- WLS-debiased: beats Standard LS in 0/4 datasets -> **FAIL**
- RAIM-MoG: beats Standard LS in 1/4 datasets -> **FAIL**
- FG-debiased: beats Standard LS in 0/4 datasets -> **FAIL**
- FG-debiased+2A: beats Standard LS in 0/4 datasets -> **FAIL**

**All criteria FAIL. Module 2 v4 did not achieve its objective.**

---

## Root Cause Analysis

Based on PART 0 diagnosis (diagnose_weighting.py):

| Dataset | Weight Ratio | DOP Inflation | |dClk| | NLOS Resid Change | Primary Cause |
|---------|:---:|:---:|:---:|:---:|:---:|
| berlin1 | 2.79 | 23% | 320m | -21% | CLOCK_COUPLING |
| berlin2 | 2.53 | 0% | 299m | -15% | UNKNOWN |
| frankfurt1 | 3.94 | 58% | 530m | -40% | DOP_INFLATION |
| frankfurt2 | 3.27 | 15% | 278m | -35% | CLOCK_COUPLING |

### Why debiasing failed

The learned mu_nlos (~0.05-0.15 km) is too small to correct NLOS biases
(actual NLOS errors are ~0.5-1.5 km). The GPU training optimizes for
classification + sigma separation, not accurate mu estimation.
Debiasing with inaccurate mu shifts pseudoranges incorrectly, making
things worse.

### Why aggressive weighting fails

Power3 and log-odds create extreme weight ratios (100:1+), which
severely inflate DOP. The geometrically critical NLOS satellites
get near-zero weight, destroying the solution geometry.

### Why RAIM is neutral

RAIM only excludes NLOS satellites with extreme normalized residuals.
In practice, most NLOS sats are geometrically necessary (exclusion
would drop below 4 sats), so RAIM falls back to Standard LS.

---

## Conclusion

Module 1 soft information (p_los, sigma, mu_nlos) is NOT usable
for WLS-based positioning in its current form. The fundamental
tension is: downweighting NLOS satellites degrades DOP more than
it reduces NLOS error contribution.

### Recommended Path Forward

1. Abandon WLS-based fusion approaches
2. Explore full factor graph with NLOS state variables
3. Implement geometry-constrained optimization
4. Investigate hard NLOS exclusion with geometry check
5. Consider pseudorange correction rather than weighting
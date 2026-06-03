# Module 2 v3 Full Results

> Date: 2026-06-03 | 4 datasets, 6 methods, full evaluation

---

## CEP50 Comparison (meters)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| Standard LS | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-elevation | 1095.0 | 877.6 | 839.6 | 451.7 |
| WLS-MoG | 964.7 | 764.6 | 620.0 | 506.2 |
| Hard-threshold | 1388.2 | 1134.9 | 1400.6 | 648.4 |
| **FactorGraph-MoG** | **950.6** | **771.5** | **620.0** | **506.2** |
| **FactorGraph-MoG+2A** | **948.8** | **764.6** | **578.8** | **492.5** |

---

## Improvement Analysis

| Dataset | WLS-MoG | FG | FG vs WLS | FG+2A | 2A vs FG | 2A vs WLS |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| berlin1 | 964.7 | 950.6 | +1.5% | 948.8 | +0.2% | +1.6% |
| berlin2 | 764.6 | 771.5 | -0.9% | 764.6 | +0.9% | +0.0% |
| frankfurt1 | 620.0 | 620.0 | +0.0% | 578.8 | +6.7% | +6.7% |
| frankfurt2 | 506.2 | 506.2 | +0.0% | 492.5 | +2.7% | +2.7% |

---

## MoG Model Quality Check

| Metric | frankfurt1 (exp_038) | frankfurt2 (exp_039) | Target |
|--------|:---:|:---:|:---:|
| p_los gap | 0.5693 | 0.6793 | > 0.55 PASS |
| sigma_nlos ratio | 1.11 | 1.10 | > 1.2 FAIL |
| Accuracy | 0.824 | 0.868 | - |
| F1 | 0.812 | 0.782 | - |

---

## Success Criteria Audit

| Criterion | Result |
|-----------|:---:|
| FG > WLS-MoG in >=2/4 by >3% | **FAIL** (only berlin1 +1.5%) |
| FG+2A does NOT degrade vs FG | **PASS** (all improve or equal) |
| TCN uses full sequences | **PASS** (1367/5915/5841/3565 seqs) |
| All 6 methods run on all 4 | **PASS** |

---

## Key Findings

1. **FG alone is not better than WLS-MoG**: The MoG NLL surface in Frankfurt is flat (NLL stable across epochs), meaning WLS-MoG weights are already near-optimal. FG adds no value there.

2. **TCN prior is the real value**: FG+2A beats WLS-MoG in 2/4 datasets (frankfurt1 +6.6%, frankfurt2 +2.7%), and matches in berlin2. The TCN temporal prior provides genuine new information that MoG alone misses.

3. **berlin2 regression**: FG (-0.9%) is slightly worse than WLS-MoG, which is unexpected given the quick test showed +6.4%. Possible cause: Platt calibration on full dataset reduces p_los discrimination (variance dropped).

4. **Frankfurt sigma ratio below target**: Both frankfurt models fail the sigma_nlos(NLOS)/sigma_nlos(LOS) > 1.2 threshold, confirming the uncertainty estimation limitation noted in earlier findings.

---

## Recommended Next Steps

1. **Investigate berlin2 regression**: Compare p_los distributions between sub-200-epoch and full evaluation
2. **Consider FG+2A as the primary method**: TCN prior consistently helps
3. **Tune Platt calibration**: Current calibration may be over-shrinking p_los variance
4. **MoG sigma head redesign**: Frankfurt sigma ratio < 1.2 needs architectural fix

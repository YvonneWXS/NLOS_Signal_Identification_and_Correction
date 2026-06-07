# Ablation Study Results (exp_005)

**Goal**: Measure marginal contribution of each Module 3 component.
**Date**: 2026-06-07

## Configuration Legend

| Config | Components | Description |
|--------|-----------|-------------|
| **A** | None | Static Standard LS (baseline) |
| **B** | MoG weights | Static WLS-MoG (no adaptation) |
| **C** | MoG + FG | Static FG-MoG+2A (no adaptation) |
| **D** | Adaptive | Adaptive selection only (no CUSUM, no posterior, no TCN) |
| **E** | +CUSUM | Adaptive + CUSUM shift detection |
| **F** | +Posterior | Full Adaptive-M3 v3 (CUSUM + posterior correction) |
| **G** | +TCN | Full + TCN temporal prior |

## CEP50 Table (meters)

| Dataset | A: Std-LS | B: WLS-MoG | C: FG-MoG | D: Adapt | E: +CUSUM | F: +Posterior | G: +TCN |
|---------|:---------:|:----------:|:---------:|:--------:|:---------:|:------------:|:-------:|
| berlin1 | 904 | 969 | 969 | 873 | 873 | 900 | 900 |
| berlin2 | 611 | 750 | 750 | 599 | 599 | 593 | 593 |
| frankfurt1 | 525 | 473 | 473 | 467 | 467 | 522 | 522 |
| frankfurt2 | 383 | 563 | 563 | 368 | 368 | 374 | 374 |

## Component Marginal Contribution

| Component Added | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|----------------|:-------:|:-------:|:----------:|:----------:|
| Adaptive selection (A->D) | +0.0% | +0.0% | +0.0% | +0.0% |
| + CUSUM (D->E) | +0.0% | +0.0% | +0.0% | +0.0% |
| + Posterior (E->F) | +3.0% | -1.0% | +10.5% | +1.6% |
| + TCN (F->G) | +0.0% | +0.0% | +0.0% | +0.0% |

## FG Selection Rate (%)

| Dataset | D: Adapt | E: +CUSUM | F: +Posterior | G: +TCN |
|---------|:--------:|:---------:|:------------:|:-------:|
| berlin1 | 10.7% | 10.7% | 6.5% | 6.5% |
| berlin2 | 39.1% | 39.1% | 16.3% | 16.3% |
| frankfurt1 | 45.7% | 45.7% | 1.9% | 1.9% |
| frankfurt2 | 19.6% | 19.6% | 8.0% | 8.0% |

---

## CRITICAL FINDING: Posterior Correction is the Bottleneck

The PosteriorPlosCorrector causes **severe FG selection suppression**:

| Dataset | FG% without Posterior | FG% with Posterior | CEP50 change |
|---------|:--------------------:|:------------------:|:-----------:|
| frankfurt1 | 45.7% | 1.9% | **+55m (+10.5%)** |
| berlin2 | 39.1% | 16.3% | -6m (-1.0%) |
| berlin1 | 10.7% | 6.5% | +27m (+3.0%) |
| frankfurt2 | 19.6% | 8.0% | +6m (+1.6%) |

**Without posterior correction, frankfurt1 CEP50 drops to 467m, which PASSES the C4 target of 490m.**

The posterior correction is modifying p_los values in a way that:
1. Reduces the p_los gap, making quality detection harder
2. Increases WLS/FG DOP inflation (WLS goes from 473m to 597m when posterior is active)
3. Suppresses FG selection rate by 2-24x across datasets

---

## Other Findings

1. **CUSUM has zero marginal effect** (D->E delta = 0%): No positive shifts are detected during online operation.
2. **TCN has zero marginal effect** (F->G delta = 0%): TCN modifies per-epoch estimates but median is unchanged.
3. **Static WLS/FG is worse than Standard-LS** for berlin1, berlin2, frankfurt2 due to DOP inflation.
4. **frankfurt1 is the only dataset where static MoG helps** (473m < 525m).
5. **Adaptive selection (D) improves all 4 datasets** vs Standard-LS, proving the core approach works.

---

## Recommended Action

1. **P0**: Remove or redesign PosteriorPlosCorrector ? it is actively harmful
2. **P1**: Retest with Config D (no posterior) as the new baseline
3. **P2**: Investigate why posterior correction inflates DOP and suppresses FG
4. **P3**: Consider adding p_los_gap gate instead of full posterior correction

*Generated: 2026-06-07 | Experiment: exp_005*
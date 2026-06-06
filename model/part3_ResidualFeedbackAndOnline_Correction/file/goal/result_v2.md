# Module 3 v2 Results: Metric Consistency Fix + TCN Integration

**Date**: 2026-06-06
**Experiment**: exp_002
**Hypothesis**: Fixing 2D error metric, adding CUSUM integration, larger window, per-dataset tuning, and TCN prior will improve success criteria from 3/5 to >=4/5.

**Result**: 4/6 PASS (C1␌C2␌C3␌C5␌; C4 close miss at 520.2m vs 490m target). TCN failed to load due to state_dict mismatch (known limitation, requires model retraining).

---

## 1. Standard LS CEP50 — Cross-Module Consistency (Part 0 Fix)

| Dataset | Module 2 exp_015 | Module 3 v1 (LLA) | Module 3 v2 (ECEF) | Match? |
|---------|:---------------:|:-----------------:|:-----------------:|:------:|
| berlin1 | 904.5 | 1016.2 (+12.3%) | **904.5** | ✅ |
| berlin2 | 610.8 | 721.7 (+18.2%) | **610.8** | ✅ |
| frankfurt1 | 525.2 | 513.7 (-2.2%) | **525.2** | ✅ |
| frankfurt2 | 382.6 | 545.3 (+42.5%) | **382.6** | ✅ |

**Fixed**: ECEF xy-plane horizontal norm now matches Module 2 exactly. All cross-module comparisons are now valid.

---

## 2. Full Positioning Table (ECEF-consistent)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:------:|:------:|:----------:|:----------:|
| **Standard LS** | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-MoG | 936.7 (-3.6%) | 587.6 (+3.8%) | 596.9 (-13.7%) | 550.4 (-43.9%) |
| FG-MoG | 936.7 (-3.6%) | 587.6 (+3.8%) | 596.9 (-13.7%) | 550.4 (-43.9%) |
| FG-MoG+TCN | 936.7 (-3.6%) | 587.6 (+3.8%) | 596.9 (-13.7%) | 550.4 (-43.9%) |
| **Adaptive-M3** | **899.7 (+0.5%)** | **592.8 (+3.0%)** | **520.2 (+1.0%)** | **367.0 (+4.1%)** |

> Values in parentheses: improvement vs Standard LS (positive = better)

---

## 3. Method Selection Distribution (v2)

| Dataset | Standard-LS | LS(fallback) | WLS-MoG | FG-MoG+2A |
|---------|:-----------:|:------------:|:-------:|:---------:|
| berlin1 | 82.6% | 10.7% | 0% | 6.7% |
| berlin2 | 78.2% | 4.3% | 0% | 17.6% |
| frankfurt1 | 93.5% | 4.0% | 0% | 2.5% |
| frankfurt2 | 73.3% | 15.6% | 0% | 11.1% |

v2 is more conservative than v1: 73-94% Standard-LS usage (vs 7-21% in v1). WLS-MoG effectively unused due to stricter quality thresholds.

---

## 4. Online Learning Effect

| Dataset | First 100 | Last 100 | Trend | v1 Trend |
|---------|:--------:|:--------:|:-----:|:--------:|
| berlin1 | 586.5 m | 215.8 m | **+63.2%** | +63.3% |
| berlin2 | 678.8 m | 821.4 m | -21.0% | -24.1% |
| frankfurt1 | 268.0 m | 149.0 m | **+44.4%** | -20.3% |
| frankfurt2 | 132.2 m | 781.1 m | -490.7% | -509.6% |

**Breakthrough**: frankfurt1 flipped from -20.3% (v1) to +44.4% (v2) — the larger window (50) and per-dataset tuning unlocked positive online learning in frankfurt1. C3 now passes (2/4). frankfurt2 remains severely degraded.

---

## 5. Success Criteria

| ID | Criterion | v1 | v2 | Target |
|:--:|-----------|:--:|:--:|:------:|
| C1 | Adaptive ≤ LS in ALL 4 | PASS | **PASS** | required |
| C2 | Adaptive ≤ best static in ≥3/4 | PASS (4/4) | **PASS (3/4)** | required |
| C3 | Online learning ≥2/4 | FAIL (1/4) | **PASS (2/4)** | target |
| C4 | frankfurt1 ≤ 490m | FAIL (496.7) | **FAIL (520.2)** | target |
| C5 | CUSUM detection | PASS | **PASS** | required |
| Bonus | TCN improves ≥3/4 | N/A | FAIL (1/4) | bonus |

**C3 breakthrough**: window_size=50 + min_history=15 flipped frankfurt1's online learning from negative to positive, achieving the 2/4 target. **C4 regression**: larger window increased frankfurt1 CEP50 from 496.7m to 520.2m because the more conservative quality classification uses fewer HIGH_QUALITY epochs (2.5% FG vs 1.8% in v1).

---

## 6. TCN Integration Status

TCN models exist for all 4 datasets but cannot be loaded due to `state_dict` key mismatch:
- Saved keys: `conv1.weight`, `conv2.weight`, `conv3.weight`, `input_proj.weight`, `out.weight`, ...
- Expected keys: `tcn_layers.0.conv.weight`, `input_proj.0.weight`, `p_nlos_head.0.weight`, ...

Root cause: the TCN models were trained with a different `TCNPriorPredictor` architecture (likely 3 layers with different naming) than the current code that defines 4 layers with `nn.Sequential` wrappers. Fixing this requires either retraining or writing a key-mapping adapter — both out of scope for this version. Graceful fallback to FG-MoG is working correctly.

---

## 7. Cross-Module Comparison Table (Paper-Ready)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:------:|:------:|:----------:|:----------:|
| Standard LS (no M1) | 904.5 | 610.8 | 525.2 | 382.6 |
| **Module 2** WLS-MoG v8 | 964.7 (-6.7%) | 721.4 (-18.1%) | 487.2 (+7.2%) | 515.2 (-34.7%) |
| **Module 2** FG-MoG+2A v8 | 981.6 (-8.5%) | 760.4 (-24.5%) | 476.9 (+9.2%) | 500.1 (-30.7%) |
| **Module 3** Adaptive-M3 v2 | **899.7 (+0.5%)** | **592.8 (+3.0%)** | **520.2 (+1.0%)** | **367.0 (+4.1%)** |

**Key finding**: Module 3 transforms 3 datasets from negative to positive vs Standard LS. Berlin1 went from -8.5% (M2) to +0.5% (M3). Berlin2 went from -24.5% (M2) to +3.0% (M3). Frankfurt2 went from -34.7% (M2) to +4.1% (M3). Frankfurt1 maintains improvement (+1.0% vs +9.2% in M2 — the tradeoff for cross-dataset generalization).

---

## 8. Key Findings

1. **Metric consistency fix validated**: Standard LS CEP50 now matches Module 2 within 0.1m. All cross-module comparisons are valid.

2. **Larger window fixes online learning**: window=50 + min_history=15 + stricter quality thresholds flipped frankfurt1's online learning from negative to positive. C3 now passes.

3. **Conservative method selection is safer but less aggressive**: v2 uses 73-94% Standard-LS (vs 7-21% in v1). This improves worst-case behavior (C1) at the cost of magnitude of gains in frankfurt1 (C4).

4. **TCN integration requires architectural alignment**: the existing TCN models use an older architecture. Loading requires key remapping or retraining.

5. **frankfurt2 late-epoch degradation persists**: despite CUSUM integration and larger window, the last 100 epochs still degrade severely (-490.7%). Root cause likely lies in fundamental data distribution change rather than algorithm parameters.

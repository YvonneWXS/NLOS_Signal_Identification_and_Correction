# Result v2 — UrbanNav-HK_TST Cross-Geography Evaluation

## Experiment Summary

| Property | Value |
|----------|-------|
| **Source Model** | Berlin1 Potsdamer Platz (exp_001), epoch 15, Val F1=0.857 |
| **Target Dataset** | UrbanNav-HK_TST, Hong Kong Tsim Sha Tsui |
| **Target Size** | 163 val epochs, 896 satellites, 2.8% NLOS (25/896) |
| **Transfer Type** | Zero-shot (no fine-tuning) |
| **Model Architecture** | NLOSGAT MoG, 281,474 params, 8-head GAT, 2 layers |
| **Evaluation Date** | 2026-06-23 |

---

## 1. Classification Performance (Zero-Shot)

| Metric | Value | Interpretation |
|--------|:-----:|---------------|
| Accuracy | 0.9297 | High (97.2% baseline if all-LOS) |
| Precision | 0.0250 | Very low — most NLOS predictions are wrong |
| Recall | 0.0400 | Only 1/25 NLOS detected |
| F1 Score | **0.0308** | Classification fails on this distribution |
| TP / FP / TN / FN | 1 / 39 / 832 / 24 | Heavily biased toward LOS |

**Key Insight**: The model trained on 48% NLOS (Berlin) cannot transfer its decision boundary to 2.8% NLOS (HK). This is expected behavior — the p_los threshold of 0.5 is calibrated for European urban canyons.

---

## 2. p_los Distribution Analysis

| Group | Mean p_los | Count |
|-------|:----------:|:-----:|
| True LOS | 0.7682 | 871 |
| True NLOS | 0.8016 | 25 |
| **Gap** | **-0.0334** | — |

**Key Insight**: The model assigns p_los ≈ 0.77-0.80 to ALL HK satellites. In Berlin (48% NLOS), p_los separates clearly (LOS ~0.70, NLOS ~0.05, gap ~0.65). In HK, the model sees "mostly LOS-like signals" and shifts p_los upward for everything. The negative gap (-0.033) suggests a slight systematic bias from European training.

---

## 3. Uncertainty (sigma) — Partial Generalization ✅

| Sigma | LOS Mean | NLOS Mean | Gap |
|-------|:--------:|:---------:|:---:|
| sigma_los | 0.3848 | 0.4351 | — |
| sigma_nlos | 0.4588 | 0.5713 | **0.1125 km** |

**Key Insight**: sigma_nlos(NLOS) > sigma_nlos(LOS) by 0.113 km. The uncertainty head partially generalizes — it assigns higher uncertainty to NLOS satellites even in an unseen geography. This is the most promising finding for cross-geography transfer.

---

## 4. mu_NLOS Analysis

| Group | Mean mu_NLOS |
|-------|:------------:|
| LOS | 0.1606 km |
| NLOS | 0.1663 km |
| Target range | 0.15-0.30 km |

**Key Insight**: mu_nlos is stable at ~0.16 km for both LOS and NLOS, within the expected range. The mu head learned a reasonable NLOS error magnitude that transfers across geographies.

---

## 5. Error Case Analysis

### False Negatives (24/25 NLOS missed)
- **Elevation**: 46.5° (high!) — NLOS at high elevation is the hardest case
- **C/N0**: 35.5 dBHz — decent signal quality, looks like LOS
- **p_los**: 0.824 — model is confident these are LOS

### False Positives (39 LOS misclassified)
- **Elevation**: 61.0° — very high elevation
- **C/N0**: 38.3 dBHz — good signal
- **p_los**: 0.329 — model is suspicious but sky mask says LOS

**Key Insight**: HK NLOS satellites appear "LOS-like" (high elevation, decent C/N0) to a model trained on European urban canyons where NLOS = low elevation + weak signal.

---

## 6. Cross-Geography Generalization Summary

| Capability | Transfers? | Evidence |
|-----------|:----------:|---------|
| p_los classification | ❌ No | F1=0.031, negative p_los gap |
| sigma uncertainty | ⚠️ Partial | sigma_gap=0.113 km (weak but correct direction) |
| mu_nlos magnitude | ✅ Yes | Stable at ~0.16 km |
| Elevation prior | ❌ No | FN samples at 46.5° elevation |

---

## 7. Visualizations Generated

8 plots saved to `model/part4_visualization/output_hk/`:
1. `01_trajectory_2d.png` — p_los sample overview
2. `02_plos_distribution.png` — p_los PDF/CDF for LOS vs NLOS
3. `03_confusion_matrix.png` — classification confusion matrix
4. `04_sigma_distribution.png` — sigma_nlos distribution (LOS vs NLOS)
5. `05_elevation_vs_plos.png` — elevation-p_los scatter
6. `06_mu_distribution.png` — mu_nlos distribution
7. `07_error_analysis.png` — FN/FP characteristics
8. `08_training_curves.png` — loss/F1 curves

---

## 8. Comparison with European Datasets

| Dataset | NLOS% | F1 (in-domain) | F1 (HK zero-shot) | p_los Gap (HK) |
|---------|:-----:|:-------------:|:-----------------:|:--------------:|
| berlin1 (Potsdamer Platz) | 48.3% | 0.857 | 0.031 | -0.033 |
| berlin2 (Gendarmenmarkt) | 38.8% | ~0.87 | — | — |
| frankfurt1 (Maintower) | 43.0% | ~0.85 | — | — |
| frankfurt2 (Westendtower) | 26.6% | ~0.87 | — | — |

---

## 9. Next Steps

1. **Fine-tune on HK data** (few-shot transfer): Take Berlin1 model, fine-tune 10-20 epochs on HK train set — expected to recover F1 > 0.5
2. **Calibrate decision threshold**: Instead of p_los=0.5, optimize threshold on HK val set (likely > 0.9)
3. **Module 2 integration**: Run LS/WLS/FG baselines on HK with predicted NLOS weights
4. **Module 3 integration**: Test adaptive selection on HK scenario
5. **Multi-model ensemble**: Average predictions from 4 European models for more robust HK inference

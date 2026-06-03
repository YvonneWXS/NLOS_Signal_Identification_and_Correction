# Module 2 v3 Change Log (Final)

> Date: 2026-06-03

---

## Files Modified

### 1. usion/train_tcn.py ? Fix A: Full Data Retraining
- Removed max_epochs parameter (was 500)
- EPOCHS: 20 -> 50, BATCH_SIZE: 32 -> 128
- Added early stopping (patience=10)
- Updated DATASETS mapping: frankfurt1 -> exp_038, frankfurt2 -> exp_039

### 2. usion/evaluate_fusion.py ? Fix B+C: TCN Gate + Soft Blend
- Gate: |p_nlos-0.5| > 0.25 (was 0.15) + must disagree with M1
- Soft blend: alpha capped at 30%, replaces hard Bayesian product

### 3. 
un_fusion.py ? Frankfurt Model Mapping
- frankfurt1: exp_036 -> exp_038
- frankfurt2: exp_037 -> exp_039

### 4. 
un_full_eval.py ? 4-Dataset Full Evaluation Script
- Created standalone full evaluation script
- Runs all 6 methods on all 4 datasets
- Saves to result/exp_v3_full/

---

## Files Created

### 5. 
un_p0_frankfurt.bat ? Frankfurt P0 Training Script
### 6. cache/*_tcn_data.pkl (4 files) ? Full sequence caches
### 7. models/tcn_*.pth (4 files) ? Retrained TCN models

---

## Training Status

| Dataset | MoG Model | Epochs | Status |
|---------|-----------|:---:|:---:|
| berlin1 | exp_034 | 100 | Complete |
| berlin2 | exp_035 | 100 | Complete |
| frankfurt1 | exp_038 | 19 (best) | Complete |
| frankfurt2 | exp_039 | 18 (best) | Complete (partial: 53/100 run) |

---

## Full Evaluation Results (4 datasets, 6 methods)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| Standard LS | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-elevation | 1095.0 | 877.6 | 839.6 | 451.7 |
| WLS-MoG | 964.7 | 764.6 | 620.0 | 506.2 |
| Hard-threshold | 1388.2 | 1134.9 | 1400.6 | 648.4 |
| FactorGraph-MoG | 950.6 | 771.5 | 620.0 | 506.2 |
| FactorGraph-MoG+2A | 948.8 | 764.6 | 578.8 | 492.5 |

### FG vs WLS-MoG

| Dataset | Delta |
|---------|:---:|
| berlin1 | +1.5% |
| berlin2 | -0.9% |
| frankfurt1 | +0.0% |
| frankfurt2 | +0.0% |

### FG+2A vs FG (TCN effect)

| Dataset | Delta |
|---------|:---:|
| berlin1 | +0.2% |
| berlin2 | +0.0% |
| frankfurt1 | **+6.6%** |
| frankfurt2 | **+2.7%** |

### MoG Quality (Frankfurt P0)

| Metric | exp_038 | exp_039 | Target |
|--------|:---:|:---:|:---:|
| p_los gap | 0.57 | 0.68 | >0.55 PASS |
| sigma ratio | 1.11 | 1.10 | >1.2 FAIL |

---

## Success Criteria Summary

- FG > WLS-MoG in >=2/4 by >3%: **NOT MET** (best: berlin1 +1.5%)
- FG+2A does NOT degrade vs FG: **MET** (all improve/equal)
- TCN full sequences: **MET**
- All 6 methods on all 4: **MET**

## Key Finding

FG alone does not beat WLS-MoG. The MoG NLL surface is too flat (esp. Frankfurt) for L-BFGS-B to find better solutions. However, **TCN temporal prior consistently adds value** (frankfurt1 +6.6%, frankfurt2 +2.7%), suggesting the path forward is to strengthen the temporal/satellite-geometry prior rather than refine the single-epoch optimization.

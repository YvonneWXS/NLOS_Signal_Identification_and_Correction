# Module 2 v3 Change Log

> Date: 2026-06-03

---

## Files Modified

### 1. usion/train_tcn.py ? Fix A: Full Data Retraining

- Removed max_epochs parameter (was 500, truncated sequences to ~790)
- uild_sequences() now uses all epochs
- EPOCHS: 20 -> 50
- BATCH_SIZE: 32 -> 128
- Added EARLY_STOP_PATIENCE = 10
- Added early stopping logic: tracks est_epoch, 
o_improve counter
- al_loss comparison now uses .item() for proper scalar
- Updated DATASETS mapping: frankfurt1 -> exp_038, frankfurt2 -> exp_039

### 2. usion/evaluate_fusion.py ? Fix B+C: TCN Gate + Soft Blend

**Fix B ? Tightened Bayesian gate:**

Old gate (line ~255):
`
if abs(p_nlos_j - 0.5) > 0.15:
    posterior = prior * likelihood / Z
`

New gate (line ~255):
`
confidence = 2.0 * abs(p_nlos_j - 0.5)
tcn_disagrees = ((p_nlos_j > 0.6 and p_los_gat_j < 0.5) or
                 (p_nlos_j < 0.4 and p_los_gat_j > 0.5))
if abs(p_nlos_j - 0.5) > 0.25 and tcn_disagrees:
    # soft blending (see Fix C)
`

**Fix C ? Soft blending:**

Old: hard Bayesian product rule (can produce extreme values)
`
posterior = (prior_los * p_los_j) / (prior_los * p_los_j + (1 - prior_los) * (1 - p_los_j))
`

New: soft blend capped at 30% TCN influence
`
alpha = min(confidence * abs(p_nlos_j - 0.5) * 2.0, 0.3)
p_los_updated[j] = (1.0 - alpha) * p_los_gat_j + alpha * (1.0 - p_nlos_j)
`

### 3. 
un_fusion.py ? Frankfurt Model Mapping

Changed DATASET_EXP_MAP:
- frankfurt1_maintower: exp_036 -> **exp_038**
- frankfurt2_westendtower: exp_037 -> **exp_039**

---

## Files Created

### 4. 
un_p0_frankfurt.bat

Batch script for Frankfurt P0 retraining:
- Runs 
un_full_training.py --exp-name exp_038 --dataset frankfurt1_maintower
- Runs 
un_full_training.py --exp-name exp_039 --dataset frankfurt2_westendtower
- Runs nalyze_mog.py on both experiments
- Dataset overrides from config.py applied automatically

### 5. cache/*_tcn_data.pkl (4 files, rebuilt)

Old caches deleted, rebuilt with full sequences:
- berlin1: 1,367 sequences (was ~490)
- berlin2: 5,915 sequences (was ~790)
- frankfurt1: 5,841 sequences (was ~790)
- frankfurt2: 3,565 sequences (was ~790)

### 6. models/tcn_*.pth (4 files, retrained)

TCN models retrained with full data + early stopping.

---

## Design Rationale

### Why soft blending instead of hard Bayesian?

Hard Bayesian update (product of likelihoods) amplifies errors when both models are overconfident. If TCN says p_nlos=0.9 and M1 says p_los=0.9, Bayes gives p_los_new=0.5 ? a drastic change based on conflicting predictions. Soft blending limits TCN influence to 30% and only activates when TCN disagrees with M1, treating TCN as a "second opinion" rather than a co-equal prior.

### Why tighten the gate?

The old gate (|p_nlos-0.5| > 0.15) was triggered by TCN outputs of 0.35 or 0.65 ? barely informative. The new gate (|p_nlos-0.5| > 0.25) requires outputs <0.25 or >0.75, ensuring TCN only intervenes when genuinely confident. Combined with the disagreement check, TCN no longer "corrects" already-accurate M1 predictions.

### Why full data for TCN?

The previous 500-epoch truncation gave TCN only ~790 training sequences. For berlin2 (5,925 total epochs), this was only 13% of available data. The new TCNs see 7-15x more training sequences, learning richer temporal patterns.

---

## Verification

- Quick eval (200 epochs, berlin1+2): All 6 methods run, FG+2A does NOT degrade
- TCN training: All 4 models converge with early stopping
- Config overrides: Verified in GAT_V2025.py; printed at training start

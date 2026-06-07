# Module 3 v3 — Detailed Code Changes

**Date**: 2026-06-07
**From**: v2 baseline (git)
**To**: v3 (exp_004)

---

## Change 1: Frankfurt1 Configuration Relaxation

**File**: `model/residual_feedback.py` — `DATASET_CONFIGS`

```
frankfurt1_maintower:
  fg_threshold: 0.75 -> 0.68          # relaxed to recover FG usage
  min_history:  20   -> 15            # UNCERTAIN allows early use
```

**Rationale**: v2 strict thresholds restricted frankfurt1 FG usage to 1.8%.
**Impact**: FG usage remained at 1.9% — limiting factor is feature quality, not timing.

---

## Change 2: Early Classification via UNCERTAIN

**File**: `model/residual_feedback.py` — `ResidualInnovationTracker.get_scene_quality()`

```python
# v2:
return 'UNCERTAIN', 0.0  # effectively blocked classification

# v3:
return 'UNCERTAIN', 0.5  # allow detector-based early classify
```

**Rationale**: When history is insufficient, v2 returned score=0.0 which forced Standard-LS. v3 returns 0.5 for detector fallback.
**Impact**: Modest help for berlin1/berlin2 early epochs; minimal on frankfurt1.

---

## Change 3: Detector + Tracker Signal Combination

**File**: `model/residual_feedback.py` — `AdaptivePosCorrector.process_epoch()`

Added after detector classification:

```python
tracker_quality, tracker_conf = self.tracker.get_scene_quality()
if tracker_quality == 'UNCERTAIN':
    final_quality = detector_quality; final_score = score * 0.8
elif tracker_quality == 'HIGH_QUALITY' and detector_quality == 'HIGH':
    final_quality = 'HIGH'; final_score = min(0.95, score + 0.1)
elif tracker_quality == 'LOW_QUALITY' and detector_quality == 'LOW':
    final_quality = 'LOW'; final_score = 0.1
else:
    final_quality = 'LOW' if tracker_quality == 'LOW_QUALITY' else detector_quality
    final_score = score * 0.7
```

**Rationale**: v2 used detector-only classification. v3 combines both signals.
**Impact**: More robust scene quality assessment, especially in early epochs.

---

## Change 4: TCN Key Remapping with LayerNorm Support

**File**: `model/residual_feedback.py` — new `load_tcn_with_key_remapping()`

**Problem**: Old TCN state_dicts contain LayerNorm keys (ln1/ln2/ln3).
**Fix**: Added `nn.LayerNorm` layers after each Conv1d in SimpleTCN_v1.

```python
self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
self.ln1 = nn.LayerNorm(hidden_dim)          # NEW
self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=2)
self.ln2 = nn.LayerNorm(hidden_dim)          # NEW
self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=4, dilation=4)
self.ln3 = nn.LayerNorm(hidden_dim)          # NEW
```

**Result**: TCN loads for all 4 datasets: "TCN loaded (v1 old arch, input=63, hidden=64, output=20)"

---

## Change 5: make_fg_tcn_solver() Simplified

**File**: `model/residual_feedback.py`

```python
# v2: 4 lines loading MotionGeometryPredictor directly
# v3: 1 line
tcn_model = load_tcn_with_key_remapping(tcn_path)
```

**Impact**: Cleaner code, backward-compatible.

---

## Change 6: Frankfurt2 Diagnosis — Remove 500-Epoch Limit

**File**: `model/run_module3.py` — `run_frankfurt2_diagnosis()`

```python
# v2:
for i in range(min(total, 500)):  # only first 500 epochs

# v3:
for i in range(total):  # analyze all 3575 epochs
```

---

## Files Modified

| File | Lines Changed | Changes |
|------|:------------:|---------|
| `model/residual_feedback.py` | ~60 | Config, UNCERTAIN, tracker integration, TCN remap, LayerNorm |
| `model/run_module3.py` | 1 | Remove 500-epoch limit |

---

*Generated: 2026-06-07*

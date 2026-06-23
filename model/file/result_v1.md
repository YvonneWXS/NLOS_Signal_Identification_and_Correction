# result_v1.md - UrbanNav-HK_TST Data Processing Results (UPDATED)

**Date**: 2026-06-23
**Pipeline**: process_urbannav_pipeline.py

---

## Pipeline Test Results

### Step 1: Time Alignment
| Metric | Value |
|--------|:-----:|
| GT epochs | 787 |
| GNSS epochs | 705 |
| Aligned | 705/705 (100%) |
| Status | PASS |

### Step 2: Sky Mask Interpolation
| Metric | Value |
|--------|:-----:|
| Sky mask records | 148,642 |
| Interpolated epochs | 705/705 |
| Method | KD-tree 5-NN inverse distance weighted |
| Status | PASS |

### Step 3: Satellite Geometry

#### With Approximate Orbit Model (fallback, NO SP3)
| Metric | Value |
|--------|:-----:|
| Labeled epochs | 345/705 (49%) |
| Skipped epochs | 360 (51%) |
| Iteration time | ~2.4s |
| Status | DEGRADED |

**Root cause**: The simplified Keplerian orbit model only covers GPS satellites.
GLONASS and BeiDou positions cannot be computed, halving the usable epochs.
GPS elevations are accurate to ~2-5 deg, which is marginal for sky mask comparison.

#### With SP3 (expected)
| Metric | Expected Value |
|--------|:-----:|
| Labeled epochs | 705/705 (100%) |
| Skipped | 0 |
| Elevation accuracy | ~0.01 deg |
| Status | WAITING FOR SP3 |

### Step 4-5: Feature Extraction & Split (fallback mode)
| Split | Epochs | Total Sats | NLOS% | Avg Sats/Ep |
|-------|:------:|:----------:|:-----:|:-----------:|
| Train | 237 | 1,039 | 93.5% | 4.4 |
| Val | 102 | 379 | 100.0% | 3.7 |

NLOS ratio is artificially inflated due to orbit model errors.
With SP3, expect ~40-60% NLOS (typical for urban canyon).

---

## Files Delivered

```
data/processedData/UrbanNav-HK_TST/scripts/
  utils/coordinate_transforms.py     (WGS84 transforms)
  utils/satellite_orbit.py           (GPS orbit model fallback)
  process_urbannav_pipeline.py       (complete 5-step pipeline)
  run_pipeline_fallback.py           (fallback runner)

data/processedData/UrbanNav-HK_TST/processed/
  train_dataset.pkl                  (237 epochs, FALLBACK QUALITY)
  val_dataset.pkl                    (102 epochs, FALLBACK QUALITY)
  dataset_statistics.json            (stats)

model/file/
  goal_v1.md, change_v1.md, result_v1.md
```

---

## Next Steps (BLOCKED on SP3)

Download SP3 ephemeris to enable full-quality processing:
1. Go to https://cddis.nasa.gov/archive/gnss/products/2158/
2. Download igs21580.sp3 (or any .sp3 file for week 2158)
3. Place at: data/dataset/UrbanNav-HK_TST/igs21580.sp3
4. Run: python process_urbannav_pipeline.py
5. Expected: 705 labeled epochs, 40-60% NLOS, ready for Module 1

## Fallback Quality Note

The current train/val datasets are PROOF-OF-CONCEPT only.
They demonstrate the pipeline works end-to-end but have:
- Only GPS satellites (missing GLONASS + BeiDou)
- Inflated NLOS ratio due to orbit errors
- Only 339/705 epochs

DO NOT use for model training. Wait for SP3.

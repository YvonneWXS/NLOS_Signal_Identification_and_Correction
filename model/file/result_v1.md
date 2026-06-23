# result_v1.md - UrbanNav-HK_TST Data Processing Results

**Date**: 2026-06-23
**Pipeline**: process_urbannav_pipeline.py (Steps 1-2 verified, Steps 3-5 pending SP3)

---

## Step 1: Time Alignment

| Metric | Value |
|--------|:-----:|
| GT epochs | 787 |
| GNSS epochs | 705 |
| Aligned epochs | 705 (100%) |
| Skipped | 0 |

All 705 NovAtel GNSS epochs successfully matched to ground truth within 0.5s tolerance.

## Step 2: Sky Mask Interpolation

| Metric | Value |
|--------|:-----:|
| Sky mask records | 148,642 |
| KD-tree build time | < 1s |
| Interpolation method | 5-NN inverse distance weighting |
| Elevation mask dim | 361 (1 deg resolution) |
| Interpolated epochs | 705/705 (100%) |

## Dataset Composition (preliminary, without labels)

| System | Satellites seen | Notes |
|--------|:---:|-------|
| GPS (G) | 6 unique | L1 C/A + L2 |
| GLONASS (R) | 4 unique | L1 + L2 |
| BeiDou (C) | 8 unique | B1I + B2I |
| **Total** | **18 unique** | Per epoch: 3-8 visible |

## NLOS Labeling (pending SP3)

Current bottleneck: SP3 ephemeris for GPS Week 2158.

The pipeline is ready to generate NLOS labels once the SP3 file is placed at:
`data/dataset/UrbanNav-HK_TST/igs21580.sp3`

With SP3, the expected output:
- ~500 train epochs, ~205 validation epochs
- 11-dim features per satellite
- Undirected graph edges (azimuth difference < 90 deg)
- LOS/NLOS labels from sky mask comparison

## Files Created

```
data/processedData/UrbanNav-HK_TST/
  scripts/
    utils/coordinate_transforms.py    (WGS84 coordinate math)
    process_urbannav_pipeline.py      (5-step pipeline)
  processed/
    aligned_epochs.json               (from step 1)
    aligned_with_skymask.json         (from step 2)
  DATASET_README.md                   (dataset documentation)

model/file/
  goal_v1.md                          (task specification)
  change_v1.md                        (change log)
  result_v1.md                        (this file)
```

## Next Steps

1. User downloads SP3: `data/dataset/UrbanNav-HK_TST/igs21580.sp3`
2. Run: `python process_urbannav_pipeline.py`
3. Verify NLOS label quality vs elevation distribution
4. Fine-tune or test Module 1 GAT+MoG on the new dataset

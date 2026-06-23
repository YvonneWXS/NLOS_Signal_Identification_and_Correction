# change_v1.md - UrbanNav-HK_TST Data Processing Change Log

**Date**: 2026-06-23
**Goal**: goal_v1.md - Build complete data preprocessing pipeline for UrbanNav-HK_TST
**Status**: Pipeline code complete; awaiting SP3 ephemeris download

---

## Code Changes

### New Files

| File | Purpose |
|------|---------|
| data/processedData/UrbanNav-HK_TST/scripts/utils/coordinate_transforms.py | WGS84 coordinate transforms (LLA-ECEF, ENU, elev/az) |
| data/processedData/UrbanNav-HK_TST/scripts/process_urbannav_pipeline.py | Full 5-step pipeline script |
| data/processedData/UrbanNav-HK_TST/DATASET_README.md | Dataset documentation |
| model/file/goal_v1.md | Task specification |
| model/file/change_v1.md | This change log |
| model/file/result_v1.md | Results report |

### Pipeline Design

5 steps with intermediate saves:

| Step | Function | Input | Output | Verified |
|------|----------|-------|--------|:--------:|
| 1 | time_alignment | GT + GNSS obs | aligned_epochs.json (705 epochs) | YES |
| 2 | sky_mask_interp | aligned + sky_mask (KD-tree) | aligned_with_skymask.json | YES |
| 3 | satellite_geometry | SP3 ephemeris + skymask | nlos_labeled.json | PENDING SP3 |
| 4 | feature_extraction | labeled epochs | full_dataset.pkl (11-dim features) | PENDING |
| 5 | train_val_split | full dataset | train/val_dataset.pkl + stats | PENDING |

### Feature Design (11-dim, compatible with existing GAT+MoG)

| Index | Name | Normalization | Source |
|:---:|------|:---:|--------|
| 0 | elevation | /90.0 | SP3 + receiver position |
| 1 | azimuth | /360.0 | SP3 + receiver position |
| 2 | C/N0 | /60.0 | RINEX S1C/S2I |
| 3 | pr_stdev | /5.0 | Default 1.0m (no RINEX stdev) |
| 4 | pr_mes | /3e7 | RINEX C1C/C2I |
| 5 | pr_error_km | /100.0 | pr_mes - geo_range - clock_bias |
| 6 | cos(elevation) | -- | cos(elevation_rad) |
| 7-10 | constellation | one-hot | GPS/GLO/GAL/BDS |

### Key Design Decisions

1. **SP3 over broadcast**: Higher accuracy (cm-level satellite positions)
2. **Temporal split** (70/30): Preserves trajectory order for realistic evaluation
3. **KD-tree sky mask interpolation**: Efficient 5-NN inverse-distance weighting
4. **Clock bias removal**: Per-epoch mean subtraction (crude but effective)
5. **pr_stdev = 1.0m default**: UrbanNav RINEX lacks formal uncertainty estimates

### SP3 Download Required

- GPS Week: 2158 (2021-05-17)
- URL: https://cddis.nasa.gov/archive/gnss/products/2158/
- Target: data/dataset/UrbanNav-HK_TST/igs21580.sp3

---

## Usage

```bash
# After downloading SP3:
cd data/processedData/UrbanNav-HK_TST/scripts
python process_urbannav_pipeline.py
```

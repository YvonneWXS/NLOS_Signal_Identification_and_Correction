# result_v1.md - UrbanNav-HK_TST Data Processing Results (FINAL)

**Date**: 2026-06-23
**Pipeline**: run_final.py (GPS-only IGS SP3)

---

## Final Results

### Dataset Summary

| Metric | Value |
|--------|:-----:|
| Total labeled epochs | 327 |
| Train set | 228 epochs (70%) |
| Validation set | 99 epochs (30%) |
| Total satellites | 1,042 |
| NLOS ratio | 17.2% |
| Average sats/epoch | 3.2 |
| Systems used | GPS only |
| SP3 source | IGS final (igs21581.sp3) |

### Quality Validation

**Elevation vs NLOS** (correct physical pattern):
| Elevation | Satellites | NLOS Rate |
|-----------|:----------:|:---------:|
| 15-30 deg | 138 | 60.1% |
| 30-45 deg | 43 | 7.0% |
| 45-60 deg | 611 | 5.7% |
| 60-75 deg | 250 | 23.2% |

NLOS rate decreases with elevation as expected.

**C/N0 separation**:
| Label | Mean C/N0 | Std |
|-------|:---------:|:---:|
| LOS | 38.8 dB-Hz | 3.0 |
| NLOS | 33.7 dB-Hz | 2.5 |

5.1 dB gap confirms effective LOS/NLOS discrimination.

**Graph structure**: Mean 3.1 edges/epoch (azimuth-difference < 90 deg connectivity).

### Known Limitations

1. **GPS-only**: SP3 (IGS final) only provides GPS positions. GLONASS and BeiDou satellites are dropped.
2. **Small dataset**: 327 epochs vs. 705 in the original GNSS file (46% utilization).
3. **No Galileo/QZSS**: Neither observed by NovAtel nor in SP3.
4. **pr_stdev = 1.0m default**: RINEX format lacks formal uncertainty estimates.

### Improvement Path

Download MGEX SP3 (gbm21581.sp3) from CDDIS for full GPS+GLONASS+BeiDou+Galileo coverage:
```
https://cddis.nasa.gov/archive/gnss/products/mgex/2158/gbm21581.sp3.Z
```
Expected with MGEX: ~600+ epochs, 6-10 sats/epoch, 30-40% NLOS.

### Files

```
data/processedData/UrbanNav-HK_TST/processed/
  train_dataset.pkl        (228 epochs, 750 satellites)
  val_dataset.pkl          (99 epochs, 292 satellites)
  dataset_statistics.json  (summary statistics)
  
data/processedData/UrbanNav-HK_TST/scripts/
  utils/coordinate_transforms.py
  utils/satellite_orbit.py
  process_urbannav_pipeline.py
  run_final.py
```

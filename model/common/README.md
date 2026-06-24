# Common Library

## 1. Module Overview

Shared utilities used by all five modules: coordinate transforms, SP3 ephemeris reading, GPS time conversion, evaluation metrics, logging, and YAML configuration management.

### Core Flow
`
GPS Time (week+sec) -> SP3 interpolated position -> ECEF
LLA (lat/lon/height) -> ECEF <-> ENU
Error arrays -> CEP50, CEP95, RMSE, MAE, STD, median
`

## 2. Architecture

| File | Function |
|------|----------|
| coordinate.py | LLA-ECEF, ECEF-ENU, azimuth/elevation computation |
| sp3_reader.py | SP3 precise ephemeris parsing and interpolation |
| time_utils.py | GPS week/second <-> datetime conversion |
| metrics.py | CEP50, CEP95, RMSE, MAE, STD, median, all_metrics() |
| logger.py | Unified logging (console + file + TensorBoard) |
| config_manager.py | YAML config loading, merging, CLI override, validation |

## 3. Configuration

No standalone config. Functions accept numpy arrays directly.

## 4. Usage

`python
from common.coordinate import lla_to_ecef, ecef_to_lla
from common.metrics import cep50, cep95, all_metrics
from common.sp3_reader import SP3Reader

# SP3
reader = SP3Reader('igs21580.sp3')
pos = reader.get_satellite_position(gps_week, gps_sec, 'G01')

# Coordinates
ecef = lla_to_ecef(lat_deg, lon_deg, height_m)

# Metrics
metrics = all_metrics(errors_km)  # dict with cep50, cep95, rmse, mae, std, median
`

## 5. API Reference

### coordinate.py
- lla_to_ecef(lat, lon, height) -> (x, y, z) in km
- ecef_to_lla(x, y, z) -> (lat, lon, height)
- ecef_to_enu(ref_ecef, target_ecef) -> (e, n, u)
- compute_azimuth_elevation(rx_ecef, sv_ecef) -> (az_deg, el_deg)

### metrics.py
- cep50(errors_km) -> float
- cep95(errors_km) -> float
- rmse(errors_km) -> float
- all_metrics(errors_km) -> dict

### sp3_reader.py
- SP3Reader(filepath): constructor
- has_satellite(svid) -> bool
- get_satellite_position(gps_week, gps_sec, svid) -> (x, y, z) in meters

## 6. Dependencies

- numpy, scipy, pyyaml
- No PyTorch dependency

## 7. Tests

`ash
pytest common/tests/ -v  # 15 tests, all pass
`

## 8. FAQ

**Q: Why are coordinates in km?**
A: Matches pseudorange measurements in km. Avoids numerical scaling issues in optimization.

**Q: SP3 reader is slow?**
A: File is parsed once on first call. Subsequent lookups are O(log n) binary search.

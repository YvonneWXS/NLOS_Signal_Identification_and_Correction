# common/coordinate.py — Coordinate transforms (WGS84)
import numpy as np

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2


def lla_to_ecef(lat_deg, lon_deg, height_m):
    """WGS84 LLA -> ECEF (km)"""
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat); cos_lat = np.cos(lat)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    x = (N + height_m) * cos_lat * np.cos(lon)
    y = (N + height_m) * cos_lat * np.sin(lon)
    z = (N * (1.0 - _WGS84_E2) + height_m) * sin_lat
    return np.array([x, y, z]) / 1000.0


def ecef_to_lla(x_km, y_km, z_km):
    """ECEF (km) -> WGS84 LLA (deg, deg, m)"""
    x, y, z = x_km * 1000.0, y_km * 1000.0, z_km * 1000.0
    lon = np.arctan2(y, x); p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(5):
        sin_lat = np.sin(lat); N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - _WGS84_E2 * N / (N + h)))
    sin_lat = np.sin(lat); N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    h = p / np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), h


def ecef_to_enu(ref_ecef_km, target_ecef_km):
    """Convert ECEF difference to local ENU (East-North-Up) frame (km).
    ref_ecef_km: (3,) reference ECEF position
    target_ecef_km: (N,3) or (3,) target ECEF positions
    Returns: ENU coordinates in km
    """
    ref = np.asarray(ref_ecef_km).flatten()
    tgt = np.atleast_2d(target_ecef_km) if target_ecef_km.ndim > 1 else np.atleast_2d(target_ecef_km)
    lat, lon, _ = ecef_to_lla(ref[0], ref[1], ref[2])
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    sin_lon, cos_lon = np.sin(lon_r), np.cos(lon_r)
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ])
    diff = tgt - ref
    enu = diff @ R.T
    return enu.squeeze()


def compute_azimuth_elevation(rx_ecef_km, sv_ecef_km):
    """Compute azimuth (deg) and elevation (deg) from receiver to satellite.
    rx_ecef_km: (3,) receiver ECEF position
    sv_ecef_km: (3,) or (N,3) satellite ECEF position(s)
    """
    enu = ecef_to_enu(rx_ecef_km, sv_ecef_km)
    if enu.ndim == 1:
        enu = enu.reshape(1, 3)
    e, n, u = enu[:, 0], enu[:, 1], enu[:, 2]
    horizontal = np.sqrt(e**2 + n**2)
    elevation = np.rad2deg(np.arctan2(u, horizontal))
    azimuth = np.rad2deg(np.arctan2(e, n)) % 360
    if len(elevation) == 1:
        return float(azimuth[0]), float(elevation[0])
    return azimuth, elevation


_GNSS_ID_MAP = {0: 'GPS', 1: 'GPS', 2: 'Galileo', 3: 'Glonass', 4: 'BeiDou', 5: 'QZSS', 6: 'Galileo'}
_GNSS_TO_SP3 = {'GPS': 'G', 'Glonass': 'R', 'Galileo': 'E', 'BeiDou': 'C'}
_GNSS_TO_INDEX = {'GPS': 7, 'Glonass': 8, 'Galileo': 9, 'BeiDou': 10}

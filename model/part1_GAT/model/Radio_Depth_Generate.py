"""
卫星几何计算模块
======================================
功能: ECEF/经纬度转换、仰角/方位角计算
输入: ECEF坐标 (x,y,z) 或 经纬度高度
输出: 仰角(elevation), 方位角(azimuth), 几何距离
"""

import numpy as np
from typing import Tuple

# WGS84 椭球参数
WGS84_A = 6378137.0           # 长半轴 (m)
WGS84_F = 1.0 / 298.257223563 # 扁率
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # 第一偏心率平方


def lla_to_ecef(lat: float, lon: float, h: float) -> Tuple[float, float, float]:
    """WGS84 经纬度高度 -> ECEF 地心地固坐标"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)
    x = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + h) * np.sin(lat_rad)
    return (x, y, z)


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ECEF 地心地固坐标 -> WGS84 经纬度高度"""
    lon = np.arctan2(y, x)
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - WGS84_E2))

    for _ in range(5):  # 迭代求解
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + h)))

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return (np.degrees(lat), np.degrees(lon), h)


def compute_geometry(rcv_ecef: Tuple[float, float, float],
                     sv_ecef: Tuple[float, float, float]) -> Tuple[float, float]:
    """计算卫星仰角和方位角 (度)"""
    rx = np.array(rcv_ecef, dtype=np.float64)
    sx = np.array(sv_ecef, dtype=np.float64)
    los = sx - rx
    range_m = float(np.linalg.norm(los))

    lat, lon, _ = ecef_to_lla(*rcv_ecef)
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)

    # ECEF -> ENU 旋转矩阵
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    enu = R @ los
    e, n, u = enu[0], enu[1], enu[2]

    elevation = float(np.degrees(np.arcsin(np.clip(u / range_m, -1, 1))))
    azimuth = float(np.degrees(np.arctan2(e, n)))
    if azimuth < 0:
        azimuth += 360.0

    return (elevation, azimuth)


def compute_geometric_range(rcv_ecef: Tuple[float, float, float],
                            sv_ecef: Tuple[float, float, float]) -> float:
    """计算接收机到卫星的几何距离 (m)"""
    return float(np.linalg.norm(np.array(sv_ecef) - np.array(rcv_ecef)))


# 测试代码
if __name__ == '__main__':
    # 验证: 柏林 Potsdamer Platz (52.5095°N, 13.3693°E)
    lat, lon, h = 52.5095, 13.3693, 35.0
    rcv_ecef = lla_to_ecef(lat, lon, h)
    lat2, lon2, h2 = ecef_to_lla(*rcv_ecef)
    print(f"LLA -> ECEF -> LLA roundtrip:")
    print(f"  Input:  lat={lat:.6f}, lon={lon:.6f}, h={h:.2f}")
    print(f"  Output: lat={lat2:.6f}, lon={lon2:.6f}, h={h2:.2f}")
    print(f"  Diff:   lat={abs(lat-lat2):.2e}, lon={abs(lon-lon2):.2e}, h={abs(h-h2):.2e}")

    # 验证: 模拟一颗GPS卫星
    sv_ecef = (22162405.973, 13363648.127, 6427957.993)  # 来自 gbm19001.sp3
    elev, az = compute_geometry(rcv_ecef, sv_ecef)
    geo_range = compute_geometric_range(rcv_ecef, sv_ecef)
    print(f"\nGPS satellite geometry:")
    print(f"  Elevation: {elev:.2f} deg")
    print(f"  Azimuth:   {az:.2f} deg")
    print(f"  Range:     {geo_range / 1000:.2f} km")

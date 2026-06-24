"""
SP3精密星历解析器
======================================
功能: 解析 SP3 精密星历文件，获取卫星 ECEF 位置
输入: .sp3 文件路径
输出: SP3Reader 对象，支持按 GPS 时间查询卫星位置
"""

from typing import Dict, Tuple, Optional
from datetime import datetime
import math


class SP3Reader:
    """SP3星历读取器 — 获取卫星ECEF位置和钟差"""

    OMEGA_EARTH = 7.2921151467e-5  # 地球自转角速度 (rad/s)
    C = 299792458.0               # 光速 (m/s)

    def __init__(self, sp3_path: str):
        self.sp3_path = sp3_path
        self.epochs: Dict[Tuple[int, float], Dict[str, Tuple[float, float, float, float]]] = {}
        self.reference_week = None
        self.reference_sec = None
        self._parse()
        if self.epochs:
            self.reference_week, self.reference_sec = list(self.epochs.keys())[0]

    def _parse(self):
        """解析 SP3 文件"""
        current_epoch = None
        current_svs = {}
        prefix_types = set()

        with open(self.sp3_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith('EOF'):
                if current_epoch is not None:
                    self.epochs[current_epoch] = current_svs
                break

            if line.startswith('+ ') or line.startswith('++') or line.startswith('%'):
                continue

            # Epoch header — starts with "* " (star + space)
            if line.startswith('* '):
                if current_epoch is not None:
                    self.epochs[current_epoch] = current_svs
                parts = line.split()
                if len(parts) >= 7:
                    year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
                    hour, minute = int(parts[4]), int(parts[5])
                    sec = float(parts[6])
                    gps_week, gps_sec = self._date_to_gps_time(year, month, day, hour, minute, sec)
                    current_epoch = (gps_week, gps_sec)
                    current_svs = {}
                continue

            # Position line — starts with P followed by satellite ID (non-space)
            if line.startswith('P') and len(line) > 1 and line[1] != ' ' and current_epoch is not None:
                parts = line.split()
                if len(parts) >= 4:
                    sv_id = parts[0][1:]  # strip 'P' prefix, e.g. 'PG01' -> 'G01'
                    x = float(parts[1]) * 1000.0  # km -> m
                    y = float(parts[2]) * 1000.0
                    z = float(parts[3]) * 1000.0
                    clock_us = float(parts[4]) if len(parts) >= 5 else 0.0
                    current_svs[sv_id] = (x, y, z, clock_us)
                    prefix_types.add(parts[0][:2])  # e.g. 'PG', 'PC', 'PR', 'PE'

        if current_epoch is not None:
            self.epochs[current_epoch] = current_svs

        # Build sorted epoch list for interpolation
        self._sorted_epochs = sorted(self.epochs.keys(), key=lambda k: (k[0], k[1]))

        # Store diagnostic info
        self._prefix_types = prefix_types
        self._total_epochs = len(self.epochs)
        self._total_sats = len(set().union(*[set(svs.keys()) for svs in self.epochs.values()])) if self.epochs else 0

    def _date_to_gps_time(self, year, month, day, hour, minute, sec):
        """日期 -> GPS 时间 (week, seconds)"""
        try:
            dt = datetime(year, month, day, hour, minute, int(sec))
        except (ValueError, OverflowError):
            dt = datetime(year, month, day, hour, minute, 0)
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        delta = dt - gps_epoch
        total_sec = delta.total_seconds()
        gps_week = int(total_sec // (7 * 24 * 3600))
        gps_sec = total_sec % (7 * 24 * 3600)
        return gps_week, gps_sec

    def _rotate_for_earth_rotation(self, x: float, y: float, z: float,
                                   delta_t: float) -> Tuple[float, float, float]:
        """绕 Z 轴旋转坐标以考虑地球自转"""
        theta = self.OMEGA_EARTH * delta_t
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return (x * cos_t - y * sin_t, x * sin_t + y * cos_t, z)

    def _get_interpolated(self, gps_week: int, gps_sec: float,
                          sv_id: str) -> Optional[Tuple[float, float, float, float]]:
        """获取卫星在指定时刻的 ECEF 位置+钟差（线性插值）

        Returns (x, y, z, clock_us) or None.
        """
        # Find bracketing epochs for interpolation
        before_idx = None
        after_idx = None

        for i, et in enumerate(self._sorted_epochs):
            if et[0] > gps_week or (et[0] == gps_week and et[1] > gps_sec):
                after_idx = i
                before_idx = i - 1 if i > 0 else None
                break
            if et[0] == gps_week and et[1] == gps_sec:
                if sv_id in self.epochs[et]:
                    return self.epochs[et][sv_id]
                return None

        if after_idx is None:
            before_idx = len(self._sorted_epochs) - 1

        if before_idx is not None and before_idx >= 0 and after_idx is not None and after_idx < len(self._sorted_epochs):
            et_before = self._sorted_epochs[before_idx]
            et_after = self._sorted_epochs[after_idx]

            if sv_id in self.epochs[et_before] and sv_id in self.epochs[et_after]:
                x1, y1, z1, c1 = self.epochs[et_before][sv_id]
                x2, y2, z2, c2 = self.epochs[et_after][sv_id]

                t_before = et_before[0] * 604800 + et_before[1]
                t_after = et_after[0] * 604800 + et_after[1]
                t_query = gps_week * 604800 + gps_sec

                alpha = (t_query - t_before) / max(t_after - t_before, 1e-9)
                alpha = max(0.0, min(1.0, alpha))

                x = x1 + alpha * (x2 - x1)
                y = y1 + alpha * (y2 - y1)
                z = z1 + alpha * (z2 - z1)
                clock_us = c1 + alpha * (c2 - c1)

                return (x, y, z, clock_us)

            if sv_id in self.epochs[et_before]:
                return self.epochs[et_before][sv_id]
            if sv_id in self.epochs[et_after]:
                return self.epochs[et_after][sv_id]
            return None

        if before_idx is not None and before_idx >= 0:
            et = self._sorted_epochs[before_idx]
            if sv_id in self.epochs[et]:
                return self.epochs[et][sv_id]

        return None

    def get_satellite_position(self, gps_week: int, gps_sec: float,
                               sv_id: str) -> Optional[Tuple[float, float, float]]:
        """获取卫星在指定时刻的 ECEF 位置（线性插值）"""
        result = self._get_interpolated(gps_week, gps_sec, sv_id)
        if result is None:
            return None
        return (result[0], result[1], result[2])

    def get_satellite_clock(self, gps_week: int, gps_sec: float,
                            sv_id: str) -> Optional[float]:
        """获取卫星钟差 (meters)。返回 clock_us * 1e-6 * C。

        GPS伪距: PR = ρ + c·dt_r - c·dt_s, 其中 dt_s 为卫星钟差。
        SP3 clock 值即为 dt_s，所以校正: PR_corrected = PR_measured + clock_correction_m
        """
        result = self._get_interpolated(gps_week, gps_sec, sv_id)
        if result is None:
            return None
        return result[3] * 1e-6 * self.C  # μs → meters

    def has_satellite(self, sv_id: str) -> bool:
        """检查 SP3 是否包含指定卫星"""
        for svs in self.epochs.values():
            if sv_id in svs:
                return True
        return False

    def get_statistics(self) -> dict:
        """获取解析统计信息（用于验证）"""
        stats = {
            "file": self.sp3_path,
            "total_epochs": self._total_epochs,
            "total_satellites": self._total_sats,
            "prefix_types": sorted(self._prefix_types),
            "reference_week": self.reference_week,
            "reference_sec": self.reference_sec,
        }
        if self.epochs:
            first_epoch = list(self.epochs.keys())[0]
            first_positions = list(self.epochs[first_epoch].values())
            if first_positions:
                xs = [p[0] for p in first_positions]
                ys = [p[1] for p in first_positions]
                zs = [p[2] for p in first_positions]
                stats["position_range_x"] = (min(xs), max(xs))
                stats["position_range_y"] = (min(ys), max(ys))
                stats["position_range_z"] = (min(zs), max(zs))
                stats["position_magnitude_km"] = (
                    math.sqrt(min(xs)**2 + min(ys)**2 + min(zs)**2) / 1000.0,
                    math.sqrt(max(xs)**2 + max(ys)**2 + max(zs)**2) / 1000.0
                )
        return stats

    def __repr__(self):
        return (f"SP3Reader(file={self.sp3_path}, "
                f"epochs={self._total_epochs}, sats={self._total_sats})")

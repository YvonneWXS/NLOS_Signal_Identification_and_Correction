"""
数据加载与预处理模块
======================================
功能: GNSS CSV 加载 + SP3 解析 + 时间同步 + 伪距误差计算
输入: 数据集名称, Config 对象
输出: [EpochData] 列表, 支持 .pkl 缓存

数据流:
  RXM-RAWX.csv ─┐
                ├─→ groupby (GPSWeek, GPSSeconds) → 每历元一组卫星测量
  NAV-POSLLH.csv┘

  每组测量行 ─→ 时间同步匹配 GT 位置 (最近邻, 容忍度 1.0s)
            ─→ SP3 查找卫星 ECEF 位置
            ─→ 计算仰角/方位角 + 几何距离
            ─→ 伪距误差 = (prMes - geometric_range) / 1000 (m→km)
            ─→ 误差截断 ±100 km
            ─→ 每历元去均值 (消除接收机钟差)
            ─→ 输出: [EpochData] + 缓存 .pkl
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from config import Config, get_config

# SP3 和几何模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp3_reader import SP3Reader
from Radio_Depth_Generate import lla_to_ecef, compute_geometry, compute_geometric_range


# ==================== 数据类型定义 ====================

@dataclass
class GNSSObservation:
    """单颗卫星单历元的观测数据"""
    gps_week: int
    gps_seconds: float
    gnss_id: str           # 'GPS', 'Glonass', 'Galileo', 'BeiDou'
    sv_id: int
    pr_mes: float           # 观测伪距 (m)
    cno: float             # 载噪比 (dBHz)
    pr_stdev: float         # 伪距标准差 (m)
    nlos_label: int        # NLOS 标注 (0=LOS, 1=NLOS)
    elevation: float       # 卫星仰角 (deg)
    azimuth: float          # 卫星方位角 (deg)
    pseudorange_error: float  # 去均值后的伪距误差 (km)


@dataclass
class EpochData:
    """单历元数据"""
    gps_week: int
    gps_seconds: float
    gt_lat: float          # 真值纬度
    gt_lon: float         # 真值经度
    gt_height: float       # 真值高度
    observations: List[GNSSObservation]


# ==================== GNSS-ID → SP3-ID 映射 ====================

_GNSS_TO_SP3_PREFIX = {
    'GPS': 'G',
    'Glonass': 'R',
    'Galileo': 'E',
    'BeiDou': 'C',
}


def _to_sp3_svid(gnss_id: str, sv_id: int) -> str:
    """GNSS ID → SP3 卫星 ID (e.g., ('GPS', 1) → 'G01')"""
    prefix = _GNSS_TO_SP3_PREFIX.get(gnss_id, '')
    if prefix:
        return f'{prefix}{sv_id:02d}'
    return f'{sv_id:02d}'


# ==================== CSV 加载 ====================

def _load_csv(data_dir: str, filename: str) -> Optional[pd.DataFrame]:
    """加载 CSV 文件 (自动检测分隔符)"""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    for sep in [';', ',']:
        try:
            df = pd.read_csv(path, sep=sep, header=0)
            return df
        except Exception:
            continue
    return None


def _find_column(cols: List[str], *candidates: str) -> Optional[str]:
    """模糊列名匹配"""
    for col in cols:
        col_lower = col.lower().replace(' ', '')
        for cand in candidates:
            if cand.lower().replace(' ', '') in col_lower:
                return col
    return None


def _load_sp3(data_dir: str) -> Optional[SP3Reader]:
    """加载 SP3 星历文件"""
    for f in os.listdir(data_dir):
        if f.endswith('.sp3') and not f.startswith('.'):
            try:
                return SP3Reader(os.path.join(data_dir, f))
            except Exception:
                return None
    return None


# ==================== Pickle 类型注册 ====================

def _register_pickle_types():
    """注册自定义类型到 __main__，确保跨模块 pickle 加载兼容"""
    import __main__
    if not hasattr(__main__, 'GNSSObservation'):
        __main__.GNSSObservation = GNSSObservation
    if not hasattr(__main__, 'EpochData'):
        __main__.EpochData = EpochData

# ==================== 数据预处理核心 ====================

def load_and_process_dataset(dataset_name: str, config: Config = None,
                             force_reprocess: bool = False) -> List[EpochData]:
    """
    加载并预处理单个数据集

    Args:
        dataset_name: 数据集名称
        config: 配置对象
        force_reprocess: 是否强制重新处理

    Returns:
        处理后的 EpochData 列表
    """
    if config is None:
        config = get_config()

    data_dir = config.get_data_dir(dataset_name)

    # 1. 检查缓存
    if not force_reprocess:
        cache_path = config.get_processed_data_path(dataset_name)
        if os.path.exists(cache_path):
            print(f"[{dataset_name}] Loading cached data from: {cache_path}")
            try:
                # 确保 pickle 能解析类型（处理不同入口点导致的 __main__ 引用问题）
                _register_pickle_types()
                with open(cache_path, 'rb') as f:
                    epochs = pickle.load(f)
                print(f"[{dataset_name}] Loaded {len(epochs)} epochs from cache")
                return epochs
            except Exception as e:
                print(f"[{dataset_name}] Cache load failed: {e}, reprocessing...")

    # 2. 加载原始数据
    print(f"[{dataset_name}] Loading raw data...")
    rxm_df = _load_csv(data_dir, 'RXM-RAWX.csv')
    nav_df = _load_csv(data_dir, 'NAV-POSLLH.csv')

    if rxm_df is None or nav_df is None:
        print(f"[{dataset_name}] ERROR: Missing data files")
        return []

    sp3_reader = _load_sp3(data_dir)
    if sp3_reader is None:
        print(f"[{dataset_name}] WARNING: No SP3 file found, geometry will be estimated")

    # 3. 列名匹配
    rxm_cols = rxm_df.columns.tolist()
    nav_cols = nav_df.columns.tolist()

    rxm_week_col = _find_column(rxm_cols, 'GPSWeek', 'week')
    rxm_sec_col = _find_column(rxm_cols, 'GPSSeconds', 'tow')
    gnss_id_col = _find_column(rxm_cols, 'gnssId', 'gnssidentifier')
    sv_id_col = _find_column(rxm_cols, 'svId', 'satelliteidentifier')
    pr_col = _find_column(rxm_cols, 'prMes', 'pseudorange')
    cno_col = _find_column(rxm_cols, 'cno', 'carrier-to-noise')
    pr_std_col = _find_column(rxm_cols, 'prStdev', 'stdev')
    nlos_col = _find_column(rxm_cols, 'NLOS')

    nav_week_col = _find_column(nav_cols, 'GPSWeek', 'week')
    nav_sec_col = _find_column(nav_cols, 'GPSSeconds', 'tow')
    gt_lon_col = _find_column(nav_cols, 'GT Lon', 'gtlon')
    gt_lat_col = _find_column(nav_cols, 'GT Lat', 'gtlat')
    gt_h_col = _find_column(nav_cols, 'GT Height', 'gtheight', 'height')

    if not all([rxm_week_col, rxm_sec_col, nav_week_col, nav_sec_col, gt_lat_col, gt_lon_col]):
        print(f"[{dataset_name}] ERROR: Cannot find required time/GPS columns")
        return []

    # 4. 构建 GT 位置时间索引
    nav_times: Dict[Tuple[int, float], Dict[str, float]] = {}
    for _, row in nav_df.iterrows():
        try:
            week = int(row[nav_week_col])
            sec = float(row[nav_sec_col])
            nav_times[(week, sec)] = {
                'lat': float(row[gt_lat_col]),
                'lon': float(row[gt_lon_col]),
                'height': float(row[gt_h_col]) if gt_h_col else 0.0
            }
        except (ValueError, TypeError):
            continue

    print(f"[{dataset_name}] GT positions: {len(nav_times)} entries")

    # 5. 按历元分组处理
    group_cols = [rxm_df[rxm_week_col], rxm_df[rxm_sec_col]]
    rxm_grouped = rxm_df.groupby(
        [rxm_df[rxm_week_col], rxm_df[rxm_sec_col]],
        sort=False
    )

    epochs: List[EpochData] = []
    tol = config.TIME_SYNC_TOLERANCE
    skipped_no_gt = 0
    skipped_no_sv = 0

    for (gps_week, gps_sec), group in rxm_grouped:
        # 5a. 时间同步：查找最近 GT 位置
        best_gt = _find_best_gt(gps_week, gps_sec, nav_times, tol)
        if best_gt is None:
            skipped_no_gt += 1
            continue

        # 5b. 提取每颗卫星的观测
        observations: List[GNSSObservation] = []
        for _, row in group.iterrows():
            try:
                obs = _extract_observation(row, gps_week, gps_sec, best_gt,
                                          sp3_reader, gnss_id_col, sv_id_col,
                                          pr_col, cno_col, pr_std_col, nlos_col)
                if obs is not None:
                    observations.append(obs)
            except Exception:
                continue

        if not observations:
            skipped_no_sv += 1
            continue

        # 5c. 每历元去均值 (消除接收机钟差) + 温和截断
        error_mean = np.mean([o.pseudorange_error for o in observations])

        normalized_obs = []
        for obs in observations:
            demeaned_error = obs.pseudorange_error - error_mean
            # Gentle clip after demeaning (±10 km) — receiver clock bias now removed
            demeaned_error = float(np.clip(demeaned_error, -10.0, 10.0))
            normalized_obs.append(GNSSObservation(
                gps_week=obs.gps_week,
                gps_seconds=obs.gps_seconds,
                gnss_id=obs.gnss_id,
                sv_id=obs.sv_id,
                pr_mes=obs.pr_mes,
                cno=obs.cno,
                pr_stdev=obs.pr_stdev,
                nlos_label=obs.nlos_label,
                elevation=obs.elevation,
                azimuth=obs.azimuth,
                pseudorange_error=demeaned_error
            ))

        epochs.append(EpochData(
            gps_week=gps_week,
            gps_seconds=gps_sec,
            gt_lat=best_gt['lat'],
            gt_lon=best_gt['lon'],
            gt_height=best_gt['height'],
            observations=normalized_obs
        ))

    print(f"[{dataset_name}] Processed {len(epochs)} epochs "
          f"(skipped: {skipped_no_gt} no-GT, {skipped_no_sv} no-SV)")

    # 6. 保存缓存
    if epochs:
        config.ensure_dirs()
        cache_path = config.get_processed_data_path(dataset_name)
        with open(cache_path, 'wb') as f:
            pickle.dump(epochs, f)
        print(f"[{dataset_name}] Saved {len(epochs)} epochs to cache")

    return epochs


def _find_best_gt(gps_week: int, gps_sec: float,
                  nav_times: Dict[Tuple[int, float], Dict],
                  tolerance: float) -> Optional[Dict]:
    """时间同步：查找最近的 GT 位置"""
    best_key = None
    best_diff = float('inf')
    for key in nav_times:
        if key[0] == gps_week:
            diff = abs(key[1] - gps_sec)
            if diff < best_diff and diff <= tolerance:
                best_diff = diff
                best_key = key
    return nav_times[best_key] if best_key else None


def _extract_observation(row, gps_week: int, gps_sec: float,
                         gt_pos: Dict, sp3_reader: Optional[SP3Reader],
                         gnss_id_col: str, sv_id_col: str, pr_col: str,
                         cno_col: Optional[str], pr_std_col: Optional[str],
                         nlos_col: Optional[str]) -> Optional[GNSSObservation]:
    """提取单个卫星观测"""
    gnss_id = str(row[gnss_id_col]).strip()
    sv_id = int(row[sv_id_col])

    # 检查必要列
    if pd.isna(row[pr_col]):
        return None

    pr_raw = float(row[pr_col])
    cno = float(row[cno_col]) if cno_col and not pd.isna(row.get(cno_col, np.nan)) else 40.0
    pr_stdev = float(row[pr_std_col]) if pr_std_col and not pd.isna(row.get(pr_std_col, np.nan)) else 0.5
    nlos_label = int(row[nlos_col]) if nlos_col and not pd.isna(row.get(nlos_col, np.nan)) else 0

    # 计算几何信息
    elevation, azimuth, pseudorange_error = 45.0, 180.0, 0.0
    pr_stored = pr_raw  # default: raw measurement (when no SP3)

    if sp3_reader:
        sv_str = _to_sp3_svid(gnss_id, sv_id)
        sv_ecef = sp3_reader.get_satellite_position(gps_week, gps_sec, sv_str)

        if sv_ecef:
            rcv_ecef = lla_to_ecef(gt_pos['lat'], gt_pos['lon'], gt_pos['height'])
            elevation, azimuth = compute_geometry(rcv_ecef, sv_ecef)
            geometric_range = compute_geometric_range(rcv_ecef, sv_ecef)
            # Apply satellite clock correction (SP3 clock → meters)
            clock_correction_m = sp3_reader.get_satellite_clock(gps_week, gps_sec, sv_str) or 0.0
            pr_stored = pr_raw + clock_correction_m
            # 伪距误差 (m → km) — clip deferred to after per-epoch demeaning
            error_km = (pr_stored - geometric_range) / 1000.0
            pseudorange_error = error_km
        else:
            # Satellite not found in SP3 — skip entirely (can't compute geometry or clock)
            return None

    return GNSSObservation(
        gps_week=gps_week,
        gps_seconds=gps_sec,
        gnss_id=gnss_id,
        sv_id=sv_id,
        pr_mes=pr_stored,
        cno=cno,
        pr_stdev=pr_stdev,
        nlos_label=nlos_label,
        elevation=elevation,
        azimuth=azimuth,
        pseudorange_error=pseudorange_error
    )


# ==================== 数据统计 ====================

def print_data_statistics(epochs: List[EpochData], dataset_name: str = ""):
    """打印数据集统计信息"""
    if not epochs:
        print(f"[{dataset_name}] No data")
        return

    all_labels = []
    all_errors = []
    sat_counts = []

    for ep in epochs:
        for obs in ep.observations:
            all_labels.append(obs.nlos_label)
            all_errors.append(obs.pseudorange_error)
        sat_counts.append(len(ep.observations))

    all_labels = np.array(all_labels)
    all_errors = np.array(all_errors)
    sat_counts = np.array(sat_counts)

    print(f"\n{'='*50}")
    print(f"[{dataset_name}] Data Statistics")
    print(f"{'='*50}")
    print(f"  Total epochs:       {len(epochs)}")
    print(f"  Total observations: {len(all_labels)}")
    print(f"  Sats per epoch:     {sat_counts.mean():.1f} ± {sat_counts.std():.1f} "
          f"[{sat_counts.min()}-{sat_counts.max()}]")
    print(f"  LOS count:          {np.sum(all_labels == 0)} ({np.mean(all_labels == 0)*100:.1f}%)")
    print(f"  NLOS count:         {np.sum(all_labels == 1)} ({np.mean(all_labels == 1)*100:.1f}%)")
    print(f"  Error mean:         {np.mean(all_errors):.4f} km")
    print(f"  Error std:          {np.std(all_errors):.4f} km")
    print(f"  Error range:        [{np.min(all_errors):.4f}, {np.max(all_errors):.4f}] km")
    print(f"{'='*50}\n")


# ==================== 测试入口 ====================

if __name__ == '__main__':
    config = get_config()
    config.ensure_dirs()

    # 测试 berlin1 数据集
    dataset = config.DATASETS[0]
    print(f"\nProcessing {dataset}...")

    epochs = load_and_process_dataset(dataset, config, force_reprocess=False)
    print_data_statistics(epochs, dataset)

import os
import glob
import pandas as pd
import numpy as np
import datetime
from scipy.interpolate import interp1d
from tqdm import tqdm
# --- 物理常数 ---
C_LIGHT = 299792458.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
def llh_to_ecef_vectorized(lat_deg, lon_deg, h_m):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    e2 = WGS84_F * (2 - WGS84_F)
    sin_lat = np.sin(lat_rad)
    N = WGS84_A / np.sqrt(1 - e2 * sin_lat ** 2)
    x = (N + h_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + h_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + h_m) * sin_lat
    return x, y, z
def ecef_to_enu_vectorized(x, y, z, ref_lat, ref_lon, ref_alt):
    ref_x, ref_y, ref_z = llh_to_ecef_vectorized(ref_lat, ref_lon, ref_alt)
    dx, dy, dz = x - ref_x, y - ref_y, z - ref_z
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return e, n, u
def parse_sp3(file_path):
    sp3_data = []
    base_time = None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('*') and base_time is None:
            parts = line.split()
            year, month, day, hour, minute = map(int, parts[1:6])
            second = float(parts[6])
            base_time = datetime.datetime(year, month, day, hour, minute, int(second))
            break
    if base_time is None:
        raise ValueError("SP3 文件格式错误：未找到时间头 (*)")
    # 计算 GPS Week Start
    gps_epoch = datetime.datetime(1980, 1, 6, 0, 0, 0)
    delta = base_time - gps_epoch
    week_num = delta.days // 7
    week_start = gps_epoch + datetime.timedelta(weeks=week_num)
    base_tow = (base_time - week_start).total_seconds()
    # 解析卫星数据
    for line in lines:
        if line.startswith('*'):
            parts = line.split()
            dt = datetime.datetime(int(parts[1]), int(parts[2]), int(parts[3]),
                                   int(parts[4]), int(parts[5]), int(float(parts[6])))
            current_tow = (dt - base_time).total_seconds() + base_tow
        elif line.startswith('P'):
            sat_id = line[1:4]
            try:
                x = float(line[4:18]) * 1000
                y = float(line[18:32]) * 1000
                z = float(line[32:46]) * 1000
                if x == 0 and y == 0 and z == 0: continue
                sp3_data.append({'sat_id': sat_id, 'tow': current_tow, 'x': x, 'y': y, 'z': z})
            except ValueError:
                continue
    return pd.DataFrame(sp3_data), base_time
def process_single_dataset(folder_path, output_filename='RXM-RAWX_processed.csv'):
    """处理单个文件夹的核心逻辑"""
    # 1. 自动寻找文件
    csv_path = os.path.join(folder_path, 'RXM-RAWX.csv')
    sp3_files = glob.glob(os.path.join(folder_path, '*.sp3'))
    if not os.path.exists(csv_path):
        return False, "缺少 RXM-RAWX.csv"
    if not sp3_files:
        return False, "缺少 .sp3 文件"
    sp3_path = sp3_files[0]
    sp3_name = os.path.basename(sp3_path)
    print(f"   -> 正在处理: CSV={os.path.basename(csv_path)}, SP3={sp3_name}")
    try:
        # 2. 读取数据
        df = pd.read_csv(csv_path, sep=';')
        sp3_df, _ = parse_sp3(sp3_path)
        # 3. 准备插值器 + 记录卫星时间范围（关键修改：避免外推）
        unique_sats = sp3_df['sat_id'].unique()
        interpolators = {}
        sat_time_ranges = {}  # 记录每个卫星的SP3时间范围（min_tow, max_tow）
        for sat in unique_sats:
            sat_data = sp3_df[sp3_df['sat_id'] == sat].sort_values('tow')
            if len(sat_data) > 3:
                t = sat_data['tow'].values
                interpolators[sat] = (
                    interp1d(t, sat_data['x'].values, kind='cubic', fill_value="extrapolate"),
                    interp1d(t, sat_data['y'].values, kind='cubic', fill_value="extrapolate"),
                    interp1d(t, sat_data['z'].values, kind='cubic', fill_value="extrapolate")
                )
                sat_time_ranges[sat] = (t.min(), t.max())  # 保存时间范围
        # 4. 映射 GNSS ID
        gnss_map = {'GPS': 'G', 'Glonass': 'R', 'Galileo': 'E', 'Beidou': 'C'}
        df['sat_key'] = df['GNSS identifier (gnssId) []'].map(gnss_map) + \
                        df['Satellite identifier (svId) []'].apply(lambda x: f"{int(x):02d}")
        # 5. 计算角度（关键修改：过滤超出SP3时间范围的数据）
        df['Elevation'] = np.nan
        df['Azimuth'] = np.nan
        groups = df.groupby('sat_key')
        for sat_key, group_idx in tqdm(groups.groups.items(), desc=f"处理卫星 {sat_key}"):
            if sat_key not in interpolators:
                continue
            indices = group_idx
            sub_df = df.loc[indices]
            t_rx = sub_df['GPSSecondsOfWeek [s]'].values
            pr = sub_df['Pseudorange measurement (prMes) [m]'].values
            lat = sub_df['Latitude (GT Lat) [deg]'].values
            lon = sub_df['Longitude (GT Lon) [deg]'].values
            hgt = sub_df['Height above ellipsoid (GT Height) [m]'].values
            t_tx = t_rx - (pr / C_LIGHT)  # 卫星发射时间
            # 过滤超出SP3时间范围的数据（允许±30秒缓冲）
            min_tow, max_tow = sat_time_ranges[sat_key]
            valid_mask = (t_tx >= min_tow - 30) & (t_tx <= max_tow + 30)
            if not np.any(valid_mask):
                print(f"  警告：卫星 {sat_key} 无有效时间范围内的数据，跳过")
                continue
            # 仅对有效数据计算角度
            fx, fy, fz = interpolators[sat_key]
            sat_x = fx(t_tx[valid_mask])
            sat_y = fy(t_tx[valid_mask])
            sat_z = fz(t_tx[valid_mask])
            # 坐标转换
            e, n, u = ecef_to_enu_vectorized(sat_x, sat_y, sat_z,
                                             lat[valid_mask], lon[valid_mask], hgt[valid_mask])
            az = np.degrees(np.arctan2(e, n))
            az = (az + 360) % 360  # 归一化到0-360度
            hor_dist = np.sqrt(e ** 2 + n ** 2)
            el = np.degrees(np.arctan2(u, hor_dist))  # 仰角（-90到90度）
            # 赋值有效数据
            df.loc[indices[valid_mask], 'Azimuth'] = az
            df.loc[indices[valid_mask], 'Elevation'] = el
        # 关键修改2：填充剩余NaN值（用全局均值，后续特征工程会用训练集均值优化）
        print(f"\n填充缺失值...")
        for col in ['Elevation', 'Azimuth']:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                print(f"  列 '{col}'：缺失值数={df[col].isnull().sum()}, 填充均值={mean_val:.2f}")
        # 6. 保存结果
        df.drop(columns=['sat_key'], inplace=True)
        save_path = os.path.join(folder_path, output_filename)
        df.to_csv(save_path, index=False, sep=';')
        return True, f"成功生成: {output_filename}"
    except Exception as e:
        return False, f"发生异常: {str(e)}"
# --- 主控制逻辑 ---
def batch_process_all(root_dir):
    print(f"🚀 开始批量处理 GNSS 数据集")
    print(f"📂 根目录: {root_dir}")
    print("-" * 60)
    results = []
    # 遍历第一层子目录
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    for folder in tqdm(subfolders, desc="总体进度"):
        folder_name = os.path.basename(folder)
        print(f"\n检查文件夹: {folder_name}")
        success, msg = process_single_dataset(folder)
        results.append({'Folder': folder_name, 'Status': 'Success' if success else 'Failed', 'Message': msg})
        if success:
            print(f"   ✅ {msg}")
        else:
            print(f"   ❌ {msg}")
    print("-" * 60)
    print("📊 处理总结:")
    res_df = pd.DataFrame(results)
    print(res_df)
    # 保存处理日志
    res_df.to_csv(os.path.join(root_dir, 'batch_process_log.csv'), index=False)
if __name__ == "__main__":
    # 请修改为你的数据集根目录
    DATASET_ROOT = r"../dataset"
    if os.path.exists(DATASET_ROOT):
        batch_process_all(DATASET_ROOT)
    else:
        print(f"错误: 找不到目录 {DATASET_ROOT}")
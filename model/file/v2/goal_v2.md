# UrbanNav-HK_TST 完整系统集成与评估
## 详细 Goal 文档（Codex 专用）

**项目代号**：GNSS-NLOS-HK-Eval  
**版本**：1.0  
**目标**：在香港尖沙咀 (TST) 数据集上完整运行 PI-PEM 三模块系统，进行多维度评估和对比实验  
**交付形式**：端到端可执行的 Python 代码 + 详细结果分析报告

---

## 一、整体任务概览

### 1.1 背景与目标

**原系统状态**：
- Module 1（NLOS 识别）：在欧洲城市（柏林、法兰克福）训练，F1 达 0.85-0.91
- Module 2（融合定位）：因 DOP 膨胀问题，仅在特定场景有效
- Module 3（自适应校正）：通过残差反馈实现普适改善

**新数据集特点**：
- 地点：香港尖沙咀（亚洲超高密集都市峡谷）
- 规模：541 个有效历元（原始 705，其中 164 因 SP3 缺失被跳过）
- NLOS 比例：仅 7.4%（原欧洲数据 40-50%），是低 NLOS 场景
- 卫星系统：GPS + GLONASS + Galileo + BeiDou（116 颗卫星可见）
- 轨迹长度：~650 m，仅 13 分钟

**任务目标**：
1. 验证系统在新地理区域（亚洲 vs 欧洲）的泛化性能
2. 理解在低 NLOS 环境下模型的行为与限制
3. 通过对比实验找出最优的模块配置与参数
4. 产出详细的定位精度评估与可视化结果

### 1.2 关键限制与假设

**已知限制**：
- NLOS 样本极少（7.4%）→ Module 1 的 NLOS 识别准确性可能不高
- 数据量小（541 个历元）→ 训练易过拟合，建议用迁移学习（Fine-tune）或从零开始但要小心
- 轨迹短且单一方向 → 特征多样性有限
- 部分历元（164 个）因 SP3 缺失无法处理

**处理策略**：
- 用"前 70% 训练 + 后 30% 验证"的时间划分，避免数据泄露
- 考虑从零训练 vs Fine-tune 两种方案，对比结果
- 在报告中明确指出低 NLOS 比例对结论的影响

---

## 二、任务 1：数据适配与预处理

### 2.1 输入数据来源

**文件清单**（假设存储在 `data/urbannav_hk_tst/raw/` 下）：

```
raw/
├── ground_truth.json          # 787 历元，INS 地面真值
├── novatel_gnss_obs.json      # 705 历元，GNSS 观测（有 82 个历元缺失）
├── sky_mask.json              # 148,642 条记录，建筑遮挡天际线
├── imu.csv                    # 314,194 条，IMU 数据 (~100 Hz)
├── receiver_summary.json      # 10 个接收机元数据
└── time_reference.csv         # GPS 时间参考脉冲
```

**数据规模**：
- 总文件大小：~750 MB（sky_mask 占 681 MB）
- 处理后预期大小：~50-100 MB（取决于压缩格式）

### 2.2 预处理流程（五大步骤）

#### Step 1：时间对齐（Time Alignment）

**输入**：ground_truth.json, novatel_gnss_obs.json, imu.csv

**关键问题**：
- GT 有 787 个历元 @ 1 Hz
- GNSS obs 有 705 个历元 @ 1 Hz（缺少 82 个）
- IMU 有 314,194 条 @ ~100 Hz（UNIX epoch 时间戳）
- sky_mask 有 148,642 条 @ ~10 Hz（无时间戳，仅空间坐标）

**对齐策略**：

```python
# 伪代码
def align_by_time():
    """
    核心逻辑：
    1. 将所有时间戳转为统一的 GPS 秒数（GPS week + seconds）
    2. 对每个 GNSS 历元，在 GT 中寻找最近的历元（时间差 < 0.5 秒）
    3. 如果找到 GT，视为"对齐成功"；否则标记为"skip"
    4. 丢弃无 GNSS 观测的 GT 历元
    """
    
    # 时间戳转换函数
    def unix_to_gps_seconds(unix_timestamp):
        """UNIX epoch (1970-01-01) 转 GPS epoch (1980-01-06)"""
        GPS_EPOCH_UNIX = 315964800  # 秒差
        return unix_timestamp - GPS_EPOCH_UNIX
    
    def gps_time_to_seconds(gps_week, gps_sec):
        """GPS 周和秒 → 总秒数（从 GPS epoch）"""
        return gps_week * 604800 + gps_sec
    
    # 加载三个源
    gt_list = load_json('ground_truth.json')         # 787 条
    gnss_obs_list = load_json('novatel_gnss_obs.json')  # 705 条
    imu_raw = load_csv('imu.csv')                    # 314k 条
    
    # 转换 GT 时间戳
    for gt in gt_list:
        gt['gps_seconds_total'] = gps_time_to_seconds(gt['gps_week'], gt['gps_seconds'])
    
    # 转换 GNSS obs 时间戳
    for obs in gnss_obs_list:
        obs['gps_seconds_total'] = gps_time_to_seconds(obs['time']['gps_week'], obs['time']['gps_seconds'])
    
    # 转换 IMU 时间戳
    imu_raw['gps_seconds'] = imu_raw['timestamp'].apply(unix_to_gps_seconds)
    
    # 对齐 GNSS obs 与 GT
    aligned_epochs = []
    for obs in gnss_obs_list:
        # 在 GT 中寻找最近的历元
        gt_candidates = [g for g in gt_list if abs(g['gps_seconds_total'] - obs['gps_seconds_total']) < 0.5]
        
        if not gt_candidates:
            continue  # 无对应 GT，跳过
        
        nearest_gt = min(gt_candidates, key=lambda g: abs(g['gps_seconds_total'] - obs['gps_seconds_total']))
        
        # 创建对齐的历元
        aligned_epoch = {
            'epoch_idx': len(aligned_epochs),
            'gps_week': obs['time']['gps_week'],
            'gps_seconds': obs['time']['gps_seconds'],
            'gps_seconds_total': obs['gps_seconds_total'],
            
            'gt': nearest_gt,  # 完整的 GT 字典
            'gnss_obs': obs,   # 完整的观测字典
            'imu_window': None,  # 稍后填充
        }
        
        aligned_epochs.append(aligned_epoch)
    
    return aligned_epochs  # 预期 ~705 个对齐历元
```

**输出格式**：
```python
aligned_epochs = [
    {
        'epoch_idx': 0,
        'gps_week': 2158,
        'gps_seconds': 95593.123,
        'gt': {
            'latitude': 22.297,
            'longitude': 114.175,
            'altitude': 3.5,
            'ecef_x': ..., 'ecef_y': ..., 'ecef_z': ...,
            'velocity': {...},
            'attitude': {...},
            'quality': 3,
        },
        'gnss_obs': {
            'sats': [
                {
                    'system': 'G', 'sv_id': 1,
                    'C1C': ..., 'S1C': ..., 'D1C': ...,
                    # ... 其他观测量
                },
                # ... 更多卫星
            ]
        },
        'imu_window': None,  # 待填充
    },
    # ... 704 个更多历元
]
```

**验证**：
- [ ] 对齐历元数应为 705（如文档所述）或接近
- [ ] 每个历元的时间戳单调递增
- [ ] gt 和 gnss_obs 的时间戳差异 < 0.5 秒

---

#### Step 2：空间插值（Sky Mask Spatial Interpolation）

**输入**：aligned_epochs + sky_mask.json

**问题**：
- sky_mask 有 148,642 条记录，密集采样沿轨迹（~10 Hz）
- 每条记录有 (lat, lon, alt, elevations[361])
- 需要为每个 1 Hz GNSS 历元（仅 705 个）找到对应的 elevation_mask

**方案**：KD-tree 1-NN 空间搜索

```python
def interpolate_sky_mask(aligned_epochs, sky_mask_list):
    """
    为每个 GNSS 历元的 (lat, lon) 找最近的 sky_mask 记录，
    提取其 elevation_mask (361,)
    """
    
    # 1. 构建 KD-tree（2D：lat, lon）
    from scipy.spatial import cKDTree
    
    sky_mask_positions = np.array([
        [sm['lat'], sm['lon']] for sm in sky_mask_list
    ])  # shape (148642, 2)
    
    kdtree = cKDTree(sky_mask_positions)
    
    # 2. 对每个 GNSS 历元查询
    for epoch in aligned_epochs:
        query_pos = np.array([
            [epoch['gt']['latitude'], epoch['gt']['longitude']]
        ])
        
        # 找最近的 k=1 条 sky_mask 记录（这里可用加权平均改进）
        distances, indices = kdtree.query(query_pos, k=1)
        
        nearest_sm = sky_mask_list[indices[0]]
        
        epoch['sky_mask'] = {
            'elevations': np.array(nearest_sm['elevations']),  # (361,)
            'source_lat': nearest_sm['lat'],
            'source_lon': nearest_sm['lon'],
            'distance_m': distances[0] * 111000,  # 近似转为米（1 度 ≈ 111 km）
        }
    
    return aligned_epochs
```

**优化建议**：
- 如果内存不足（sky_mask 文件 681 MB），考虑流式处理或分块加载
- 可用加权 k-NN（k=3-5）提高鲁棒性，但速度稍慢

**验证**：
- [ ] 每个历元都有 elevations (361,)
- [ ] 距离通常 < 100 m（同轨迹相邻采样点）
- [ ] elevations 值在 [2, 82] 度范围内

---

#### Step 3：卫星几何计算（Satellite Geometry）

**输入**：aligned_epochs + SP3 星历文件

**关键：需要卫星的 ECEF 位置**

**SP3 星历来源**：

```
WUM（武汉大学）MGEX 产品
- 链接：http://igs.gnsswhu.cn/ 或 https://ftp.gfsc.esa.int/
- 文件命名：WUM0MGXULA_20211380200_01D_05M_ORB.SP3
  (day-138, 2021年5月17日，精度5分钟采样)
- 包含系统：GPS(G), GLONASS(R), Galileo(E), BeiDou(C) 共 ~116 颗卫星
- 文件大小：~2.5 MB
```

**解析方案**（两种）：

**方案 A（推荐）：用 pyginv 库**

```python
# 安装：pip install pyginv
from pyginv import SP3Reader

def load_sp3_ephemeris(sp3_file_path):
    """加载 SP3 星历"""
    reader = SP3Reader(sp3_file_path)
    return reader.eph  # dict: {(system, sv_id, gps_seconds): (x, y, z)}
```

**方案 B（备选）：自己写简单解析器**

```python
def parse_sp3_simple(sp3_file_path):
    """
    简化的 SP3 解析器
    返回：dict {(system_id, sv_num, epoch_gps_seconds): (x_km, y_km, z_km)}
    """
    eph = {}
    with open(sp3_file_path, 'r') as f:
        lines = f.readlines()
        
        current_epoch = None
        for line in lines:
            if line.startswith('*'):  # 新历元标记
                # 解析 *  YYYY  M  D  H  M SS
                parts = line.split()
                year, month, day, hour, minute, second = map(int, parts[1:7])
                # 转为 GPS 秒数（略，需要日期转换函数）
                current_epoch = gps_time_from_date(year, month, day, hour, minute, second)
            
            elif line.startswith('P'):  # 卫星位置行
                # 格式：P<system><sv_id> X Y Z CLK [more fields]
                system = line[1]  # G/R/E/C
                sv_id = int(line[2:4])
                x = float(line[4:14])
                y = float(line[14:24])
                z = float(line[24:34])
                
                eph[(system, sv_id, current_epoch)] = (x, y, z)
    
    return eph
```

**计算仰角/方位角**：

```python
def compute_satellite_geometry(aligned_epochs, sp3_eph):
    """
    为每颗卫星计算仰角、方位角、NLOS 标签
    """
    
    for epoch in aligned_epochs:
        epoch['satellites'] = []
        
        for sat_obs in epoch['gnss_obs']['sats']:
            system = sat_obs['system']  # 'G', 'R', 'E', 'C'
            sv_id = sat_obs['sv_id']
            
            # 从 SP3 查询卫星位置
            gps_seconds = epoch['gps_seconds_total']
            
            # SP3 是 5 分钟采样，需要插值
            sv_ecef = interpolate_sv_position(system, sv_id, gps_seconds, sp3_eph)
            
            if sv_ecef is None:
                continue  # 该卫星在 SP3 中无数据，跳过
            
            # 计算接收机 ECEF（从 LLA）
            rx_ecef = lla_to_ecef(
                epoch['gt']['latitude'],
                epoch['gt']['longitude'],
                epoch['gt']['altitude']
            )
            
            # 计算 ENU 坐标系并转换
            enu = ecef_to_enu(rx_ecef, sv_ecef, epoch['gt']['latitude'], epoch['gt']['longitude'])
            
            # 计算仰角、方位角
            elevation_deg, azimuth_deg = enu_to_elevation_azimuth(enu)
            
            # 获取该方位角的建筑遮挡仰角
            az_idx = int(round(azimuth_deg)) % 361
            elevation_mask = epoch['sky_mask']['elevations'][az_idx]
            
            # 判断 NLOS/LOS
            if elevation_deg >= elevation_mask:
                nlos_label = 0  # LOS
            else:
                nlos_label = 1  # NLOS
            
            # 计算几何距离（用于后续 LS 定位）
            geometric_range = np.linalg.norm(sv_ecef - rx_ecef)
            
            # 保存该卫星的信息
            sat_result = {
                'system': system,
                'sv_id': sv_id,
                'sv_ecef_x': sv_ecef[0],
                'sv_ecef_y': sv_ecef[1],
                'sv_ecef_z': sv_ecef[2],
                'elevation_deg': elevation_deg,
                'azimuth_deg': azimuth_deg,
                'elevation_mask_deg': elevation_mask,
                'elevation_above_mask': elevation_deg - elevation_mask,
                'nlos_label': nlos_label,
                'geometric_range_m': geometric_range,
                
                # 观测量
                'pr_m': sat_obs.get('C1C', None),  # 伪距（米）
                'cn0_dbhz': sat_obs.get('S1C', None),  # 载噪比（dB-Hz）
                'doppler_hz': sat_obs.get('D1C', None),  # 多普勒（Hz）
            }
            
            epoch['satellites'].append(sat_result)
        
        if not epoch['satellites']:
            epoch['skip'] = True  # 无有效卫星，标记为跳过
    
    return aligned_epochs
```

**关键函数签名**：

```python
def lla_to_ecef(latitude_deg, longitude_deg, altitude_m):
    """LLA → ECEF (WGS84)
    返回：(x, y, z) in meters
    """
    pass

def ecef_to_enu(rx_ecef, sv_ecef, latitude_deg, longitude_deg):
    """ECEF → ENU (相对于接收机位置)
    返回：(east, north, up) in meters
    """
    pass

def enu_to_elevation_azimuth(enu_vector):
    """ENU → 仰角、方位角
    返回：(elevation_deg, azimuth_deg)
    """
    pass

def interpolate_sv_position(system, sv_id, gps_seconds, sp3_eph):
    """从 SP3 插值卫星位置（5 分钟采样 → 1 秒采样）
    使用拉格朗日插值或线性插值
    返回：(x, y, z) in km，或 None 如果无数据
    """
    pass
```

**验证**：
- [ ] elevation_deg ∈ [-90, 90]
- [ ] azimuth_deg ∈ [0, 360)
- [ ] NLOS 标签与 elevation ↔ elevation_mask 的大小关系一致
- [ ] 仰角低（< 20°）的卫星 NLOS 比例应较高

---

#### Step 4：特征提取（Feature Extraction）

**输入**：aligned_epochs（含 satellites 信息）

**目标**：为每颗卫星生成 11 维特征向量（与 Module 1 兼容）

**特征设计**（与原项目保持一致）：

```python
def extract_features(aligned_epochs):
    """
    为每个历元的每颗卫星提取 11 维特征
    """
    
    for epoch in aligned_epochs:
        if epoch.get('skip', False):
            epoch['node_features'] = np.empty((0, 11))
            epoch['nlos_labels'] = np.empty(0, dtype=int)
            continue
        
        n_sats = len(epoch['satellites'])
        features = np.zeros((n_sats, 11))
        labels = np.zeros(n_sats, dtype=int)
        
        for i, sat in enumerate(epoch['satellites']):
            # 特征维度 0：仰角归一化
            features[i, 0] = sat['elevation_deg'] / 90.0
            
            # 特征维度 1：方位角归一化
            features[i, 1] = sat['azimuth_deg'] / 360.0
            
            # 特征维度 2：C/N0 归一化
            cn0 = sat['cn0_dbhz'] if sat['cn0_dbhz'] is not None else 30.0
            features[i, 2] = np.clip(cn0 / 60.0, 0, 2.0)  # 上界裁剪
            
            # 特征维度 3：伪距测量标准差（如果无则用默认 1m）
            pr_stdev = sat_obs.get('pr_stdev', 1.0) if hasattr(sat, 'pr_stdev') else 1.0
            features[i, 3] = np.clip(pr_stdev / 5.0, 0, 2.0)
            
            # 特征维度 4：伪距测量值归一化
            pr_m = sat['pr_m'] if sat['pr_m'] is not None else 2e7
            features[i, 4] = pr_m / 3e7
            
            # 特征维度 5：伪距创新量（需要先估计钟偏差）
            # 暂时用 0（后续可优化）
            features[i, 5] = 0.0
            
            # 特征维度 6：仰角余弦（几何精度代理）
            features[i, 6] = np.cos(np.radians(sat['elevation_deg']))
            
            # 特征维度 7-10：星座 one-hot 编码
            system = sat['system']
            features[i, 7] = 1.0 if system == 'G' else 0.0  # GPS
            features[i, 8] = 1.0 if system == 'R' else 0.0  # GLONASS
            features[i, 9] = 1.0 if system == 'E' else 0.0  # Galileo
            features[i, 10] = 1.0 if system == 'C' else 0.0  # BeiDou
            
            # NLOS 标签
            labels[i] = sat['nlos_label']
        
        epoch['node_features'] = features  # shape (N, 11)
        epoch['nlos_labels'] = labels      # shape (N,)
    
    return aligned_epochs
```

**说明**：
- 特征维度 5（pr_innovation）暂时置 0，可在后续优化时计算（需要钟偏差估计）
- 所有特征应归一化到大致 [0, 1] 范围（除 dim 6 的余弦已在 [-1, 1]）

**验证**：
- [ ] node_features 的最小值 ≥ -0.1，最大值 ≤ 2.0
- [ ] nlos_labels ∈ {0, 1}
- [ ] 特征 7-10 中有且仅有一个为 1.0（one-hot 约束）

---

#### Step 5：图结构与数据集生成（Graph Structure & Dataset Split）

**输入**：aligned_epochs（含 node_features 和 nlos_labels）

**图结构构建**：

```python
def build_graph_structure(aligned_epochs):
    """
    为每个历元构建图结构（邻接矩阵）
    邻接规则：方位角差 < 90° 的卫星之间连边（无向）
    """
    
    for epoch in aligned_epochs:
        if epoch.get('skip', False):
            epoch['edge_index'] = np.empty((2, 0), dtype=int)
            continue
        
        n_sats = len(epoch['satellites'])
        edges = []
        
        for i in range(n_sats):
            for j in range(i + 1, n_sats):
                az_i = epoch['satellites'][i]['azimuth_deg']
                az_j = epoch['satellites'][j]['azimuth_deg']
                
                # 计算两个方位角的差异（考虑环绕）
                az_diff = abs(az_i - az_j)
                az_diff = min(az_diff, 360 - az_diff)
                
                if az_diff < 90:
                    # 双向边
                    edges.append((i, j))
                    edges.append((j, i))
        
        if edges:
            edge_index = np.array(edges, dtype=int).T  # shape (2, E)
        else:
            # 无边的情况（边界条件），加自环
            edge_index = np.array([[i, i] for i in range(n_sats)], dtype=int).T
        
        epoch['edge_index'] = edge_index
    
    return aligned_epochs
```

**数据集划分**（时间划分，无数据泄露）：

```python
def split_train_val(aligned_epochs, train_ratio=0.7):
    """
    时间划分：前 70% 训练，后 30% 验证
    """
    
    # 过滤掉被标记为 skip 的历元
    valid_epochs = [e for e in aligned_epochs if not e.get('skip', False)]
    
    n_total = len(valid_epochs)
    n_train = int(n_total * train_ratio)
    
    train_dataset = valid_epochs[:n_train]      # 前 70%
    val_dataset = valid_epochs[n_train:]        # 后 30%
    
    return train_dataset, val_dataset
```

**统计信息**：

```python
def compute_dataset_statistics(train_dataset, val_dataset):
    """计算数据集的统计指标"""
    
    stats = {
        'train': {
            'num_epochs': len(train_dataset),
            'total_satellites': sum(len(e['satellites']) for e in train_dataset),
            'total_los': sum((e['nlos_labels'] == 0).sum() for e in train_dataset),
            'total_nlos': sum((e['nlos_labels'] == 1).sum() for e in train_dataset),
            'systems': {},
            'elevation_distribution': {},
        },
        'val': {
            'num_epochs': len(val_dataset),
            'total_satellites': sum(len(e['satellites']) for e in val_dataset),
            'total_los': sum((e['nlos_labels'] == 0).sum() for e in val_dataset),
            'total_nlos': sum((e['nlos_labels'] == 1).sum() for e in val_dataset),
            'systems': {},
            'elevation_distribution': {},
        },
    }
    
    # 各星座统计
    for dataset, key in [(train_dataset, 'train'), (val_dataset, 'val')]:
        systems_count = {}
        elev_bins = {}
        
        for epoch in dataset:
            for sat in epoch['satellites']:
                system = sat['system']
                systems_count[system] = systems_count.get(system, 0) + 1
                
                # 仰角分布
                elev = int(sat['elevation_deg'] // 10) * 10  # 分组
                elev_bins[elev] = elev_bins.get(elev, 0) + 1
        
        stats[key]['systems'] = systems_count
        stats[key]['elevation_distribution'] = elev_bins
    
    return stats
```

**预期数据集规模**：

根据 DATASET_README.md 的"Section 12.3"：
- 总有效历元：541（包含 train + val，164 个因 SP3 缺失被排除）
- Train 历元：~378 (70%)
- Val 历元：~163 (30%)
- 总卫星观测数：3,426
- NLOS 总数：253 (7.4%)

---

### 2.3 预处理输出

**最终输出结构**：

```
data/urbannav_hk_tst/processed/
├── train_dataset.pkl        # 训练集（~378 历元）
├── val_dataset.pkl          # 验证集（~163 历元）
├── dataset_statistics.json  # 统计汇总
└── preprocessing_log.txt    # 详细日志
```

**每个历元的数据结构**：

```python
epoch = {
    # 元数据
    'epoch_idx': 0,
    'gps_week': 2158,
    'gps_seconds': 95593.123,
    
    # 地面真值
    'gt_ecef': np.array([6378000, -2200000, 2400000]),  # (3,)
    'gt_lla': {'lat': 22.297, 'lon': 114.175, 'alt': 3.5},
    
    # 图结构
    'node_features': np.array([...]),  # (N, 11)，N = 该历元卫星数
    'edge_index': np.array([...]),     # (2, E)，E = 边数
    'nlos_labels': np.array([...]),    # (N,)
    
    # 辅助信息（用于可视化和调试）
    'satellites': [
        {
            'system': 'G',
            'sv_id': 1,
            'elevation_deg': 45.3,
            'azimuth_deg': 287.2,
            'nlos_label': 0,
            'pr_m': 22456789.5,
            'cn0_dbhz': 38.5,
            'geometric_range_m': 22456000,
        },
        # ...
    ],
}
```

---

## 三、任务 2：完整流程执行

### 3.1 Module 1：NLOS 感知与误差建模

#### 3.1.1 模型选项

**选项 A：从零训练**

```
用 train_dataset 从头训练新的 GAT+MoG 模型
- 优点：完全适应香港数据特性
- 缺点：数据量少（378 历元），易过拟合
- 预期 F1：0.70-0.78（比欧洲数据低，因为 NLOS 样本少）
```

**选项 B：Fine-tune（推荐）**

```
加载欧洲预训练模型（exp_051），在香港数据上微调
- 优点：利用迁移学习，对抗过拟合
- 缺点：需要原模型权重
- 预期 F1：0.78-0.85（介于从零与预训练之间）
```

**建议**：用选项 B（Fine-tune），学习率设为原始的 0.1-0.2 倍（如 5e-6），训练 50-100 个 epoch

#### 3.1.2 训练配置

```python
# config_hk.py
DATASET = 'urbannav_hk_tst'

# 数据
BATCH_SIZE = 32
USE_BLOCK_DIAGONAL_BATCHING = True  # 处理可变卫星数

# 模型
IN_FEATURES = 11
HIDDEN_FEATURES = 128
NUM_LAYERS = 2
NUM_HEADS = 8

# 训练
NUM_EPOCHS = 100  # Fine-tune 的情况
LEARNING_RATE = 5e-6  # 原始 5e-5，现在降 10 倍
GRADIENT_CLIP = 10.0
WEIGHT_DECAY = 1e-4

# 损失函数
MOG_PURE_BCE_EPOCHS = 8
MOG_BLEND_EPOCHS = 25
# 阶段 3 = 100 - 8 - 25 = 67

LAMBDA_BCE = 0.6
LAMBDA_MU_REG = 0.20
LAMBDA_MU_DIRECTION = 1.0
MU_NLOS_TARGET = 0.30  # km

# Early stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_DELTA = 0.001
```

#### 3.1.3 推理与输出

```python
def run_module1_inference(val_dataset, model_path):
    """
    在验证集上运行 Module 1 推理
    """
    
    model = load_model(model_path)
    model.eval()
    
    mog_predictions = []
    
    for epoch in val_dataset:
        node_features = torch.tensor(epoch['node_features'], dtype=torch.float32).to(device)
        edge_index = torch.tensor(epoch['edge_index'], dtype=torch.long).to(device)
        
        with torch.no_grad():
            p_los, mu_nlos, sigma_los, sigma_nlos = model(node_features, edge_index)
        
        # 转为 numpy
        p_los = p_los.cpu().numpy().flatten()
        mu_nlos = mu_nlos.cpu().numpy().flatten()
        sigma_los = sigma_los.cpu().numpy().flatten()
        sigma_nlos = sigma_nlos.cpu().numpy().flatten()
        
        mog_output = {
            'epoch_idx': epoch['epoch_idx'],
            'gps_seconds': epoch['gps_seconds'],
            'p_los': p_los,
            'mu_nlos': mu_nlos,  # km
            'sigma_los': sigma_los,  # km
            'sigma_nlos': sigma_nlos,  # km
        }
        
        mog_predictions.append(mog_output)
    
    # 评估分类性能
    all_p_los = np.concatenate([p['p_los'] for p in mog_predictions])
    all_labels = np.concatenate([e['nlos_labels'] for e in val_dataset])
    
    # 硬判决
    predictions = (all_p_los > 0.5).astype(int)
    
    f1 = f1_score(all_labels, predictions)
    accuracy = accuracy_score(all_labels, predictions)
    
    print(f"Module 1 Performance on HK dataset:")
    print(f"  F1: {f1:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")
    
    return mog_predictions, f1, accuracy
```

**输出文件**：
```
results/module1/
├── mog_predictions.pkl        # 所有推理结果
├── classification_metrics.json # F1, Accuracy 等
└── model_analysis.json        # p_los gap, mu 分布等
```

---

### 3.2 Module 2：融合定位

#### 3.2.1 基准方法（Standard LS）

```python
def solve_standard_ls(observations, sv_positions):
    """
    标准最小二乘定位
    
    参数：
        observations: 伪距观测值 (N,)
        sv_positions: 卫星 ECEF 位置 (N, 3)
    
    返回：
        position: 估计的接收机 ECEF 位置 (3,)
        clock_bias: 钟偏差 (标量)
    """
    
    # Gauss-Newton 迭代
    x_est = np.array([6378000, -2200000, 2400000])  # 初始猜测
    clock_bias = 0
    
    for iteration in range(10):  # 最多 10 次迭代
        # 计算设计矩阵 H
        distances = np.linalg.norm(sv_positions - x_est, axis=1)
        direction = (sv_positions - x_est) / distances[:, None]
        
        H = np.column_stack([
            -direction,
            np.ones(len(observations))
        ])  # shape (N, 4)
        
        # 计算残差
        predicted_pr = distances + clock_bias
        residuals = observations - predicted_pr
        
        # 正规方程：(H^T H)^{-1} H^T r
        HTH = H.T @ H
        HTr = H.T @ residuals
        
        try:
            delta = np.linalg.solve(HTH, HTr)
        except np.linalg.LinAlgError:
            break
        
        x_est = x_est + delta[:3]
        clock_bias = clock_bias + delta[3]
        
        # 收敛检查
        if np.linalg.norm(delta) < 0.01:
            break
    
    return x_est, clock_bias
```

#### 3.2.2 加权最小二乘（WLS-MoG）

```python
def solve_wls_mog(observations, sv_positions, mog_output):
    """
    MoG 信息加权最小二乘
    
    权重：w_i = p_los_i / (sigma_los_i)^2
    """
    
    p_los = mog_output['p_los']
    sigma_los = mog_output['sigma_los']
    
    # 权重矩阵（对角）
    weights = p_los / (sigma_los ** 2 + 1e-6)  # 加小常数避免除零
    weights = np.clip(weights, 0.01, 100)  # 合理范围
    
    # 迭代 WLS
    x_est = np.array([6378000, -2200000, 2400000])
    clock_bias = 0
    
    for iteration in range(10):
        distances = np.linalg.norm(sv_positions - x_est, axis=1)
        direction = (sv_positions - x_est) / distances[:, None]
        
        H = np.column_stack([-direction, np.ones(len(observations))])
        
        residuals = observations - (distances + clock_bias)
        
        # 加权正规方程
        W = np.diag(weights)
        HTW = H.T @ W
        delta = np.linalg.solve(HTW @ H, HTW @ residuals)
        
        x_est = x_est + delta[:3]
        clock_bias = clock_bias + delta[3]
        
        if np.linalg.norm(delta) < 0.01:
            break
    
    return x_est, clock_bias
```

#### 3.2.3 因子图融合（FG-MoG+2A）

```python
def solve_factor_graph_mog(observations, sv_positions, mog_output):
    """
    MoG 混合高斯因子图优化
    
    目标函数：max_x ∑_i log[ p_los·N(r_i|0,σ_los) + (1-p_los)·N(r_i|μ_nlos,σ_nlos) ]
    
    使用 L-BFGS-B 数值优化
    """
    
    p_los = mog_output['p_los']
    mu_nlos = mog_output['mu_nlos'] * 1000  # km → m
    sigma_los = mog_output['sigma_los'] * 1000
    sigma_nlos = mog_output['sigma_nlos'] * 1000
    
    def negative_log_likelihood(state):
        """NLL（用于最小化）"""
        x = state[:3]
        clk = state[3]
        
        distances = np.linalg.norm(sv_positions - x, axis=1)
        residuals = observations - (distances + clk)
        
        # 混合高斯 NLL
        log_likelihood = []
        for i, r in enumerate(residuals):
            log_los = np.log(p_los[i]) + np.log(norm.pdf(r, 0, sigma_los[i]))
            log_nlos = np.log(1 - p_los[i]) + np.log(norm.pdf(r, mu_nlos[i], sigma_nlos[i]))
            
            # logsumexp 数值稳定计算
            log_likelihood.append(logsumexp([log_los, log_nlos]))
        
        return -np.sum(log_likelihood)
    
    # 从三个起点优化，选最优解
    x0_candidates = [
        np.array([6378000, -2200000, 2400000, 0]),  # Standard LS 结果
        np.array([6377000, -2220000, 2420000, 0]),  # WLS 结果
        np.array([6380000, -2180000, 2380000, 0]),  # 备选起点
    ]
    
    best_nll = np.inf
    best_state = None
    
    for x0 in x0_candidates:
        result = minimize(negative_log_likelihood, x0, method='L-BFGS-B')
        if result.fun < best_nll:
            best_nll = result.fun
            best_state = result.x
    
    return best_state[:3], best_state[3]
```

#### 3.2.4 定位结果汇总

```python
def run_module2_all_methods(val_dataset, mog_predictions):
    """
    对每个验证历元，运行三种定位方法
    """
    
    positioning_results = []
    
    for epoch, mog_output in zip(val_dataset, mog_predictions):
        # 提取伪距和卫星位置
        observations = np.array([
            sat['pr_m'] for sat in epoch['satellites']
        ])
        sv_positions = np.array([
            [sat['sv_ecef_x'], sat['sv_ecef_y'], sat['sv_ecef_z']]
            for sat in epoch['satellites']
        ])
        
        # 三种方法
        pos_ls, clk_ls = solve_standard_ls(observations, sv_positions)
        pos_wls, clk_wls = solve_wls_mog(observations, sv_positions, mog_output)
        pos_fg, clk_fg = solve_factor_graph_mog(observations, sv_positions, mog_output)
        
        # 计算 2D 水平误差（ENU 坐标系）
        gt_ecef = epoch['gt_ecef']
        
        error_ls = compute_2d_error(pos_ls, gt_ecef, epoch['gt_lla'])
        error_wls = compute_2d_error(pos_wls, gt_ecef, epoch['gt_lla'])
        error_fg = compute_2d_error(pos_fg, gt_ecef, epoch['gt_lla'])
        
        result = {
            'epoch_idx': epoch['epoch_idx'],
            'gps_seconds': epoch['gps_seconds'],
            
            'error_ls_m': error_ls,
            'error_wls_m': error_wls,
            'error_fg_m': error_fg,
            
            'position_ls_ecef': pos_ls,
            'position_wls_ecef': pos_wls,
            'position_fg_ecef': pos_fg,
        }
        
        positioning_results.append(result)
    
    return positioning_results
```

**输出文件**：
```
results/module2/
├── positioning_results.pkl    # 所有定位结果
├── method_comparison.json     # 三种方法的 CEP50/CEP95
└── error_distribution.pkl     # 误差统计
```

---

### 3.3 Module 3：残差反馈自适应选择

#### 3.3.1 核心组件

```python
class ResidualInnovationTracker:
    """残差创新量跟踪器"""
    
    def __init__(self, window_size=50, min_history=15):
        self.window_size = window_size
        self.min_history = min_history
        self.window = []
    
    def update(self, error_fg, error_ls):
        """
        创新量 = FG 误差 - LS 误差
        负值表示 FG 更好
        """
        innovation = error_fg - error_ls
        self.window.append(innovation)
        if len(self.window) > self.window_size:
            self.window.pop(0)
    
    def get_quality(self):
        """返回 (quality_label, confidence_score)"""
        if len(self.window) < self.min_history:
            return 'UNCERTAIN', 0.5
        
        mean_innov = np.mean(self.window)
        frac_better = sum(1 for x in self.window if x < 0) / len(self.window)
        
        if mean_innov < -20 and frac_better > 0.65:
            return 'HIGH', min(0.95, 0.5 + frac_better)
        elif mean_innov > 20 and frac_better < 0.35:
            return 'LOW', min(0.95, 0.5 + (1 - frac_better))
        else:
            return 'MEDIUM', 0.5


class SceneQualityDetector:
    """场景质量检测器（当前历元）"""
    
    def classify(self, mog_output, sv_positions, rx_pos):
        """
        基于 Module 1 输出和几何信息判断场景质量
        返回 (quality_label, confidence_score)
        """
        
        p_los = mog_output['p_los']
        
        # 特征 1：p_los Gap（LOS 和 NLOS 卫星的区分度）
        los_mask = p_los > 0.6
        nlos_mask = p_los < 0.4
        if los_mask.sum() == 0 or nlos_mask.sum() == 0:
            plos_gap = 0
        else:
            plos_gap = p_los[los_mask].mean() - p_los[nlos_mask].mean()
        
        # 特征 2：DOP 比率
        weights = p_los / (0.01 + sigma_los ** 2)
        pdop_weighted = compute_pdop(weights, sv_positions)
        pdop_uniform = compute_pdop(np.ones_like(p_los), sv_positions)
        pdop_ratio = pdop_weighted / pdop_uniform if pdop_uniform > 0 else 1.0
        
        # 综合评分（0-1）
        score = (
            0.5 * (plos_gap > 0.4) +  # p_los gap > 0.4 得 0.5 分
            0.5 * (pdop_ratio < 1.1)  # DOP 比率 < 1.1 得 0.5 分
        )
        
        if score >= 0.8:
            return 'HIGH', score
        elif score >= 0.4:
            return 'MEDIUM', score
        else:
            return 'LOW', score


class AdaptivePositionSelector:
    """自适应定位选择器"""
    
    def __init__(self, fg_threshold=0.70, wls_threshold=0.50):
        self.fg_threshold = fg_threshold
        self.wls_threshold = wls_threshold
        self.tracker = ResidualInnovationTracker()
        self.detector = SceneQualityDetector()
    
    def select(self, error_ls, error_wls, error_fg, 
               mog_output, sv_positions, rx_pos, epoch_idx):
        """
        根据多个信号综合判断，选择最佳定位方法
        """
        
        # 更新跟踪器
        self.tracker.update(error_fg, error_ls)
        tracker_quality, tracker_conf = self.tracker.get_quality()
        
        # 检测当前场景
        detector_quality, detector_conf = self.detector.classify(
            mog_output, sv_positions, rx_pos
        )
        
        # 融合两个信号
        if tracker_quality == 'HIGH' and detector_quality == 'HIGH':
            final_quality = 'HIGH'
            final_score = 0.9
        elif tracker_quality in ['HIGH', 'MEDIUM'] and detector_quality in ['HIGH', 'MEDIUM']:
            final_quality = 'MEDIUM'
            final_score = 0.6
        else:
            final_quality = 'LOW'
            final_score = 0.3
        
        # 方法选择
        if final_quality == 'HIGH' and final_score >= self.fg_threshold:
            selected_method = 'FG-MoG+2A'
            selected_error = error_fg
            selected_pos = None  # 稍后填充
        elif final_quality in ['HIGH', 'MEDIUM'] and final_score >= self.wls_threshold:
            selected_method = 'WLS-MoG'
            selected_error = error_wls
            selected_pos = None
        else:
            selected_method = 'Standard-LS'
            selected_error = error_ls
            selected_pos = None
        
        # 安全保险：如果选择方法差于 LS > 5%，回退到 LS
        if selected_error > error_ls * 1.05:
            selected_method = 'Standard-LS (fallback)'
            selected_error = error_ls
        
        return {
            'method': selected_method,
            'error_m': selected_error,
            'tracker_quality': tracker_quality,
            'detector_quality': detector_quality,
            'final_score': final_score,
        }
```

#### 3.3.2 完整运行

```python
def run_module3_adaptive(val_dataset, mog_predictions, positioning_results):
    """
    运行 Module 3 自适应选择
    """
    
    selector = AdaptivePositionSelector()
    adaptive_results = []
    
    for epoch, mog_out, pos_res in zip(val_dataset, mog_predictions, positioning_results):
        selection = selector.select(
            error_ls=pos_res['error_ls_m'],
            error_wls=pos_res['error_wls_m'],
            error_fg=pos_res['error_fg_m'],
            mog_output=mog_out,
            sv_positions=...,  # 从 epoch 提取
            rx_pos=...,
            epoch_idx=epoch['epoch_idx']
        )
        
        adaptive_results.append(selection)
    
    # 统计算法选用比例
    method_counts = {}
    for res in adaptive_results:
        method = res['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print("Algorithm selection distribution:")
    for method, count in method_counts.items():
        print(f"  {method}: {count/len(adaptive_results)*100:.1f}%")
    
    return adaptive_results
```

**输出文件**：
```
results/module3/
├── adaptive_results.pkl       # 自适应选择结果
├── algorithm_distribution.json # 各算法使用比例
└── innovation_trace.pkl       # 残差创新值时间序列
```

---

## 四、任务 3：结果评估与可视化

### 4.1 评估指标体系

#### 4.1.1 基础定位精度指标

```python
def compute_localization_metrics(estimated_positions, gt_positions):
    """
    计算全面的定位精度指标
    
    参数：
        estimated_positions: (N, 3) ECEF 坐标
        gt_positions: (N, 3) ECEF 真值坐标
    
    返回：dict 包含所有指标
    """
    
    # 转为 ENU 并计算 2D 误差
    errors_2d = []  # (N,)
    errors_3d = []
    
    for i in range(len(estimated_positions)):
        est = estimated_positions[i]
        gt = gt_positions[i]
        
        # 3D 误差
        error_3d = np.linalg.norm(est - gt)
        errors_3d.append(error_3d)
        
        # 2D 误差（需要转 ENU）
        error_2d = compute_2d_error(est, gt, ...)
        errors_2d.append(error_2d)
    
    errors_2d = np.array(errors_2d)
    errors_3d = np.array(errors_3d)
    
    # 计算各项指标
    metrics = {
        # 2D 指标
        'cep50_2d': np.percentile(errors_2d, 50),
        'cep95_2d': np.percentile(errors_2d, 95),
        'mean_error_2d': np.mean(errors_2d),
        'median_error_2d': np.median(errors_2d),
        'std_error_2d': np.std(errors_2d),
        'max_error_2d': np.max(errors_2d),
        'min_error_2d': np.min(errors_2d),
        
        # 3D 指标
        'mean_error_3d': np.mean(errors_3d),
        'rmse_3d': np.sqrt(np.mean(errors_3d ** 2)),
        'cep50_3d': np.percentile(errors_3d, 50),
        'cep95_3d': np.percentile(errors_3d, 95),
        
        # CDF 曲线数据（用于绘图）
        'cdf_errors': np.sort(errors_2d),
        'cdf_percentiles': np.linspace(0, 100, len(errors_2d)),
        
        # 错误集中度
        'error_below_10m': np.sum(errors_2d < 10) / len(errors_2d),
        'error_below_50m': np.sum(errors_2d < 50) / len(errors_2d),
        'error_below_100m': np.sum(errors_2d < 100) / len(errors_2d),
    }
    
    return metrics, errors_2d, errors_3d
```

#### 4.1.2 分层统计（按 NLOS 占比）

```python
def compute_stratified_metrics(val_dataset, positioning_results):
    """
    按历元中 NLOS 卫星比例分层计算指标
    """
    
    # 分类：高 NLOS (>50%), 中 NLOS (20-50%), 低 NLOS (<20%)
    low_nlos_epochs = []
    medium_nlos_epochs = []
    high_nlos_epochs = []
    
    for epoch, pos_res in zip(val_dataset, positioning_results):
        nlos_ratio = epoch['nlos_labels'].mean()
        
        if nlos_ratio < 0.2:
            low_nlos_epochs.append(pos_res)
        elif nlos_ratio <= 0.5:
            medium_nlos_epochs.append(pos_res)
        else:
            high_nlos_epochs.append(pos_res)
    
    # 分别计算各层的指标
    stratified_metrics = {
        'low_nlos': compute_localization_metrics(...),
        'medium_nlos': compute_localization_metrics(...),
        'high_nlos': compute_localization_metrics(...),
    }
    
    return stratified_metrics
```

#### 4.1.3 各算法性能对比

```python
def compare_all_methods(positioning_results, adaptive_results):
    """
    对 Standard LS, WLS, FG, Adaptive 四种方法的误差进行统计对比
    """
    
    methods = ['Standard-LS', 'WLS-MoG', 'FG-MoG+2A', 'Adaptive']
    comparison = {}
    
    for method in methods:
        if method == 'Adaptive':
            # 从 adaptive_results 提取
            errors = [r['error_m'] for r in adaptive_results]
        else:
            # 从 positioning_results 提取
            errors = [r[f'error_{method.lower().replace("+", "").replace("-", "_")}_m'] 
                      for r in positioning_results]
        
        metrics, _, _ = compute_localization_metrics(errors)
        comparison[method] = metrics
    
    return comparison
```

**输出文件**：
```
results/metrics/
├── localization_metrics.json      # 全局指标
├── stratified_metrics.json        # 分层指标
├── method_comparison.json         # 四方法对比
└── error_distributions.pkl        # 误差分布数据
```

---

### 4.2 可视化模块（详细清单）

#### 可视化 1：轨迹对比图（2D 俯视图）

```python
def plot_trajectory_2d(val_dataset, positioning_results, adaptive_results):
    """
    2D 俯视图：真值轨迹与各方法估计轨迹叠加
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 坐标列表
    gt_lats = [e['gt_lla']['lat'] for e in val_dataset]
    gt_lons = [e['gt_lla']['lon'] for e in val_dataset]
    
    methods = ['Standard-LS', 'WLS-MoG', 'FG-MoG+2A', 'Adaptive']
    axes = axes.flatten()
    
    for idx, (ax, method) in enumerate(zip(axes, methods)):
        # 绘制真值
        ax.plot(gt_lons, gt_lats, 'k-', linewidth=2, label='Ground Truth', zorder=10)
        ax.scatter(gt_lons[0], gt_lats[0], c='green', s=100, marker='o', 
                   label='Start', zorder=11)
        ax.scatter(gt_lons[-1], gt_lats[-1], c='red', s=100, marker='s', 
                   label='End', zorder=11)
        
        # 绘制估计轨迹（从 ECEF 转为 LLA）
        if method == 'Adaptive':
            est_ecefs = [r['position_adaptive_ecef'] for r in positioning_results]
        else:
            key = f'position_{method.lower().replace("+", "").replace("-", "_")}_ecef'
            est_ecefs = [r[key] for r in positioning_results]
        
        est_llas = [ecef_to_lla(pos) for pos in est_ecefs]
        est_lats = [lla['lat'] for lla in est_llas]
        est_lons = [lla['lon'] for lla in est_llas]
        
        ax.plot(est_lons, est_lats, '--', linewidth=1.5, alpha=0.7, label=f'{method}')
        
        # 着色标记误差大小
        errors = [r[f'error_{method.lower().replace("+", "").replace("-", "_")}_m'] 
                  for r in positioning_results]
        scatter = ax.scatter(est_lons, est_lats, c=errors, cmap='RdYlGn_r', s=50)
        plt.colorbar(scatter, ax=ax, label='Error (m)')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Trajectory: {method}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/01_trajectory_2d.png', dpi=150)
    plt.close()
```

#### 可视化 2：轨迹对比图（3D 视图）

```python
def plot_trajectory_3d(positioning_results):
    """3D 视图：ECEF 坐标下的轨迹"""
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 真值轨迹
    gt_ecefs = [r['gt_ecef'] for r in positioning_results]
    gt_x, gt_y, gt_z = zip(*gt_ecefs)
    ax.plot(gt_x, gt_y, gt_z, 'k-', linewidth=2, label='Ground Truth')
    
    # 各方法轨迹
    for method in ['Standard-LS', 'WLS-MoG', 'FG-MoG+2A']:
        est_ecefs = [r[f'position_{method.lower().replace("+", "").replace("-", "_")}_ecef'] 
                     for r in positioning_results]
        est_x, est_y, est_z = zip(*est_ecefs)
        ax.plot(est_x, est_y, est_z, '--', label=method, alpha=0.7)
    
    ax.set_xlabel('X (ECEF, km)')
    ax.set_ylabel('Y (ECEF, km)')
    ax.set_zlabel('Z (ECEF, km)')
    ax.set_title('3D Trajectory Comparison (ECEF)')
    ax.legend()
    
    plt.savefig('results/visualizations/02_trajectory_3d.png', dpi=150)
    plt.close()
```

#### 可视化 3：误差时间序列

```python
def plot_error_timeseries(positioning_results, adaptive_results):
    """各算法误差随时间变化"""
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    epochs = np.arange(len(positioning_results))
    
    ax.plot(epochs, [r['error_ls_m'] for r in positioning_results], 
            '-o', label='Standard LS', linewidth=1.5)
    ax.plot(epochs, [r['error_wls_m'] for r in positioning_results], 
            '-s', label='WLS-MoG', linewidth=1.5)
    ax.plot(epochs, [r['error_fg_m'] for r in positioning_results], 
            '-^', label='FG-MoG+2A', linewidth=1.5)
    ax.plot(epochs, [r['error_m'] for r in adaptive_results], 
            '-*', label='Adaptive', linewidth=2, markersize=8)
    
    ax.axhline(y=np.median([r['error_ls_m'] for r in positioning_results]), 
               color='k', linestyle=':', alpha=0.5, label='LS Median')
    
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('2D Position Error (m)')
    ax.set_title('Localization Error Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/03_error_timeseries.png', dpi=150)
    plt.close()
```

#### 可视化 4：误差 CDF 曲线

```python
def plot_error_cdf(localization_metrics):
    """多算法误差 CDF 对比"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = ['Standard-LS', 'WLS-MoG', 'FG-MoG+2A', 'Adaptive']
    
    for method in methods:
        errors = localization_metrics[method]['cdf_errors']
        percentiles = localization_metrics[method]['cdf_percentiles']
        ax.plot(errors, percentiles, '-', linewidth=2, label=method)
    
    # 标记关键百分位数
    ax.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax.text(50, 5, 'CEP50', fontsize=10)
    ax.axvline(x=95, color='gray', linestyle=':', alpha=0.5)
    ax.text(95, 5, 'CEP95', fontsize=10)
    
    ax.set_xlabel('2D Position Error (m)')
    ax.set_ylabel('Cumulative Probability (%)')
    ax.set_title('Error CDF Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/04_error_cdf.png', dpi=150)
    plt.close()
```

#### 可视化 5：热力图（误差空间分布）

```python
def plot_error_heatmap(val_dataset, positioning_results):
    """轨迹路径上的误差热力图"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    lats = [e['gt_lla']['lat'] for e in val_dataset]
    lons = [e['gt_lla']['lon'] for e in val_dataset]
    errors = [r['error_ls_m'] for r in positioning_results]  # LS 误差作为示例
    
    scatter = ax.scatter(lons, lats, c=errors, cmap='RdYlGn_r', s=100)
    
    # 连接轨迹
    ax.plot(lons, lats, 'k-', alpha=0.2, linewidth=1)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Localization Error (m)')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Error Heatmap Along Trajectory')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/05_error_heatmap.png', dpi=150)
    plt.close()
```

#### 可视化 6：Module 1 输出分布

```python
def plot_module1_distributions(mog_predictions, val_dataset):
    """Module 1 输出的直方图与散点图"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 提取所有输出
    all_p_los = np.concatenate([p['p_los'] for p in mog_predictions])
    all_mu_nlos = np.concatenate([p['mu_nlos'] for p in mog_predictions])
    all_sigma_los = np.concatenate([p['sigma_los'] for p in mog_predictions])
    all_sigma_nlos = np.concatenate([p['sigma_nlos'] for p in mog_predictions])
    all_labels = np.concatenate([e['nlos_labels'] for e in val_dataset])
    
    # p_los 分布
    axes[0, 0].hist([all_p_los[all_labels==0], all_p_los[all_labels==1]], 
                    bins=20, label=['LOS', 'NLOS'], alpha=0.6)
    axes[0, 0].set_xlabel('p_los')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('p_los Distribution')
    axes[0, 0].legend()
    
    # mu_nlos 分布
    axes[0, 1].hist([all_mu_nlos[all_labels==0], all_mu_nlos[all_labels==1]], 
                    bins=20, label=['LOS', 'NLOS'], alpha=0.6)
    axes[0, 1].set_xlabel('mu_nlos (km)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('mu_nlos Distribution')
    axes[0, 1].legend()
    
    # sigma 分布
    axes[0, 2].hist([all_sigma_los, all_sigma_nlos], bins=20, 
                    label=['sigma_los', 'sigma_nlos'], alpha=0.6)
    axes[0, 2].set_xlabel('Standard Deviation (km)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Sigma Distribution')
    axes[0, 2].legend()
    
    # 散点图：p_los vs mu_nlos
    axes[1, 0].scatter(all_p_los[all_labels==0], all_mu_nlos[all_labels==0], 
                       label='LOS', alpha=0.5, s=20)
    axes[1, 0].scatter(all_p_los[all_labels==1], all_mu_nlos[all_labels==1], 
                       label='NLOS', alpha=0.5, s=20)
    axes[1, 0].set_xlabel('p_los')
    axes[1, 0].set_ylabel('mu_nlos (km)')
    axes[1, 0].set_title('p_los vs mu_nlos')
    axes[1, 0].legend()
    
    # 散点图：elevation vs p_los
    all_elevations = np.concatenate([np.array([s['elevation_deg'] for s in e['satellites']]) 
                                      for e in val_dataset])
    axes[1, 1].scatter(all_elevations[all_labels==0], all_p_los[all_labels==0], 
                       label='LOS', alpha=0.5, s=20)
    axes[1, 1].scatter(all_elevations[all_labels==1], all_p_los[all_labels==1], 
                       label='NLOS', alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Elevation (degrees)')
    axes[1, 1].set_ylabel('p_los')
    axes[1, 1].set_title('Elevation vs p_los')
    axes[1, 1].legend()
    
    # 散点图：CNO vs NLOS 标签
    all_cn0 = np.concatenate([np.array([s['cn0_dbhz'] for s in e['satellites']]) 
                               for e in val_dataset])
    axes[1, 2].scatter(all_cn0[all_labels==0], all_labels[all_labels==0], 
                       label='LOS', alpha=0.5, s=20)
    axes[1, 2].scatter(all_cn0[all_labels==1], all_labels[all_labels==1], 
                       label='NLOS', alpha=0.5, s=20)
    axes[1, 2].set_xlabel('C/N0 (dBHz)')
    axes[1, 2].set_ylabel('NLOS Label')
    axes[1, 2].set_title('C/N0 vs NLOS Label')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('results/visualizations/06_module1_distributions.png', dpi=150)
    plt.close()
```

#### 可视化 7：Module 3 决策分布

```python
def plot_module3_decision_distribution(adaptive_results):
    """算法选用比例与创新值变化"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 饼图：算法选用比例
    methods = [r['method'] for r in adaptive_results]
    method_counts = {}
    for m in methods:
        method_counts[m] = method_counts.get(m, 0) + 1
    
    ax1.pie(method_counts.values(), labels=method_counts.keys(), autopct='%1.1f%%')
    ax1.set_title('Algorithm Selection Distribution')
    
    # 时间序列：创新值变化
    epochs = np.arange(len(adaptive_results))
    innovations = [r.get('innovation', 0) for r in adaptive_results]  # 需从跟踪器提取
    
    ax2.plot(epochs, innovations, '-o', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.fill_between(epochs, 0, innovations, where=np.array(innovations) < 0, 
                     alpha=0.3, color='green', label='FG better')
    ax2.fill_between(epochs, 0, innovations, where=np.array(innovations) >= 0, 
                     alpha=0.3, color='red', label='LS better')
    
    ax2.set_xlabel('Epoch Index')
    ax2.set_ylabel('Innovation (FG Error - LS Error) (m)')
    ax2.set_title('Residual Innovation Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/07_module3_decisions.png', dpi=150)
    plt.close()
```

#### 可视化 8：误差箱线图

```python
def plot_error_boxplots(comparison_metrics):
    """四种方法的误差箱线图"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = list(comparison_metrics.keys())
    errors_list = [comparison_metrics[m]['cdf_errors'] for m in methods]
    
    bp = ax.boxplot(errors_list, labels=methods, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_ylabel('2D Position Error (m)')
    ax.set_title('Error Distribution Boxplot Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/08_error_boxplot.png', dpi=150)
    plt.close()
```

**输出文件**：
```
results/visualizations/
├── 01_trajectory_2d.png
├── 02_trajectory_3d.png
├── 03_error_timeseries.png
├── 04_error_cdf.png
├── 05_error_heatmap.png
├── 06_module1_distributions.png
├── 07_module3_decisions.png
└── 08_error_boxplot.png
```

---

## 五、任务 4：横向对比实验

### 5.1 Module 1 损失函数阶段对比（实验 A）

**目标**：找出最优的训练策略

**四个方案对比**：

| 方案 | 阶段配置 | 预期优点 | 预期缺点 |
|------|---------|---------|---------|
| **A（基线）** | BCE 8 + Blend 25 + NLL 67 | 稳定、收敛好 | 三阶段转换复杂 |
| **B（纯 BCE）** | BCE 100 | 简单、快速 | NLOS 误差分布学习不足 |
| **C（纯 NLL）** | NLL 100 | 直接优化目标 | 初期梯度大、不稳定 |
| **D（两阶段）** | BCE 20 + NLL 80 | 平衡 | 中间态 |

**实现**：

```python
def experiment_training_strategy():
    """
    实验 A：四种训练策略对比
    """
    
    configs = {
        'A_baseline': {
            'pure_bce_epochs': 8,
            'blend_epochs': 25,
            'name': 'Three-stage (BCE 8 + Blend 25 + NLL 67)',
        },
        'B_pure_bce': {
            'pure_bce_epochs': 100,
            'blend_epochs': 0,
            'name': 'Pure BCE (100)',
        },
        'C_pure_nll': {
            'pure_bce_epochs': 0,
            'blend_epochs': 0,
            'name': 'Pure NLL (100)',
        },
        'D_two_stage': {
            'pure_bce_epochs': 20,
            'blend_epochs': 0,  # 直接到 NLL
            'name': 'Two-stage (BCE 20 + NLL 80)',
        },
    }
    
    results = {}
    
    for config_name, config_params in configs.items():
        print(f"\n{'='*60}")
        print(f"Training Strategy {config_name}: {config_params['name']}")
        print(f"{'='*60}")
        
        # 训练模型
        model = NLOSGATModel(...)
        
        # 使用特定的损失函数配置
        train_module1_with_strategy(
            model, 
            train_dataset, 
            val_dataset,
            strategy_config=config_params
        )
        
        # 评估
        f1, accuracy = evaluate_module1(model, val_dataset)
        
        # 运行定位
        adaptive_results = run_full_pipeline(model, val_dataset)
        metrics = compute_localization_metrics(adaptive_results)
        
        results[config_name] = {
            'f1': f1,
            'accuracy': accuracy,
            'cep50': metrics['cep50_2d'],
            'cep95': metrics['cep95_2d'],
            'mean_error': metrics['mean_error_2d'],
        }
    
    # 汇总对比表
    summary_df = pd.DataFrame(results).T
    print("\n" + "="*80)
    print("Summary: Training Strategy Comparison")
    print("="*80)
    print(summary_df)
    
    summary_df.to_csv('results/experiments/exp_A_training_strategy.csv')
    
    return results
```

**输出**：
```
results/experiments/
├── exp_A_training_strategy.csv    # 四方案的数值对比
├── exp_A_models/                  # 四个训练好的模型权重
└── exp_A_analysis.md              # 定性分析结论
```

---

### 5.2 Module 3 场景判断阈值对比（实验 B）

**目标**：找出最优的综合评分阈值

**三个阈值设置**：

```python
def experiment_threshold_sensitivity():
    """
    实验 B：阈值敏感性实验
    """
    
    thresholds = [0.50, 0.65, 0.80]  # FG 使用阈值
    
    results = {}
    
    for threshold in thresholds:
        selector = AdaptivePositionSelector(fg_threshold=threshold)
        adaptive_results = run_module3_adaptive(val_dataset, mog_predictions, 
                                               positioning_results, 
                                               selector_config={'fg_threshold': threshold})
        
        # 统计 FG 使用率
        fg_count = sum(1 for r in adaptive_results if 'FG' in r['method'])
        fg_ratio = fg_count / len(adaptive_results)
        
        # 定位精度
        metrics = compute_localization_metrics(adaptive_results)
        
        results[f'threshold_{threshold}'] = {
            'fg_usage_ratio': fg_ratio,
            'cep50': metrics['cep50_2d'],
            'cep95': metrics['cep95_2d'],
            'mean_error': metrics['mean_error_2d'],
        }
    
    # 汇总
    summary_df = pd.DataFrame(results).T
    print("Module 3 Threshold Sensitivity Analysis")
    print(summary_df)
    
    summary_df.to_csv('results/experiments/exp_B_threshold_sensitivity.csv')
    
    return results
```

---

### 5.3 Module 3 指标权重对比（实验 C）

**目标**：探索不同权重配置对性能的影响

**三种权重组合**：

```python
def experiment_weight_combinations():
    """
    实验 C：指标权重组合
    """
    
    weight_configs = {
        'uniform': {'plos_gap': 0.33, 'pdop_ratio': 0.33, 'nlos_redundancy': 0.34},
        'plos_dominant': {'plos_gap': 0.6, 'pdop_ratio': 0.2, 'nlos_redundancy': 0.2},
        'dop_dominant': {'plos_gap': 0.2, 'pdop_ratio': 0.6, 'nlos_redundancy': 0.2},
    }
    
    results = {}
    
    for config_name, weights in weight_configs.items():
        # 修改 SceneQualityDetector 的权重计算逻辑
        adaptive_results = run_module3_with_weights(val_dataset, mog_predictions,
                                                    positioning_results,
                                                    weights=weights)
        
        metrics = compute_localization_metrics(adaptive_results)
        
        results[config_name] = {
            'cep50': metrics['cep50_2d'],
            'cep95': metrics['cep95_2d'],
            'mean_error': metrics['mean_error_2d'],
        }
    
    summary_df = pd.DataFrame(results).T
    print("Module 3 Weight Combination Comparison")
    print(summary_df)
    
    summary_df.to_csv('results/experiments/exp_C_weight_combinations.csv')
    
    return results
```

---

### 5.4 滑动窗口大小对比（实验 D）

**目标**：找出最优的残差跟踪窗口长度

```python
def experiment_window_sizes():
    """
    实验 D：滑动窗口大小
    """
    
    window_sizes = [20, 50, 100, 200]
    
    results = {}
    
    for ws in window_sizes:
        selector = AdaptivePositionSelector()
        selector.tracker.window_size = ws
        
        adaptive_results = run_module3_adaptive(val_dataset, mog_predictions,
                                               positioning_results,
                                               selector=selector)
        
        # 统计
        metrics = compute_localization_metrics(adaptive_results)
        
        # 算法切换频率
        methods = [r['method'] for r in adaptive_results]
        switches = sum(1 for i in range(len(methods)-1) if methods[i] != methods[i+1])
        
        # 安全保险触发次数
        fallback_count = sum(1 for r in adaptive_results if 'fallback' in r['method'].lower())
        
        results[f'window_{ws}'] = {
            'cep50': metrics['cep50_2d'],
            'cep95': metrics['cep95_2d'],
            'mean_error': metrics['mean_error_2d'],
            'algorithm_switches': switches,
            'fallback_triggers': fallback_count,
        }
    
    summary_df = pd.DataFrame(results).T
    print("Sliding Window Size Comparison")
    print(summary_df)
    
    summary_df.to_csv('results/experiments/exp_D_window_sizes.csv')
    
    return results
```

---

### 5.5 多起点优化对比（实验 E，可选）

```python
def experiment_multistart_optimization():
    """
    实验 E：因子图优化的多起点策略
    """
    
    multistart_configs = {
        '1_start': {'num_starts': 1, 'starts': ['ls']},
        '3_starts': {'num_starts': 3, 'starts': ['ls', 'wls', 'elevation_weighted']},
        '5_starts': {'num_starts': 5, 'starts': [...]},
    }
    
    results = {}
    
    for config_name, config in multistart_configs.items():
        # 用不同的起点数运行 FG 优化
        positioning_results = run_module2_with_multistart(
            val_dataset, mog_predictions, 
            num_starts=config['num_starts']
        )
        
        # 统计 FG 解的稳定性（多起点结果的方差）
        stability_metric = compute_multistart_stability(positioning_results)
        
        metrics = compute_localization_metrics(positioning_results)
        
        results[config_name] = {
            'cep50': metrics['cep50_2d'],
            'stability': stability_metric,
        }
    
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv('results/experiments/exp_E_multistart_optimization.csv')
    
    return results
```

---

### 5.6 实验总结报告

```python
def generate_experiment_summary_report():
    """
    生成跨所有对比实验的总结报告
    """
    
    report = """
# 横向对比实验总结报告
## UrbanNav-HK_TST 数据集

### 实验 A：Module 1 损失函数阶段配置
**结论**：
  - 基线三阶段方案（A）性能最稳定，CEP50 = XX m
  - 纯 BCE（B）过度简化，NLOS 误差分布学习不足，CEP50 = YY m（+Z%）
  - 纯 NLL（C）初期不稳定，最终性能可与 A 接近，CEP50 = ...
  - 两阶段（D）是速度与精度的良好折中

**推荐**：继续使用基线三阶段配置

### 实验 B：Module 3 阈值敏感性
**结论**：
  - 阈值 0.50：FG 使用率最高，但精度反而下降（阈值太低）
  - 阈值 0.65（原设置）：平衡点，CEP50 = XX m
  - 阈值 0.80：FG 使用率降低，但总体精度无显著变化

**推荐**：保持阈值 0.65 不变

### 实验 C：指标权重组合
**结论**：
  - 均匀权重：基线，CEP50 = XX m
  - p_los 主导：过于激进，易误判，CEP50 = YY m（+Z%）
  - DOP 主导：更保守，CEP50 = WW m（-Z%）

**推荐**：DOP 主导权重配置

### 实验 D：滑动窗口大小
**结论**：
  - 窗口 20：反应太快，算法切换频繁
  - 窗口 50（原设置）：最优，CEP50 = XX m，切换次数适中
  - 窗口 100/200：反应迟缓，对场景变化不敏感

**推荐**：保持窗口大小 50

### 综合结论
在香港低 NLOS 环境下（NLOS 仅 7.4%），系统的主要改进空间来自：
1. Module 1 的分类精度提升（当前 F1 ≈ 0.8）
2. Module 3 的场景适应性调整

通过本次对比实验，确定了最优的配置组合。
    """
    
    with open('results/experiments/SUMMARY_REPORT.md', 'w') as f:
        f.write(report)
```

**输出文件**：
```
results/experiments/
├── exp_A_training_strategy.csv        # 4 种策略对比
├── exp_B_threshold_sensitivity.csv    # 3 种阈值对比
├── exp_C_weight_combinations.csv      # 3 种权重对比
├── exp_D_window_sizes.csv             # 4 种窗口大小对比
├── exp_E_multistart_optimization.csv  # 5 种起点配置对比（可选）
├── exp_A_models/                      # 四个训练好的模型
└── SUMMARY_REPORT.md                  # 汇总结论
```

---

## 六、输出文件结构

```
results/
├── metrics/
│   ├── localization_metrics.json      # 全局精度指标
│   ├── stratified_metrics.json        # 分层指标
│   ├── method_comparison.json         # 四方法对比
│   └── error_distributions.pkl
│
├── module1/
│   ├── mog_predictions.pkl            # 推理结果
│   ├── classification_metrics.json    # F1, Accuracy
│   └── model_analysis.json
│
├── module2/
│   ├── positioning_results.pkl
│   ├── method_comparison.json
│   └── error_distribution.pkl
│
├── module3/
│   ├── adaptive_results.pkl
│   ├── algorithm_distribution.json
│   └── innovation_trace.pkl
│
├── visualizations/
│   ├── 01_trajectory_2d.png           # 8 张可视化图
│   ├── 02_trajectory_3d.png
│   ├── 03_error_timeseries.png
│   ├── 04_error_cdf.png
│   ├── 05_error_heatmap.png
│   ├── 06_module1_distributions.png
│   ├── 07_module3_decisions.png
│   └── 08_error_boxplot.png
│
├── experiments/
│   ├── exp_A_training_strategy.csv
│   ├── exp_B_threshold_sensitivity.csv
│   ├── exp_C_weight_combinations.csv
│   ├── exp_D_window_sizes.csv
│   ├── exp_E_multistart_optimization.csv
│   ├── exp_A_models/
│   │   ├── strategy_A_best_model.pth
│   │   ├── strategy_B_best_model.pth
│   │   ├── strategy_C_best_model.pth
│   │   └── strategy_D_best_model.pth
│   └── SUMMARY_REPORT.md
│
├── logs/
│   ├── preprocessing.log              # 数据预处理日志
│   ├── training.log                   # Module 1 训练日志
│   └── pipeline_execution.log         # 完整流程执行日志
│
└── final_report.md                    # 总结报告（包含所有结论）
```

---

## 七、环境依赖与运行说明

### 7.1 环境要求

```
Python >= 3.8
PyTorch >= 1.9.0 with CUDA 11.x (或 CPU 版本)
PyTorch Geometric >= 2.0
NumPy >= 1.20
SciPy >= 1.7
Scikit-Learn >= 0.24
Pandas >= 1.3
Matplotlib >= 3.4
Seaborn >= 0.11
tqdm >= 4.62

可选：
pyginv >= 0.1          （用于 SP3 星历解析）
h5py >= 3.0            （用于高效数据存储）
```

### 7.2 安装

```bash
# 创建虚拟环境
conda create -n gnss-nlos-hk python=3.9
conda activate gnss-nlos-hk

# 安装 PyTorch（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 PyTorch Geometric
pip install torch_geometric

# 安装其他依赖
pip install scikit-learn pandas matplotlib seaborn tqdm pyginv h5py

# 如果使用 GPU，验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 7.3 数据准备

```bash
# 1. 下载 SP3 星历（必要）
# 从 WUM（武汉大学）下载
# http://igs.gnsswhu.cn/
# 或
# https://ftp.gfsc.esa.int/
# 文件：WUM0MGXULA_20211380200_01D_05M_ORB.SP3

# 2. 放置数据集
mkdir -p data/urbannav_hk_tst/raw
mkdir -p data/urbannav_hk_tst/processed
# 解压 UrbanNav 数据集到 data/urbannav_hk_tst/raw/

# 3. 创建输出目录
mkdir -p results/{metrics,module{1,2,3},visualizations,experiments,logs}
```

### 7.4 运行命令

```bash
# 完整流程（从数据预处理到最终报告）
python main_pipeline.py \
    --dataset urbannav_hk_tst \
    --data_path data/urbannav_hk_tst \
    --sp3_file data/sp3/WUM0MGXULA_20211380200_01D_05M_ORB.SP3 \
    --output_dir results/ \
    --mode full \
    --device cuda:0

# 仅数据预处理
python main_pipeline.py \
    --dataset urbannav_hk_tst \
    --mode preprocess_only

# 仅运行 Module 1
python main_pipeline.py \
    --dataset urbannav_hk_tst \
    --mode module1_only \
    --pretrained_model path/to/model.pth

# 运行对比实验
python run_experiments.py \
    --exp_list A,B,C,D \
    --config config_hk.yaml

# 生成可视化
python generate_visualizations.py \
    --results_dir results/
```

### 7.5 配置文件（config_hk.yaml）

```yaml
dataset:
  name: urbannav_hk_tst
  data_path: data/urbannav_hk_tst
  sp3_file: data/sp3/WUM0MGXULA_20211380200_01D_05M_ORB.SP3
  train_val_split: 0.7
  skip_rate: 0.0

preprocessing:
  time_tolerance: 0.5  # 秒
  elevation_mask_method: kd_tree_1nn
  window_size_sky_mask: 5  # KD-tree 搜索半径（km）

module1:
  model_type: gat_mog
  in_features: 11
  hidden_features: 128
  num_layers: 2
  num_heads: 8
  
  training:
    learning_rate: 5.0e-6
    batch_size: 32
    num_epochs: 100
    early_stopping_patience: 15
    
  loss:
    pure_bce_epochs: 8
    blend_epochs: 25
    lambda_bce: 0.6
    lambda_mu_direction: 1.0
    lambda_sigma_sep: 5.0

module3:
  window_size: 50
  min_history: 15
  fg_threshold: 0.70
  wls_threshold: 0.50

experiments:
  exp_a_strategies:
    - baseline
    - pure_bce
    - pure_nll
    - two_stage
  exp_b_thresholds: [0.50, 0.65, 0.80]
  exp_d_window_sizes: [20, 50, 100, 200]
```

### 7.6 输出示例

```
================================================================================
UrbanNav-HK_TST Complete Pipeline Execution
================================================================================

[Preprocessing]
✓ Loaded 787 GT epochs
✓ Loaded 705 GNSS observation epochs
✓ Time alignment: 705 epochs aligned
✓ Sky mask interpolation: KD-tree distance = 45.2 m (avg)
✓ Satellite geometry: 541 epochs with SP3 coverage
✓ Feature extraction: 11-dim features, 3426 satellites
✓ Train/Val split: 378 train, 163 val

[Module 1 - NLOS Perception]
✓ Training completed (100 epochs)
  Best Val F1: 0.812
  Best Val Accuracy: 0.803
✓ Inference on validation set: 163 epochs
  p_los gap: 0.521
  mu_nlos (NLOS samples): 0.298 km

[Module 2 - Localization]
✓ Standard LS:    CEP50 = 315.2 m, CEP95 = 847.3 m
✓ WLS-MoG:        CEP50 = 412.5 m, CEP95 = 1125.4 m
✓ FG-MoG+2A:      CEP50 = 398.7 m, CEP95 = 1098.2 m

[Module 3 - Adaptive Selection]
✓ Algorithm distribution:
  - Standard LS:     45.4%
  - LS (fallback):   28.2%
  - WLS-MoG:         15.3%
  - FG-MoG+2A:       11.1%
✓ Adaptive Result: CEP50 = 318.5 m (+1.0% vs LS)

[Evaluation Metrics]
✓ 2D localization metrics saved
✓ Stratified analysis by NLOS ratio completed
✓ 8 visualizations generated

[Experiments]
✓ Exp A (training strategy): 4 models trained
✓ Exp B (threshold sensitivity): 3 configurations tested
✓ Exp C (weight combinations): 3 configurations tested
✓ Exp D (window sizes): 4 configurations tested

================================================================================
Results Summary:
- All results saved to: results/
- Final report: results/final_report.md
- Execution time: 47 minutes 32 seconds
================================================================================
```

---

## 八、关键实现注意事项

### 8.1 SP3 星历处理

**关键**：SP3 文件必须包含数据集的时间范围

```python
def validate_sp3_coverage(sp3_file, gps_week, gps_seconds_range):
    """验证 SP3 覆盖范围"""
    reader = SP3Reader(sp3_file)
    
    # 检查是否包含所需时间范围
    sp3_gps_weeks = [eph_key[2] // 604800 for eph_key in reader.eph.keys()]
    
    if gps_week not in sp3_gps_weeks:
        raise ValueError(f"SP3 does not cover GPS week {gps_week}")
    
    print(f"✓ SP3 coverage validated: {len(reader.eph)} ephemerides loaded")
```

### 8.2 处理缺失 SP3 数据的历元

```python
def handle_missing_sp3(epoch):
    """对无 SP3 覆盖的历元的处理策略"""
    
    # 策略 1：跳过（推荐）
    if sp3_coverage_missing:
        epoch['skip'] = True
        return
    
    # 策略 2：用前一个历元的卫星位置（时间平移）
    # 策略 3：用外推（线性或高阶）
```

### 8.3 低 NLOS 比例环境的特殊处理

由于香港数据集 NLOS 仅 7.4%（原欧洲 40-50%），需要调整：

```python
def adjust_for_low_nlos():
    """
    在低 NLOS 环境下的调整
    """
    
    # 1. 损失函数权重调整
    # NLOS 样本少，考虑加权采样或过采样
    NLOS_oversample_ratio = 3.0  # NLOS 样本出现频率提升 3 倍
    
    # 2. 评估指标调整
    # 关注整体精度，不过度强调 NLOS 识别
    # 使用分层评估（按 NLOS 占比分组）
    
    # 3. Module 3 场景判断调整
    # 在低 NLOS 环境下，FG 优势不明显
    # 阈值应更保守（更容易回退到 LS）
    adapter = AdaptivePositionSelector(fg_threshold=0.80)  # 提高阈值
```

### 8.4 误差计算的数值稳定性

```python
def compute_ecef_error(pos_ecef, gt_ecef):
    """3D ECEF 误差（防止数值问题）"""
    diff = pos_ecef - gt_ecef
    error = np.linalg.norm(diff)
    
    # 防止浮点溢出
    if error > 1e6:
        print(f"Warning: Large error detected: {error:.2f} m")
    
    return error
```

---

## 九、预期结果与基准

### 9.1 预期定位精度

基于原系统在欧洲数据上的性能和香港数据的特点：

| 方法 | 欧洲数据（原项目） | 香港数据（预期） | 差异原因 |
|------|:--:|:--:|---------|
| Standard LS | ~500 m CEP50 | 300-350 m CEP50 | HK 轨迹短，几何更稳定 |
| WLS-MoG | ~750 m | 400-500 m | NLOS 少，权重优势不明显 |
| FG-MoG+2A | ~500 m | 380-450 m | 与 LS 接近（低 NLOS） |
| Adaptive-M3 | ~470 m | 310-340 m | 自适应选择的益处 |

### 9.2 Module 1 分类性能

| 指标 | 欧洲数据 | 香港数据（预期） |
|------|:--:|:--:|
| F1 | 0.85-0.91 | 0.75-0.85 |
| Accuracy | 0.84-0.87 | 0.78-0.85 |
| p_los Gap | 0.52-0.68 | 0.45-0.60 |

---

## 十、故障排查与常见问题

### Q1：SP3 文件格式错误或缺失

**症状**：`[ERROR] Failed to parse SP3 file`

**解决**：
- 确认文件后缀是 `.sp3`，不是 `.sp3.gz`
- 确认文件包含所需的 GPS 周
- 使用 `sp3check.py` 验证文件完整性

### Q2：时间对齐后历元数过少（< 400）

**症状**：仅有 300 个历元被成功对齐

**解决**：
- 检查 `time_tolerance` 参数（默认 0.5 秒）
- 检查是否有时区转换错误
- 手动检查几个 GT 和 GNSS obs 的时间戳一致性

### Q3：梯度爆炸导致 Module 1 训练失败

**症状**：loss 变为 NaN，训练中断

**解决**：
- 降低学习率（当前 5e-6，试试 1e-6）
- 增加梯度裁剪阈值（当前 10.0，试试 20.0）
- 启用梯度累积（batch_size / accumulation_steps）

### Q4：Module 3 算法切换过于频繁

**症状**：每个历元都在改变定位方法

**解决**：
- 增加滑动窗口大小（从 50 改为 100）
- 增加阈值（从 0.70 改为 0.80）
- 检查 ResidualInnovationTracker 的 innovation 计算

---

## 十一、性能优化建议

### 时间优化

| 步骤 | 当前耗时 | 优化方法 | 预期加速 |
|------|:--:|:--:|:--:|
| 数据预处理 | 5-10 min | GPU 加速 KD-tree | 2-3× |
| Module 1 训练 | 20-30 min | 混合精度 (FP16) | 1.5-2× |
| Module 2 优化 | 10-15 min | 多进程并行 | 2-4× |
| 可视化生成 | 5-10 min | 批量 Matplotlib | 1.5× |

### 内存优化

- sky_mask 文件（681 MB）可分块加载
- 用 HDF5 替代 pickle 存储（压缩率 50-70%）
- 梯度累积替代大 batch_size

---

## 十二、交付清单

**代码文件**（~30 个 Python 文件）：
- [ ] 1_preprocess.py - 数据预处理
- [ ] 2_module1_train.py - Module 1 训练
- [ ] 3_module1_inference.py - Module 1 推理
- [ ] 4_module2_localization.py - Module 2 定位
- [ ] 5_module3_adaptive.py - Module 3 自适应
- [ ] 6_evaluation_metrics.py - 指标计算
- [ ] 7_visualizations.py - 可视化生成
- [ ] 8_experiments.py - 对比实验
- [ ] main_pipeline.py - 完整流程入口
- [ ] utils/ - 辅助函数库
  - [ ] coordinate_transform.py
  - [ ] sp3_reader.py
  - [ ] time_utils.py
  - [ ] geometry_utils.py
- [ ] config_hk.yaml - 配置文件
- [ ] requirements.txt - 环境依赖

**数据文件**（生成后）：
- [ ] results/ 下所有中间和最终结果
- [ ] final_report.md - 总结报告
- [ ] SUMMARY_REPORT.md - 对比实验总结

---

*文档生成时间*：2026-06-07  
*目标受众*：Codex AI + 专业工程师  
*预期完成时间*：5-7 天（含对比实验）
# UrbanNav-HK_TST 数据集适配 & 模型输入设计
## 完整 Goal 描述（Codex 用）

**项目名称**：GNSS NLOS 识别系统 - UrbanNav-HK_TST 数据集适配  
**任务阶段**：数据预处理与模型输入设计  
**目标**：为新数据集设计完整的数据管道，生成可直接输入到 GAT+MoG 模型的训练数据集

---

## 1. 任务概览与上下文

### 1.1 背景

原项目（柏林/法兰克福数据集）已经完成了三模块系统的开发：
- **Module 1**：GAT+MoG 模型，输入 11 维特征，输出 NLOS 分类概率和误差分布参数
- **Module 2**：因子图融合定位
- **Module 3**：残差反馈自适应校正

现在需要将这套系统适配到新数据集（UrbanNav-HK_TST），以验证模型在不同城市环境中的泛化性能。

### 1.2 新数据集特点

| 特性 | 值 | 影响 |
|------|-----|------|
| 轨迹长度 | ~650 m | 小样本，仅 787 个历元 |
| 城市环境 | 香港尖沙咀超高密集 | NLOS 比例可能 > 50% |
| 接收机类型 | 10 种（专业 + 手机） | 可做多接收机对比 |
| 星座数 | GPS, GLONASS, BeiDou | 更多样本，但各星座数据量不均 |
| NLOS 标签 | 需从 sky_mask 推导 | 不是直接给出，需计算 |
| 历元不对齐 | 787 GT ≠ 705 GNSS ≠ 148,642 sky_mask | 需精细时空对齐 |

### 1.3 最终交付物

生成三个 Python 脚本 + 一个验证笔记本：

```
data/urbannav_hk_tst/
├── raw/                                # 原始数据（从数据集解压）
│   ├── ground_truth.json
│   ├── novatel_gnss_obs.json
│   ├── sky_mask.json
│   ├── imu.csv
│   └── [other files]
├── processed/                          # 预处理后的输出
│   ├── aligned_epochs.pkl             # 时空对齐后的历元
│   ├── nlos_labels.pkl                # NLOS/LOS 标签
│   ├── training_dataset.pkl           # 训练集特征矩阵
│   ├── validation_dataset.pkl         # 验证集特征矩阵
│   └── dataset_statistics.json        # 数据统计汇总
│
scripts/
├── step1_time_alignment.py            # 时间对齐
├── step2_sky_mask_interpolation.py    # 天际线插值
├── step3_satellite_geometry.py        # 仰角/方位角计算
├── step4_feature_extraction.py        # 特征提取
├── step5_train_val_split.py           # 数据划分
└── validate_preprocessing.ipynb       # 数据验证与可视化
```

---

## 2. 详细任务分解

### 2.1 任务 A：时间与空间对齐（Time & Spatial Alignment）

**目标**：将三个不同时间频率的数据源对齐到统一的时间轴

#### 2.1.1 输入

- `ground_truth.json`：787 个 GT 历元，每个有 gps_week, gps_seconds, latitude, longitude, altitude
- `novatel_gnss_obs.json`：705 个 GNSS 历元，每个有时间戳、卫星观测列表
- `imu.csv`：314,194 条 IMU 记录，UNIX epoch 时间戳
- `sky_mask.json`：148,642 条天际线记录，lat/lon/alt

#### 2.1.2 核心问题与解决方案

**问题 1：GT 有 787 个历元，GNSS 只有 705 个，为什么差异这么大？**

根据文档："NovAtel obs has gaps，~10% of epochs have no GNSS data"

解决方案：
- 对于有 GNSS 观测的历元（705 个），直接使用
- 对于无 GNSS 观测的历元，**标记为"skip"不参与训练**
- 最终训练数据集大约 705 个历元（实际可用）

**问题 2：Sky_mask 有 148,642 条记录，与 GNSS 历元怎么对应？**

根据文档："sky_mask.json: ~10 Hz along trajectory，needs spatial interpolation for matching"

解决方案：
- Sky_mask 是沿轨迹以 ~10 Hz 密集采样
- GNSS 历元只有 705 个（1 Hz），需要用空间 (lat, lon) 匹配，而不是时间匹配
- 对每个 GNSS 历元的 (lat, lon)，在 sky_mask 中找**最近的 5 条记录**（KD-tree 搜索）
- 用这 5 条记录的加权平均（距离倒数权重）插值出该历元的天际线

**问题 3：时间戳格式不统一怎么办？**

解决方案：
- ground_truth 和 novatel_gnss_obs：都有 gps_week 和 gps_seconds，直接转为"GPS 绝对秒数"
- imu.csv：用 UNIX epoch，需要转为 GPS 秒
- 转换公式：GPS_sec = UNIX_epoch_sec - 315964800（GPS epoch 1980-01-06 到 UNIX epoch 1970-01-01 的秒差）

#### 2.1.3 输出

```python
# 对齐后的单个历元结构
aligned_epoch = {
    'epoch_idx': 0,                    # 全局历元索引
    'gps_week': 2158,
    'gps_seconds': 95593.0,            # 浮点秒数
    'timestamp_unix': 1621243393.0,    # UNIX 时间戳
    
    # 从 GT 获取
    'gt_position': {
        'latitude': 22.297,            # 度
        'longitude': 114.175,          # 度
        'altitude': 3.5,               # 米
        'ecef_x': 6378000,             # 近似值，需从 LLA 转 ECEF
        'ecef_y': -2200000,
        'ecef_z': 2400000,
    },
    'gt_velocity': {
        'body_x': 8.5,                 # 车体纵向速度（前向）
        'body_y': -0.2,                # 车体横向速度（左右）
        'body_z': 0.1,                 # 车体竖向速度
    },
    'gt_attitude': {
        'roll': 1.2,                   # 度
        'pitch': -0.5,                 # 度
        'heading': 142.3,              # 度（磁方位角）
    },
    'gt_quality': 3,                   # INS 质量标志
    
    # 从 GNSS 获取
    'num_satellites': 6,               # 该历元有多少颗卫星
    'satellites': [
        {
            'system': 'G',             # GPS
            'sv_id': 1,                # 卫星编号
            'pr_m': 22456789.5,        # 伪距（米）
            'cn0_dbhz': 38.5,          # C/N0（dB-Hz）
            'doppler_hz': -1250.3,     # 多普勒（Hz）
            'raw_phase': 118045678.5,  # 原始载波相位（周期）
        },
        # ... 更多卫星
    ],
    
    # 从 sky_mask 获取
    'sky_mask': {
        'elevation_mask': [82, 82, 81, ..., 75],  # 361 个值，每个方位角 1°
        'interpolation_method': 'weighted_5nn',    # 如何从 148k 条插值的
    },
    
    # 从 IMU 获取（可选，100 Hz 数据的窗口）
    'imu_window': {
        'timestamps': [...],           # 该 1 秒内的 IMU 时间戳（~10 条）
        'gyro_xyz': [...],             # 陀螺仪角速度（rad/s）
        'accel_xyz': [...],            # 加速度计（m/s²，含重力）
        'orientation_quat': [...],     # 四元数（x, y, z, w）
    },
}
```

#### 2.1.4 实现要点

```python
# 伪代码示例
def align_epochs():
    """
    核心流程：
    1. 从 ground_truth.json 加载 787 个 GT 历元
    2. 从 novatel_gnss_obs.json 加载 705 个 GNSS 历元
    3. 对每个 GNSS 历元：
       a) 根据 gps_seconds 查找最近的 GT 历元（时间差 < 0.5 秒）
       b) 如果找到 GT，记录 GT 位置、速度、姿态、质量
       c) 如果没找到，标记为 skip
    4. 对每个成功对齐的历元：
       a) 获取该历元的 (lat, lon, alt)
       b) 在 sky_mask 中用 KD-tree 找最近的 5 条记录
       c) 用距离倒数权重插值得到 361 维的 elevation_mask
    5. 保存对齐后的历元列表
    """
    
    # 时间同步工具函数
    def gps_time_to_unix(gps_week, gps_seconds):
        """GPS 时间转 UNIX 时间戳"""
        GPS_EPOCH_UNIX = 315964800  # GPS epoch 相对于 UNIX epoch 的秒差
        return gps_week * 604800 + gps_seconds + GPS_EPOCH_UNIX
    
    def find_nearest_gt(gps_seconds, gt_list, tolerance=0.5):
        """在 GT 列表中找最近的历元（时间差 < tolerance 秒）"""
        pass
    
    # 空间插值工具函数
    def interpolate_sky_mask(lat, lon, sky_mask_list):
        """
        对给定的 (lat, lon)，在 sky_mask 中找最近的 5 条记录，
        用距离倒数加权插值出 361 维的 elevation_mask
        
        返回：elevation_mask (361,)
        """
        pass
    
    # 坐标转换工具函数
    def lla_to_ecef(latitude, longitude, altitude):
        """LLA 转 ECEF（WGS84 椭球）"""
        pass
```

---

### 2.2 任务 B：计算卫星几何与 NLOS 标签（Satellite Geometry & NLOS Labeling）

**目标**：对每颗卫星，计算仰角 (elevation) 和方位角 (azimuth)，与 sky_mask 比较得到 NLOS/LOS 标签

#### 2.2.1 关键挑战：卫星位置

**原始数据中没有卫星坐标**。文档明确说："Satellite positions: Not included in this dataset. Need SP3 ephemeris or broadcast ephemeris"

**解决方案选项**：

**选项 1（推荐）：下载 SP3 精密星历**
- 时间范围：2021-05-17（GPS week 2158）
- 来源：IGS（International GNSS Service）
- 下载链接：ftp://cddis.gsfc.nasa.gov/pub/gps/products/2158/
- 文件名格式：`final20158.sp3`（final 表示精密）
- 解析：需要 SP3 读取库（pyginv、或自己写简单解析器）
- 精度：约 5 cm

**选项 2（快速替代）：广播星历（broadcast ephemeris）
- 需要从原始 RINEX 观测中提取
- 精度较低（米级），但足以计算仰角（仰角误差 < 1°）
- 优点：包含在原始数据中（novatel_gnss_obs.json）
- 缺点：需要实现 GPS/GLONASS/BeiDou 轨道计算（复杂）

**选项 3（最简化）：近似方法
- 使用伪距 + 接收机位置推算卫星方向（不计算绝对位置）
- 公式：`unit_direction = (pr_vector) / ||pr_vector||`
- 从单位向量计算仰角和方位角
- 精度：仰角误差可能 5-10°（不理想，但可以验证 NLOS 分布）

**建议**：
- **首选**：用选项 1（下载 SP3）+ pyginv 库，最接近原项目的做法
- **备选**：如果时间紧张，用选项 3 快速验证数据管道，然后升级到选项 1

#### 2.2.2 仰角和方位角计算

```python
# 伪代码
def compute_elevation_azimuth(rx_ecef, sv_ecef):
    """
    输入：
        rx_ecef: 接收机 ECEF 位置 (3,)
        sv_ecef: 卫星 ECEF 位置 (3,)
    
    输出：
        elevation: 度数 [-90, 90]
        azimuth: 度数 [0, 360)
    """
    # 1. 计算接收机处的 ENU 坐标系
    #    E = East，N = North，U = Up（地心角向外）
    # 2. 将卫星位置变换到 ENU 坐标系
    # 3. 计算仰角：el = arctan2(U, sqrt(E² + N²))
    # 4. 计算方位角：az = arctan2(E, N)，并规范化到 [0, 360)
    pass
```

需要实现的函数：

```python
def lla_to_enu_matrix(latitude, longitude):
    """
    返回 ENU 坐标系的旋转矩阵，将 ECEF 向量变换到 ENU
    
    矩阵推导：
    ENU = R_y(-90°-lat) × R_z(lon) × ECEF
    其中 R_y, R_z 是旋转矩阵
    """
    pass

def ecef_to_enu(rx_ecef, sv_ecef, latitude, longitude):
    """
    将卫星 ECEF 相对位置变换到接收机处的 ENU 坐标系
    """
    enu_matrix = lla_to_enu_matrix(latitude, longitude)
    relative_ecef = sv_ecef - rx_ecef
    enu = enu_matrix @ relative_ecef
    return enu  # (3,)

def enu_to_elevation_azimuth(enu):
    """
    从 ENU 向量计算仰角和方位角
    """
    e, n, u = enu
    elevation_rad = np.arctan2(u, np.sqrt(e**2 + n**2))
    azimuth_rad = np.arctan2(e, n)
    
    elevation_deg = np.degrees(elevation_rad)
    azimuth_deg = np.degrees(azimuth_rad) % 360  # [0, 360)
    
    return elevation_deg, azimuth_deg
```

#### 2.2.3 NLOS 标签生成

```python
def generate_nlos_label(elevation_deg, azimuth_deg, elevation_mask):
    """
    输入：
        elevation_deg: 卫星仰角（度）
        azimuth_deg: 卫星方位角（度）
        elevation_mask: 361 维向量，elevation_mask[i] 表示方位角 i° 的建筑遮挡仰角
    
    输出：
        label: 0 (LOS) 或 1 (NLOS)
    
    逻辑：
        az_idx = int(round(azimuth_deg)) % 361
        mask_el = elevation_mask[az_idx]
        if elevation_deg >= mask_el:
            return 0  # LOS（卫星高于建筑）
        else:
            return 1  # NLOS（卫星被建筑遮挡）
    """
    pass
```

#### 2.2.4 输出格式

```python
# 每个历元的卫星列表扩展
satellite = {
    'system': 'G',
    'sv_id': 1,
    'pr_m': 22456789.5,
    'cn0_dbhz': 38.5,
    
    # ===== 新增：几何信息 =====
    'sv_ecef': {
        'x': 6378000,
        'y': -2200000,
        'z': 2400000,
    },
    'elevation_deg': 45.3,           # 仰角（度）
    'azimuth_deg': 287.2,            # 方位角（度）
    
    # ===== 新增：NLOS 标签 =====
    'nlos_label': 0,                 # 0=LOS, 1=NLOS
    'elevation_mask_at_az': 25.5,    # 该方位角的建筑遮挡仰角
    'elevation_above_mask': 19.8,    # elevation_deg - elevation_mask_at_az（可用于调试）
}
```

---

### 2.3 任务 C：特征提取与张量设计（Feature Extraction & Tensor Design）

**目标**：从原始观测和几何信息中提取特征，设计模型输入张量

#### 2.3.1 模型输入张量要求

回顾原项目的 Module 1 模型：
- **输入**：图结构 + 节点特征矩阵
- **节点特征**：11 维（见原项目 6.2 节）
- **边结构**：方位角差 < 90° 的卫星之间连边

**新数据集的张量设计**：

目标是保持与原模型的兼容性，同时利用新数据集提供的额外观测（如多普勒、原始相位）。

**特征设计方案（12 维或可扩展）**：

```python
features = [
    # ========== 原有的 11 维 ==========
    0.  elevation_deg / 90.0,              # 归一化仰角
    1.  azimuth_deg / 360.0,               # 归一化方位角
    2.  cn0_dbhz / 60.0,                   # 归一化 C/N0（60 dBHz 为上界）
    3.  (pr_stdev or 1.0) / 5.0,          # 伪距不确定度（如果有）或默认 1m
    4.  pr_m / 3e7,                        # 伪距（归一化到光速相关的数值）
    5.  pr_innovation / 100.0,             # 伪距创新量（需要计算）
    6.  np.cos(np.radians(elevation_deg)), # 仰角余弦（几何精度代理）
    7.  int(system == 'G'),                # GPS one-hot
    8.  int(system == 'R'),                # GLONASS one-hot
    9.  int(system == 'E'),                # Galileo one-hot
    10. int(system == 'C'),                # BeiDou one-hot
    
    # ========== 新增特征（可选） ==========
    11. doppler_hz / 10000.0,              # 多普勒（假设最大 ±5 kHz）
    # 12. cn0_trend / 10.0,                # C/N0 变化趋势（100ms 窗口）
    # 13. multipath_indicator / 100.0,     # 多径指示（如果有）
]
```

#### 2.3.2 关键特征的计算说明

**特征 5：伪距创新量（Pseudorange Innovation）**

定义：`pr_innovation = pr_measured - pr_predicted`

其中：
- `pr_measured`：观测到的伪距
- `pr_predicted`：从接收机位置和卫星位置预测的伪距

计算步骤：
```python
def compute_pr_innovation(pr_measured, rx_ecef, sv_ecef, clock_bias_m):
    """
    计算伪距创新量
    
    输入：
        pr_measured: 观测伪距（米）
        rx_ecef: 接收机 ECEF 位置（米）
        sv_ecef: 卫星 ECEF 位置（米）
        clock_bias_m: 接收机钟偏差（米，可从 LS 估计）
    
    输出：
        innovation: 创新量（米）
    """
    geometric_range = np.linalg.norm(sv_ecef - rx_ecef)
    pr_predicted = geometric_range + clock_bias_m
    innovation = pr_measured - pr_predicted
    return innovation
```

**问题**：钟偏差 (clock_bias) 需要先估计，这是个鸡生蛋问题。

**解决方案**：
- 对每个历元，用标准 LS 估计钟偏差（初次粗估）
- 然后用这个钟偏差计算所有卫星的 pr_innovation
- 这个循环迭代 2-3 次直到收敛

```python
def estimate_clock_bias(pr_measured, rx_ecef_approx, sv_positions, clock_bias_init=0):
    """
    迭代估计钟偏差（简化版）
    """
    clock_bias = clock_bias_init
    for iteration in range(3):
        # 计算每颗卫星的预测伪距
        pr_predicted = [np.linalg.norm(sv - rx_ecef_approx) + clock_bias 
                        for sv in sv_positions]
        # 计算残差
        residuals = pr_measured - np.array(pr_predicted)
        # 钟偏差的新估计：残差的平均值
        clock_bias = np.mean(residuals)
    return clock_bias
```

**特征 11（新增）：多普勒**

多普勒在 novatel_gnss_obs.json 中直接提供（D1C, D2W 等字段）

归一化方式：
```python
doppler_normalized = doppler_hz / 10000.0  # 假设最大 ±5 kHz
```

#### 2.3.3 节点特征矩阵的格式

对于单个历元（可变数量的卫星 N）：

```
node_features: shape (N, 12)  # N 颗卫星，12 维特征
例如：
[[0.50, 0.80, 0.64, 0.20, 0.75, -0.05, 0.88, 1.0, 0.0, 0.0, 0.0, -0.02],
 [0.30, 0.45, 0.55, 0.15, 0.72, 0.10, 0.96, 0.0, 1.0, 0.0, 0.0, 0.05],
 [0.15, 0.92, 0.45, 0.25, 0.80, -0.20, 0.99, 0.0, 0.0, 1.0, 0.0, -0.08],
 ...
]

edge_index: shape (2, E)  # E 条边，邻接矩阵的 COO 格式
例如（4 颗卫星，方位角差 < 90° 的卫星对）：
[[0, 0, 1, 1, 2, 2, 3],
 [1, 2, 0, 3, 0, 3, 1]]

nlos_labels: shape (N,)  # 每颗卫星的真值标签
例如：
[0, 1, 0, 1]  # 卫星 0 和 2 是 LOS，卫星 1 和 3 是 NLOS
```

**处理可变卫星数**（与原项目相同）：
- 使用 **Block-Diagonal Batching**：多个历元的图拼成一个大图
- 在 DataLoader 中实现 custom collate_fn
- 详见原项目的 GAT_V2025.py 中的 `custom_collate_for_batch_graphs` 函数

#### 2.3.4 输出格式

```python
# 单个历元的特征字典
epoch_features = {
    'epoch_idx': 0,
    'gps_week': 2158,
    'gps_seconds': 95593.0,
    
    'node_features': np.array([...]),  # shape (N, 12)
    'edge_index': np.array([...]),     # shape (2, E)
    'nlos_labels': np.array([...]),    # shape (N,)
    
    # 元数据（不参与训练，但用于调试和验证）
    'num_satellites': 6,
    'num_los': 4,
    'num_nlos': 2,
    'nlos_ratio': 0.333,
    'receiver_id': 'novatel.flexpak6',
    'systems': ['G', 'R', 'G', 'C', 'G', 'R'],  # 每颗卫星的星座
    'sv_ids': [1, 3, 12, 23, 7, 8],           # 每颗卫星的编号
}
```

---

### 2.4 任务 D：数据集划分与输出（Train/Val Split & Dataset Export）

**目标**：将 705 个有效历元分割为训练集和验证集，保存为易于加载的格式

#### 2.4.1 划分策略

**问题**：705 个历元非常少（原项目有 77,000 个历元），容易过拟合

**划分方案**：

```
总历元数：705（有效的 GNSS 观测）

方案 A（时间划分，推荐）：
- 前 500 个历元（约 71%）→ 训练集
- 后 205 个历元（约 29%）→ 验证集
- 优点：模拟真实部署（新位置未见过）
- 缺点：验证集较小

方案 B（随机划分）：
- 随机选择 70% → 训练集，30% → 验证集
- 优点：训练数据更充足
- 缺点：训练和验证数据来自同一轨迹，有泄露风险

方案 C（接收机划分，如果用多个接收机）：
- 7 个接收机的数据都差不多
- 可以用不同接收机做训练/验证（需要验证各接收机数据对齐）

建议：使用方案 A（时间划分）
```

#### 2.4.2 输出格式

```python
# 最终训练数据集（可以用 pickle 或 HDF5）
train_dataset = [
    epoch_features_0,
    epoch_features_1,
    ...
    epoch_features_499,  # 共 500 个历元
]

val_dataset = [
    epoch_features_500,
    epoch_features_501,
    ...
    epoch_features_704,  # 共 205 个历元
]

# 保存
pickle.dump(train_dataset, open('data/urbannav_hk_tst/processed/train_dataset.pkl', 'wb'))
pickle.dump(val_dataset, open('data/urbannav_hk_tst/processed/val_dataset.pkl', 'wb'))

# 或者用 HDF5（更高效）
h5file = h5py.File('data/urbannav_hk_tst/processed/urbannav_dataset.h5', 'w')
h5file.create_dataset('train/node_features', data=..., compression='gzip')
h5file.create_dataset('train/edge_index', data=..., compression='gzip')
h5file.create_dataset('train/nlos_labels', data=..., compression='gzip')
# 类似地保存 val/...
h5file.close()
```

#### 2.4.3 数据统计汇总

```python
# 生成统计报告
dataset_stats = {
    'total_epochs': 705,
    'train_epochs': 500,
    'val_epochs': 205,
    
    'train': {
        'total_satellites': 3156,
        'total_los': 1847,        # ~58%
        'total_nlos': 1309,       # ~42%
        'avg_satellites_per_epoch': 6.3,
        'systems': {
            'G': 1200,            # GPS
            'R': 800,             # GLONASS
            'C': 1156,            # BeiDou
        }
    },
    
    'val': {
        'total_satellites': 1289,
        'total_los': 753,         # ~58%
        'total_nlos': 536,        # ~42%
        'avg_satellites_per_epoch': 6.3,
        'systems': {
            'G': 390,
            'R': 260,
            'C': 639,
        }
    },
    
    'elevation_distribution': {
        '[0, 10)': 234,
        '[10, 20)': 456,
        '[20, 30)': 678,
        # ... etc
    },
}

# 保存为 JSON
json.dump(dataset_stats, open('data/urbannav_hk_tst/processed/dataset_statistics.json', 'w'), indent=2)
```

---

### 2.5 任务 E：数据验证与可视化（Data Validation & Visualization）

**目标**：验证数据的正确性，识别潜在的问题

#### 2.5.1 验证检查清单

```python
class DataValidator:
    """验证生成的数据集"""
    
    def check_time_alignment(self, aligned_epochs):
        """检查时间对齐是否正确"""
        # ✓ GT 和 GNSS 的时间差应该 < 0.5 秒
        # ✓ 没有时间倒序
        # ✓ IMU 数据与 GNSS 有重叠
        pass
    
    def check_satellite_geometry(self, epochs):
        """检查卫星几何计算是否合理"""
        # ✓ elevation 在 [-90, 90] 范围内
        # ✓ azimuth 在 [0, 360) 范围内
        # ✓ 仰角低的卫星 NLOS 比例应该高
        # ✓ 各星座的仰角分布差异不大
        pass
    
    def check_feature_ranges(self, node_features):
        """检查特征值范围是否合理"""
        # ✓ 所有特征大致在 [-1, 2] 范围内（主要 [0, 1]）
        # ✓ 没有 NaN 或 Inf
        # ✓ 不同星座的特征分布差异合理
        pass
    
    def check_nlos_labels(self, epochs):
        """检查 NLOS 标签质量"""
        # ✓ LOS/NLOS 比例合理（不应该全 0 或全 1）
        # ✓ elevation > elevation_mask 的卫星标签是 LOS
        # ✓ elevation < elevation_mask 的卫星标签是 NLOS
        pass
    
    def check_graph_structure(self, epochs):
        """检查图结构是否合理"""
        # ✓ edge_index 中没有自环（除非必要）
        # ✓ edge_index 是无向图（i→j 和 j→i 都有）
        # ✓ 没有重复边
        # ✓ 相邻卫星（方位角差 < 90°）都被连接
        pass
```

#### 2.5.2 可视化

```python
# 可视化示例（Jupyter notebook）

# 1. 轨迹地图
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(lons, lats, 'b-', linewidth=2, label='Vehicle trajectory')
ax.scatter(lons[::10], lats[::10], c=nlos_ratios[::10], cmap='RdYlGn_r', s=50)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('UrbanNav Hong Kong - Tsim Sha Tsui')
plt.colorbar(label='NLOS Ratio')

# 2. 仰角与 NLOS 的关系
elevation_bins = np.arange(0, 95, 5)
nlos_by_elev = []
for bin_start in elevation_bins:
    mask = (elevations >= bin_start) & (elevations < bin_start + 5)
    if mask.sum() > 0:
        nlos_by_elev.append(labels[mask].mean())
    else:
        nlos_by_elev.append(np.nan)
plt.bar(elevation_bins, nlos_by_elev, width=4)
plt.xlabel('Elevation (degrees)')
plt.ylabel('NLOS Ratio')
plt.title('NLOS Ratio vs. Elevation')

# 3. C/N0 分布
plt.hist(cn0_los, bins=20, alpha=0.5, label='LOS')
plt.hist(cn0_nlos, bins=20, alpha=0.5, label='NLOS')
plt.xlabel('C/N0 (dBHz)')
plt.ylabel('Count')
plt.legend()
plt.title('Signal Strength Distribution')

# 4. 星座对比
systems_count = pd.DataFrame({
    'GPS': [gps_los, gps_nlos],
    'GLONASS': [glonass_los, glonass_nlos],
    'BeiDou': [bd_los, bd_nlos],
}, index=['LOS', 'NLOS'])
systems_count.plot(kind='bar', stacked=True)
plt.ylabel('Count')
plt.title('Satellite Distribution by Constellation')
```

---

### 2.6 任务 F：集成与 Module 1 兼容性（Integration & Module 1 Compatibility）

**目标**：确保新数据集生成的张量能直接输入到现有的 GAT+MoG 模型

#### 2.6.1 兼容性检查

```python
def check_module1_compatibility(train_dataset, model_config):
    """
    验证数据集是否与 Module 1 兼容
    """
    
    # ✓ 特征维度匹配
    assert train_dataset[0]['node_features'].shape[1] == 12, \
        f"Expected 12 features, got {train_dataset[0]['node_features'].shape[1]}"
    
    # ✓ 标签是二分类（0 或 1）
    all_labels = np.concatenate([ep['nlos_labels'] for ep in train_dataset])
    assert set(np.unique(all_labels)) == {0, 1}, "Labels must be 0 (LOS) or 1 (NLOS)"
    
    # ✓ 图的边结构是 COO 格式且无向
    for ep in train_dataset[:10]:
        edge_index = ep['edge_index']
        assert edge_index.shape[0] == 2, "Edge index must have shape (2, E)"
        # 检查是否无向
        for i, j in edge_index.T:
            assert ((edge_index[0] == j) & (edge_index[1] == i)).any(), \
                f"Edge ({i}, {j}) exists but ({j}, {i}) does not"
    
    print("✓ Dataset is compatible with Module 1")
```

#### 2.6.2 迁移学习考虑

原项目的模型在欧洲城市（柏林、法兰克福）训练，现在用在香港数据上。

**可能的性能下降原因**：
- 建筑风格不同（香港高密集摩天楼 vs 欧洲混合）
- 卫星可见性分布不同（纬度差异）
- 接收机类型不同（可能用了手机）

**应对方案**：
- **方案 A**：Fine-tune 模型（用新数据集的前 400 个历元训练，保留后 305 个做验证）
- **方案 B**：从零开始训练新模型（需要 more data，705 个历元较少）
- **方案 C**：混合训练（用原项目的 77,000 个历元 + 新数据集的 705 个历元，加权采样）

**建议**：使用方案 A（Fine-tune），学习率设为原始的 0.1 倍，防止灾难性遗忘

---

## 3. 实现路线图与依赖

### 3.1 Python 依赖

```
numpy >= 1.20
scipy >= 1.7
pandas >= 1.3
scikit-learn >= 0.24
torch >= 1.9
torch_geometric >= 2.0  # 用于 GNN 计算
h5py >= 3.0            # 可选，用于高效存储大数据集
pyginv >= 0.1          # 用于 SP3 星历解析（可选，如果不用可自己写解析器）
```

### 3.2 脚本执行顺序

```
1. step1_time_alignment.py
   ├─ 输入：raw/ground_truth.json, raw/novatel_gnss_obs.json, raw/sky_mask.json, raw/imu.csv
   └─ 输出：processed/aligned_epochs.pkl
           ├─ 每个历元包含对齐的 GT、GNSS、sky_mask 信息
           └─ 共 705 个历元（无 GNSS 观测的 82 个 GT 历元被排除）

2. step2_sky_mask_interpolation.py
   ├─ 输入：processed/aligned_epochs.pkl
   └─ 输出：processed/aligned_epochs.pkl（更新）
           └─ 每个历元新增 sky_mask.elevation_mask 字段（361 维）

3. step3_satellite_geometry.py（关键）
   ├─ 输入：processed/aligned_epochs.pkl, SP3 星历文件（或广播星历）
   └─ 输出：processed/nlos_labels.pkl
           ├─ 每个历元的每颗卫星新增：
           │  ├─ sv_ecef (3,)
           │  ├─ elevation_deg
           │  ├─ azimuth_deg
           │  └─ nlos_label
           └─ 共 705 个历元

4. step4_feature_extraction.py
   ├─ 输入：processed/nlos_labels.pkl
   └─ 输出：processed/training_dataset.pkl
           ├─ 每个历元包含：
           │  ├─ node_features (N, 12)
           │  ├─ edge_index (2, E)
           │  └─ nlos_labels (N,)
           └─ 共 705 个历元

5. step5_train_val_split.py
   ├─ 输入：processed/training_dataset.pkl
   └─ 输出：processed/train_dataset.pkl（500 epochs）
           processed/val_dataset.pkl（205 epochs）
           processed/dataset_statistics.json

6. validate_preprocessing.ipynb
   ├─ 输入：processed/train_dataset.pkl, processed/val_dataset.pkl
   ├─ 验证数据质量
   └─ 输出：可视化图表
```

### 3.3 时间估计

| 步骤 | 工作量 | 时间（小时） |
|------|--------|:----------:|
| 1. 时间对齐 | 中等（坐标转换、KD-tree） | 2-3 |
| 2. Sky_mask 插值 | 低 | 1-1.5 |
| 3. 卫星几何（关键） | 高（需 SP3 解析或星历计算） | 3-4 |
| 4. 特征提取 | 低（直接特征化） | 1-1.5 |
| 5. 数据划分 | 很低 | 0.5 |
| 6. 验证与可视化 | 中等 | 1.5-2 |
| **总计** | | **9-12** |

---

## 4. 关键决策点与替代方案

### 决策 1：星历来源（SP3 vs 广播星历 vs 近似）

| 方案 | 精度 | 难度 | 推荐度 |
|------|:---:|:---:|:----:|
| **SP3（精密）** | ~5 cm | 中 | ⭐⭐⭐ 首选 |
| 广播星历 | 1-5 m | 高 | ⭐⭐ 次选 |
| 近似（伪距推导） | 5-10° elevation error | 低 | ⭐ 快速验证用 |

**建议**：使用 SP3，可从 IGS 免费下载

### 决策 2：特征维度（11 维 vs 12 维 vs 更多）

| 方案 | 特征 | 优点 | 缺点 |
|------|:---:|:----:|:----:|
| **11 维（与原模型同）** | 无多普勒 | 完全兼容，可直接用原模型 | 丢失多普勒信息 |
| **12 维（推荐）** | +多普勒 | 利用新数据集的多观测 | 需微调原模型输入头 |
| **15 维+** | +多径、C/N0 趋势、等 | 信息最丰富 | 模型改动大，可能过拟合（数据少） |

**建议**：用 12 维（11 原有 + 多普勒），改动最小

### 决策 3：数据划分（时间 vs 随机 vs 多接收机）

| 方案 | 说明 | 优点 | 缺点 |
|------|:---:|:----:|:----:|
| **时间划分（推荐）** | 前 500 训练，后 205 验证 | 模拟真实部署 | 验证集小（205） |
| 随机划分 | 70% 训练，30% 验证 | 统计性更好 | 泄露风险（同轨迹） |
| 多接收机 | 某些接收机训练，其他验证 | 跨接收机泛化 | 需更多工作验证对齐 |

**建议**：时间划分，因为有轨迹的自然顺序

---

## 5. 预期输出与验收标准

### 5.1 最终文件清单

```
data/urbannav_hk_tst/processed/
├── aligned_epochs.pkl              # 中间产物，可删除
├── nlos_labels.pkl                 # 中间产物，可删除
├── train_dataset.pkl               # ✓ 训练集（500 历元）
├── val_dataset.pkl                 # ✓ 验证集（205 历元）
├── dataset_statistics.json         # ✓ 数据统计
└── preprocessing_log.txt           # ✓ 预处理日志

scripts/
├── step1_time_alignment.py
├── step2_sky_mask_interpolation.py
├── step3_satellite_geometry.py
├── step4_feature_extraction.py
├── step5_train_val_split.py
├── validate_preprocessing.ipynb
└── utils/
    ├── coordinate_transforms.py    # LLA↔ECEF, ENU 等
    ├── sp3_reader.py              # SP3 星历解析（或用 pyginv）
    └── geometry_utils.py           # 仰角/方位角计算
```

### 5.2 验收标准

✓ **数据完整性**
- [ ] 705 个历元全部成功对齐
- [ ] 每个历元有 3-10 颗卫星观测
- [ ] 所有特征值在 [-1, 2] 范围内，无 NaN/Inf
- [ ] NLOS 标签与 elevation_mask 逻辑一致（elevation >= mask → LOS）

✓ **特征质量**
- [ ] 仰角和方位角计算正确（与 sky_mask 对应）
- [ ] elevation_deg 在 [-90, 90] 范围内
- [ ] 仰角低（< 20°）的卫星 NLOS 比例 > 60%
- [ ] 仰角高（> 60°）的卫星 NLOS 比例 < 10%

✓ **图结构正确**
- [ ] edge_index 是无向图（对称）
- [ ] 邻接条件正确：仅 azimuth_diff < 90° 的卫星连边
- [ ] 没有孤立节点（每颗卫星至少与 1 颗卫星连接，除非只有 1 颗卫星）

✓ **数据集质量**
- [ ] 训练集 500 个历元，验证集 205 个历元
- [ ] LOS/NLOS 比例合理（不是极端 0/1）
- [ ] 各星座（GPS, GLONASS, BeiDou）都有足够样本
- [ ] 无时间倒序或重复

✓ **可复现性**
- [ ] 所有脚本可独立运行
- [ ] 结果可重现（固定随机种子）
- [ ] 有详细的日志和错误处理

---

## 6. 开发注意事项

### 6.1 常见坑

**坑 1：时间戳单位混乱**
- ground_truth.json 中是浮点 GPS 秒数（95593.123）
- imu.csv 中可能是 UNIX epoch 时间戳（1621243393.456）
- 需要显式转换，并在所有地方验证单位

**坑 2：坐标系错误**
- ECEF 中计算仰角时需要正确的 ENU 变换矩阵
- 仰角的符号容易搞反（-90° 到 +90° 还是 0° 到 180°？）
- 建议用 `scipy.spatial.transform.Rotation` 库做旋转，不要手写矩阵

**坑 3：Sky_mask 空间插值**
- 148,642 条记录在 681 MB，直接加载可能内存爆炸
- 建议用 KD-tree 或 Ball-tree 加速搜索（scipy.spatial.cKDTree）
- 如果内存还是不够，考虑分批处理（每 100 个历元加载一次 sky_mask）

**坑 4：多个接收机的对齐**
- 不同接收机可能有不同的观测卫星集合
- 建议一开始先用主接收机（novatel.flexpak6）做数据管道，验证正确性
- 再扩展到其他接收机

**坑 5：星历精度不足导致仰角计算错误**
- 如果用广播星历而非 SP3，卫星位置误差可能 1-5 m
- 这会导致仰角计算误差 0.5-2°
- 对于低仰角卫星（10° 附近），这个误差可能导致 NLOS 标签反转
- 一开始调试时，用 SP3；如果时间紧张，再降级到广播星历

---

## 7. 后续与原项目 Module 1 的联系

### 7.1 训练方式

```python
# 新数据集的训练 flow
train_dataset = pickle.load(open('processed/train_dataset.pkl', 'rb'))
val_dataset = pickle.load(open('processed/val_dataset.pkl', 'rb'))

# 使用与原项目相同的 DataLoader 和 Block-Diagonal Batching
from model.part1_GAT.model.GAT_V2025 import NLOSGATModel, train_epoch, evaluate

model = NLOSGATModel(
    in_features=12,  # 改为 12 维（原为 11）
    hidden_features=128,
    num_layers=2,
    num_heads=8,
)

# Fine-tune 模式（加载原模型权重，学习率降低）
checkpoint = torch.load('model/part1_GAT/result/exp_051/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # strict=False 容许层数差异

optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)  # 原学习率 5e-5，现在 5e-6

for epoch in range(50):  # 少数 epoch 即可（只需 fine-tune，不需从零训练）
    loss = train_epoch(model, optimizer, train_dataloader, device, epoch, config)
    val_f1 = evaluate(model, val_dataloader, device)
    print(f"Epoch {epoch}: Val F1 = {val_f1:.3f}")
```

### 7.2 预期性能

基于迁移学习理论，预期性能：

```
情景 1：直接用原 Module 1（exp_051）在新数据测试
- 预期 F1：0.75-0.80（有 5-10% 性能下降）
- 原因：新城市、新接收机、新信号特性不同

情景 2：Fine-tune 原 Module 1（本任务的数据集）
- 预期 F1：0.82-0.88（恢复到接近原性能）
- 前提：fine-tune 50-100 个 epoch，学习率低

情景 3：从零训练新模型（705 个历元很少）
- 预期 F1：0.70-0.75（容易过拟合）
- 不推荐，除非数据量增加到 > 2000 个历元
```

---

## 总结

此任务的核心是**将非结构化的 UrbanNav 原始数据转化为 Module 1 模型的输入格式**。

**关键难点**（按难度排序）：
1. **卫星位置计算**（需 SP3 或星历）→ 决定仰角/方位角精度
2. **时空对齐**（三个不同频率的数据源）→ 数据完整性的基础
3. **Sky_mask 插值**（148k 条密集数据）→ 计算效率问题
4. **特征提取与标准化**（多种观测类型）→ 模型输入质量

**推荐执行顺序**：
1. 从简单开始（时间对齐 → sky_mask 插值）
2. 逐步增加复杂度（星历计算 → 特征提取）
3. 最后验证与可视化

**质量检验**：
- 时刻检查数据范围、分布、一致性
- 用可视化（轨迹图、elevation-NLOS 关系、星座分布）验证合理性
- 与原项目的数据特性做对比（LOS/NLOS 比例、特征统计量）

---

*文档生成时间*：2026-06-07  
*目标受众*：Codex AI 编码助手  
*格式*：详细任务分解，便于 AI 阅读和逐步编码
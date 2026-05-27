# RadioGAT — GNSS NLOS 信号软误差感知模块 (PI-PEM)

## 1. 项目概述

本项目是 **PI-PEM（NLOS 感知与误差分布建模）** 框架的模块一实现，将原 RadioGAT 的 2D 无线电图重建改造为 **GNSS NLOS 信号软误差感知**。

### 核心功能

对每个 GNSS 历元的每颗可见卫星，基于 11 维节点特征和图注意力网络（GAT），输出混合高斯分布参数三元组：

```
(p_i^LOS, μ_i^NLOS, σ_i^NLOS)
```

- **p_i^LOS**：卫星 i 的信号为 LOS 的概率 ∈ [0, 1]
- **μ_i^NLOS**：若为 NLOS，伪距误差的均值（km），恒 ≥ 0
- **σ_i^NLOS**：若为 NLOS，伪距误差的标准差（km），恒 > 0

### 物理模型

```
p(error_i | θ) = p_i^LOS · N(0, σ_LOS) + (1 - p_i^LOS) · N(μ_i^NLOS, σ_i^NLOS)
```

其中 `σ_LOS = 3.0 km` 为固定假设值（LOS 误差标准差）。

---

## 2. 代码结构

所有代码位于 `model/part1_GAT/RadioGAT-Multi-band-Radiomap-Reconstruction/` 目录下。

| 文件 | 行数 | 功能 | 数据流位置 |
|------|------|------|-----------|
| `config.py` | ~170 | 集中配置管理（路径、超参数、常量），`get_config()` 工厂函数 | 全局 |
| `sp3_reader.py` | ~160 | SP3 精密星历独立解析器，含地球自转校正 | 步骤2：卫星位置获取 |
| `Data_read.py` | ~410 | GNSS CSV 加载 + SP3 解析 + 时间同步 + 伪距误差计算、缓存管理 | 步骤1：数据加载与预处理 |
| `Radio_Depth_Generate.py` | ~100 | 卫星几何计算（ECEF/LLA 转换、仰角/方位角、几何距离） | 步骤2：卫星几何计算 |
| `NodeFeature_Generate.py` | ~170 | 从 EpochData 提取 11 维固定节点特征 | 步骤3：节点特征工程 |
| `Depth_Adj_Generate.py` | ~120 | 基于方位角相近度构建图邻接矩阵 | 步骤4：图结构构建 |
| `GAT_V2025.py` | ~760 | **主文件**：模型定义 + 损失函数 + 训练循环 + 评估 + 主入口 | 步骤5：模型训练与评估 |
| `run_full_training.py` | ~30 | 一键启动脚本：数据处理 + 全量训练 | 入口脚本 |

### 5 步流水线

```
原始 CSV 数据
    │
    ▼
[步骤1] Data_read.py          ── 加载 RXM-RAWX + NAV-POSLLH + SP3
    │                             时间同步、伪距误差计算、去均值、缓存 .pkl
    ▼
[步骤2] Radio_Depth_Generate.py ── LLA↔ECEF 转换、仰角/方位角/几何距离
    │
    ▼
[步骤3] NodeFeature_Generate.py ── 提取 (N, 11) 特征矩阵（11维固定，禁止动态扩展）
    │
    ▼
[步骤4] Depth_Adj_Generate.py  ── 方位角差 < 90° → 双向边，边权重 = az_diff/90
    │
    ▼
[步骤5] GAT_V2025.py           ── GAT 模型 + NLL+BCE 损失 + 训练循环
    │
    ▼
输出: (p_los, mu_nlos, sigma_nlos) 三元组
```

---

## 3. 数据格式说明

### 3.1 输入数据目录结构

```
data/dataset/{scene_name}/
├── RXM-RAWX.csv       # 卫星原始测量（每行一颗卫星，同历元多行共享GT列）
├── NAV-POSLLH.csv      # 地面真值位置（5Hz采样）
└── *.sp3               # 精密星历（如 gbm19001.sp3）
```

### 3.2 RXM-RAWX.csv 关键列

| 列名 | 含义 | 单位 |
|------|------|------|
| GPSWeek | GPS 周数 | - |
| GPSSeconds | GPS 秒数 | s |
| gnssId | 星座类型（GPS/Glonass/Galileo/BeiDou） | - |
| svId | 卫星编号 | - |
| prMes | 伪距测量值 | m |
| cno | 载噪比 C/N₀ | dBHz |
| prStdev | 伪距标准差（接收机报告） | m |
| NLOS | NLOS 标签（第34列，0=LOS, 1=NLOS） | - |

### 3.3 处理后数据格式

处理后数据缓存为 `data/processedData/{scene_name}_processed.pkl`，包含 `List[EpochData]`：

```python
EpochData:
    gps_week: int           # GPS 周
    gps_seconds: float      # GPS 秒
    gt_lat: float           # 真值纬度
    gt_lon: float           # 真值经度
    gt_height: float        # 真值高度
    observations: List[GNSSObservation]

GNSSObservation:
    gnss_id: str            # 'GPS' / 'Glonass' / 'Galileo' / 'BeiDou'
    sv_id: int              # 卫星编号
    pr_mes: float           # 观测伪距 (m)
    cno: float              # 载噪比 (dBHz)
    pr_stdev: float         # 伪距标准差 (m)
    nlos_label: int         # 0=LOS, 1=NLOS
    elevation: float        # 仰角 (deg)
    azimuth: float          # 方位角 (deg)
    pseudorange_error: float # 去均值伪距误差 (km)
```

---

## 4. 模型架构

### 4.1 网络结构

```
输入: (N, 11) 节点特征矩阵
     │
     ▼
[输入投影]: Linear(11 → 128) + ReLU + Dropout(0.1)
     │
     ▼
[GAT 层 × 2]:
  每层: GATLayer(128 → 128, heads=8, concat=False)
       + ELU + LayerNorm + Dropout(0.1)
     │
     ▼
[输出投影]: Linear(128 → 128) + ReLU + Dropout(0.1)
     │
     ├── p_los_head:   Linear(128 → 1) → Sigmoid           → p(LOS) ∈ [0, 1]
     ├── mu_nlos_head: Linear(128 → 1) → Softplus           → μ(NLOS) ≥ 0
     └── sigma_nlos_head: Linear(128 → 1) → Softplus + σ_min → σ(NLOS) ∈ [σ_min, σ_max]
```

### 4.2 GATLayer（自定义实现）

- **非标准注意力**：使用 `softmax` 对固定的可学习注意力向量 `att ∈ R^{2*out_features}` 进行归一化，生成每头标量权重
- **聚合方式**：加权求和（非拼接），`concat=False` 时对多注意力头取均值
- **不依赖 torch-geometric**：纯 PyTorch 实现，无额外依赖

### 4.3 邻接矩阵构建

- 方位角差 `|az_i - az_j| < 90°` → 添加双向边
- 边权重 = `az_diff / 90°` ∈ [0, 1]，越小越相关
- 无有效边时自动添加自环（防止 GAT 梯度消失）

---

## 5. 11维节点特征

| 维度 | 特征 | 归一化 | 物理意义 |
|------|------|--------|----------|
| 0 | elevation | ÷ 90.0 | 低仰角 → 更可能被遮挡 |
| 1 | azimuth | ÷ 360.0 | 方向信息 |
| 2 | C/N₀ | ÷ 60.0 | 信号质量 |
| 3 | prStdev | ÷ 5.0 | 测量不确定度 |
| 4 | prMes | ÷ 3×10⁷ | 观测距离量级 |
| 5 | prInnovation | ÷ 100.0 | 伪距误差（最强NLOS信号） |
| 6 | cos(elevation) | - | 几何精度代理 |
| 7 | GPS one-hot | 0/1 | 星座类型 |
| 8 | Glonass one-hot | 0/1 | |
| 9 | Galileo one-hot | 0/1 | |
| 10 | BeiDou one-hot | 0/1 | |

> **硬约束**：特征维度固定为 11，禁止动态扩展。NLOS 标签不进入特征矩阵。

---

## 6. 损失函数

### 6.1 总损失

```
L_total = L_NLL + λ_bce · L_BCE
```

- **L_NLL**：加权混合高斯负对数似然，使用 `logsumexp` 数值稳定计算
- **L_BCE**：二分类交叉熵辅助损失（λ_bce=0.3），防止 p_los 坍缩
- **权重**：NLOS 样本权重 = `pos_weight`（≈1.07，自动从数据估算）

### 6.2 数值保护

- p_los：clamp 到 [ε, 1-ε]，ε=1e-6
- σ_nlos：clamp 到 [σ_min, σ_max]，σ_min=5.0, σ_max=10.0 km
- 标准化残差 z：clamp 到 [-200, 200] 防止 exp 溢出
- 标签平滑：BCE 目标平滑 0.05

---

## 7. 使用方法

### 7.1 环境依赖

```txt
torch >= 2.0.0
numpy
pandas
scipy
```

### 7.2 快速开始

```bash
# 一键运行：处理所有数据集 + 20 epoch 训练
cd model/part1_GAT/RadioGAT-Multi-band-Radiomap-Reconstruction
python run_full_training.py
```

### 7.3 分步运行

```bash
# 步骤1：单独处理某个数据集并查看统计
python Data_read.py

# 步骤2：验证 SP3 解析
python sp3_reader.py

# 步骤3：测试节点特征提取
python NodeFeature_Generate.py

# 步骤4：测试图构建
python Depth_Adj_Generate.py

# 步骤5：完整训练
python GAT_V2025.py
```

### 7.4 配置修改

通过 `get_config()` 工厂函数覆盖默认值：

```python
from config import get_config

# 修改超参数
config = get_config(
    NUM_EPOCHS=30,
    LEARNING_RATE=5e-5,
    LAMBDA_BCE=0.5,
    SIGMA_LOS=2.0,
    AZIMUTH_THRESHOLD=60,
)
```

或在 `run_full_training.py` 中修改 `get_config()` 的参数。

---

## 8. 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SIGMA_LOS` | 3.0 km | LOS 误差标准差（固定假设） |
| `SIGMA_MIN` | 5.0 km | σ_nlos 下界 |
| `SIGMA_MAX` | 10.0 km | σ_nlos 上界（防止吞并 LOS 分量） |
| `LAMBDA_BCE` | 0.3 | BCE 辅助损失权重 |
| `LABEL_SMOOTHING` | 0.05 | BCE 标签平滑 |
| `POS_WEIGHT` | 1.07 | NLOS 类别权重 |
| `HIDDEN_FEATURES` | 128 | GAT 隐藏维度 |
| `NUM_HEADS` | 8 | 注意力头数 |
| `NUM_LAYERS` | 2 | GAT 层数 |
| `DROPOUT` | 0.1 | Dropout 率 |
| `LEARNING_RATE` | 1e-4 | 初始学习率 |
| `GRADIENT_CLIP` | 2.0 | 梯度裁剪阈值 |
| `GRADIENT_ACCUMULATION` | 4 | 梯度累积步数 |
| `BATCH_SIZE` | 1 | 强制=1（每历元卫星数不同） |
| `AZIMUTH_THRESHOLD` | 90° | 邻接矩阵边阈值 |
| `EARLY_STOPPING_PATIENCE` | 20 | 早停耐心值 |

---

## 9. 输出结果

训练结果保存在 `model/part1_GAT/result/exp_XXX/` 目录下：

```
result/exp_001/
├── best_model.pth              # 验证集最佳模型（含 optimizer state）
├── final_model.pth             # 训练结束时的最终模型
├── checkpoints/
│   ├── checkpoint_epoch_10.pth # 每 10 epoch 的检查点
│   └── checkpoint_epoch_20.pth
├── tensorboard/                # TensorBoard 事件文件
│   └── events.out.tfevents.*
└── predictions/                # 预留：预测输出目录
```

### 9.1 TensorBoard 可视化

训练过程中实时记录以下指标到 TensorBoard：

| 分组 | 指标 | 说明 |
|------|------|------|
| Train/ | Loss, NLL, BCE | 训练损失及分量 |
| Train/ | GradNorm_Before, GradNorm_After | 梯度裁剪前后范数 |
| Train/ | p_LOS_avg, mu_NLOS_avg, sigma_NLOS_avg | 输出分布均值 |
| Train/ | NaN_Batches | NaN 批次计数 |
| Val/ | Loss, NLL | 验证损失 |
| Val/ | Accuracy, F1, Precision, Recall | 分类指标 |
| LR | (学习率曲线) | 学习率调度 |
| Gradients/ | (每 N epoch) | 梯度直方图 |
| Weights/ | (每 N epoch) | 权重直方图 |

启动 TensorBoard：

```bash
tensorboard --logdir=model/part1_GAT/result/exp_001/tensorboard
```

训练开始时控制台会打印完整的 `--logdir` 路径，直接复制使用。

模型文件为 PyTorch checkpoint 格式：

```python
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# checkpoint 还包含: epoch, val_loss, val_metrics, optimizer_state_dict
```

---

## 10. 设计决策记录

### 10.1 为什么使用 BCE 辅助损失？

混合高斯 NLL 存在**梯度饥饿**问题：当 p_los → 1 时，(1-p_los) → 0，NLOS 头几乎收不到梯度。BCE 辅助损失直接监督 p_los 的二分类任务，提供稳定的分类梯度信号。

### 10.2 为什么 σ_max = 10.0？

若 σ_nlos 过大（> 15 km），NLOS 高斯分量退化为近似均匀分布，可以"吞噬"几乎任何误差，NLL 会推动 p_los → 0。σ_max 上界防止此坍缩模式。

### 10.3 为什么 batch_size = 1？

每个 GNSS 历元的可见卫星数不同（通常 6~25 颗），无法组成等长 batch。梯度累积（accumulation=4）模拟大 batch 的稳定性。

### 10.4 为什么使用自定义 GATLayer？

保留原 RadioGAT 项目风格，轻量级实现，不引入 torch-geometric 及 torch-scatter、torch-sparse 等重型依赖链。

### 10.5 为什么 σ_LOS = 3.0 km？

实际 LOS 伪距误差的 std 约 ~30km（包括未建模的钟差、电离层等），但每历元去均值后降至约 3 km。该值匹配处理后的数据分布。

---

## 11. 评估指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| NLL Loss | 负对数似然（主要指标） | 验证集加权 NLL |
| Accuracy | NLOS 二分类准确率 | p_nlos > 0.5 为 NLOS |
| Precision | NLOS 精确率 | TP / (TP + FP) |
| Recall | NLOS 召回率 | TP / (TP + FN) |
| F1 | 调和平均 | 2PR / (P+R) |
| Grad Norm | 梯度范数 | 裁剪前后分别记录 |

---

## 原始项目致谢

本项目基于 RadioGAT 改造：
> X. Li et al., "RadioGAT: A Joint Model-Based and Data-Driven Framework for Multi-Band Radiomap Reconstruction via Graph Attention Networks," IEEE Transactions on Wireless Communications, vol. 23, no. 11, pp. 17777-17792, Nov. 2024

原项目作者联系：xiaojieli@nuaa.edu.cn 或 xiaojieli@seu.edu.cn

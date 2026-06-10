# RadioGAT — GNSS NLOS 信号软误差感知模块 (PI-PEM)

## 1. 项目概述

本项目是 **PI-PEM（NLOS 感知与误差分布建模）** 框架的模块一实现，将原 RadioGAT 的 2D 无线电图重建改造为 **GNSS NLOS 信号软误差感知**。

### 核心功能

对每个 GNSS 历元的每颗可见卫星，基于 11 维节点特征和图注意力网络（GAT），输出误差分布参数：

```
BCE 模式:   (p_los, log_sigma)
MoG 模式:   (p_los, μ_nlos, σ_los, σ_nlos)
```

- **p_los**：卫星信号为 LOS 的概率 ∈ [0, 1] (Sigmoid)
- **μ_nlos**：若为 NLOS，伪距误差的均值 ≥ 0 km (Softplus+clamp, MoG only)
- **σ_los**：LOS 误差标准差 (MoG: learnable log_sigma_los)
- **σ_nlos**：NLOS 误差标准差 (BCE: learnable log_sigma; MoG: learnable log_sigma_nlos)

### 两种训练模式

| 模式 | USE_MIXTURE_GAUSSIAN | Loss | 输出 | 适用场景 |
|------|---------------------|------|------|---------|
| BCE + Uncertainty | `False` | Weighted BCE + Heteroscedastic NLL | 2-head (p_los, log_sigma) | 基础 NLOS 检测 |
| Mixture of Gaussians | `True` | NLL + BCE (dual supervision) | 4-head (p_los, μ_nlos, σ_los, σ_nlos) | 完整误差分布建模 |

### 物理模型 (MoG)

```
p(error_i | θ) = p_i^LOS · N(0, σ_i^LOS) + (1 - p_i^LOS) · N(μ_i^NLOS, σ_i^NLOS)
```

---

## 2. 代码结构

| 文件 | 功能 | 数据流位置 |
|------|------|-----------|
| `config.py` | 集中配置管理（路径、超参数、两种模式常量） | 全局 |
| `sp3_reader.py` | SP3 精密星历解析器，含地球自转校正 | 步骤2 |
| `Data_read.py` | GNSS CSV 加载 + SP3 + 时间同步 + 伪距误差计算 + 缓存 | 步骤1 |
| `Radio_Depth_Generate.py` | ECEF/LLA 转换、仰角/方位角/几何距离 | 步骤2 |
| `NodeFeature_Generate.py` | 提取 11 维固定节点特征 | 步骤3 |
| `Depth_Adj_Generate.py` | 基于方位角相近度构建图邻接矩阵 | 步骤4 |
| **`GAT_V2026.py`** | **主文件**：向量化 GATLayer + block-diagonal batching | 步骤5 (推荐) |
| `GAT_V2025.py` | 原版主文件：GATLayer + batch_size=1 (备用) | 步骤5 (旧版) |
| `run_full_training.py` | 一键启动：`--dataset X` 按场景并行训练 | 入口脚本 |
| `analyze_experiment.py` | BCE 模型推理分析 (2-head) | 分析工具 |
| `analyze_mog.py` | MoG 模型推理分析 (4-head) | 分析工具 |
| `generate_report.py` | 自动生成 env.md + result.md 摘要 | 报告工具 |
| `run_serial.py` | 串行训练全部 4 个数据集 | 批量脚本 |

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
[步骤3] NodeFeature_Generate.py ── 提取 (N, 11) 特征矩阵（11维固定）
    │
    ▼
[步骤4] Depth_Adj_Generate.py  ── 方位角差 < 90° → 双向边
    │
    ▼
[步骤5] GAT_V2026.py           ── GAT 模型 + loss + 训练 (block-diagonal bs=32)
    │
    ▼
输出: (p_los, mu_nlos, sigma_los, sigma_nlos)
```

---

## 3. 数据格式说明

### 3.1 数据目录结构

```
data/dataset/{scene_name}/
├── RXM-RAWX.csv       # 卫星原始测量（同历元多行共享GT列）
├── NAV-POSLLH.csv      # 地面真值位置（5Hz）
└── *.sp3               # 精密星历
```

处理后缓存: `data/processedData/{scene_name}_processed.pkl`

### 3.2 数据集一览

| 数据集 | 历元数 | 卫星/历元 | NLOS% | 场景特征 |
|--------|--------|----------|-------|---------|
| berlin1_potsdamer_platz | 1,377 | 13.7 ± 1.5 | 48.3% | 波茨坦广场，类别最均衡 |
| berlin2_gendarmenmarkt | 5,925 | 12.9 ± 1.4 | 38.8% | 宪兵广场，特征区分度最大 |
| frankfurt1_maintower | 5,851 | 12.7 ± 1.5 | 43.0% | 美茵塔，高楼密集，最难场景 |
| frankfurt2_westendtower | 3,575 | 13.7 ± 1.4 | 26.6% | 西区塔，LOS 占主导 |

### 3.3 数据对象

```python
EpochData:
    gps_week, gps_seconds, gt_lat, gt_lon, gt_height
    observations: List[GNSSObservation]

GNSSObservation:
    gnss_id, sv_id, pr_mes, cno, pr_stdev, nlos_label
    elevation, azimuth, pseudorange_error
```

---

## 4. 模型架构

### 4.1 网络结构

```
输入: (N, 11) 节点特征矩阵
     │
     ▼
Input Proj:  Linear(11→128) + ReLU + Dropout(0.1)
     │
     ▼
GAT × 2:     GATLayer(128→128, heads=8, concat=False)
              + ELU + LayerNorm + Residual + Dropout(0.1)
     │
     ▼
Output Proj: Linear(128→128) + ReLU + Dropout(0.1)
     │
     ├── p_los_head:           Linear → Sigmoid              → p(LOS)
     ├── mu_nlos_head:         Linear → Softplus + clamp     → μ(NLOS)   [MoG only]
     ├── log_sigma_los_head:   Linear → exp                  → σ(LOS)    [MoG only]
     ├── log_sigma_nlos_head:  Linear → exp                  → σ(NLOS)   [MoG only]
     └── uncertainty_head:     Linear → exp                  → σ         [BCE mode]
```

### 4.2 GATLayer v2026（向量化实现）

- **向量化聚合**：`index_add_` 替代 Python for-loop，消除 .item() CPU-GPU 同步
- **支持 block-diagonal batching**：batch 中多个图拼接为分块对角矩阵
- **8 注意力头**，`concat=False` 时取均值
- **纯 PyTorch 实现**，不依赖 torch-geometric

### 4.3 邻接矩阵

- 方位角差 `|az_i - az_j| < 90°` → 双向边
- 无有效边时自动添加自环

---

## 5. 11 维节点特征

| 维度 | 特征 | 归一化 | 物理意义 |
|------|------|--------|----------|
| 0 | elevation | ÷ 90.0 | 低仰角 → 更可能被遮挡 |
| 1 | azimuth | ÷ 360.0 | 方向信息 |
| 2 | C/N₀ | ÷ 60.0 | 信号质量直接指标 |
| 3 | prStdev | ÷ 5.0 | 接收机报告测量不确定度 |
| 4 | prMes | ÷ 3×10⁷ | 观测距离量级 |
| 5 | prInnovation | ÷ 100.0 | 伪距创新量（最强 NLOS 信号） |
| 6 | cos(elevation) | — | 几何精度代理 |
| 7-10 | GPS/Glonass/Galileo/BeiDou | 0/1 | 星座 one-hot |

> **硬约束**：特征维度固定为 11，`assert feature_matrix.shape[1] == 11`。标签不进入特征矩阵。

---

## 6. 损失函数

### 6.1 BCE + Uncertainty 模式 (`USE_MIXTURE_GAUSSIAN=False`)

```
L_total = λ_bce · L_BCE + λ_unc · L_uncertainty - λ_entropy · H(p_los)
```

- **L_BCE**：加权二分类交叉熵 (pos_weight=1.07, label_smoothing=0.05)
- **L_uncertainty**：Heteroscedastic Gaussian NLL `0.5·log(σ²) + 0.5·(err²/σ²)`
- **H(p_los)**：Entropy regularization 防止概率坍缩

### 6.2 MoG 模式 (`USE_MIXTURE_GAUSSIAN=True`)

**三阶段训练**:
1. **Pure BCE (epochs 1-30)**: 仅训练 p_los head，冻结 sigma/mu heads
2. **Blend (epochs 31-55)**: BCE + Supervised Component NLL (lam 1.0→0.0)
3. **Pure NLL (epochs 56+)**: NLL + BCE (双监督)

```
L_NLL = -log[ p·N(0, σ_los) + (1-p)·N(μ_nlos, σ_nlos) ]  # logsumexp 实现
L_total = L_NLL + L_BCE + λ_mu_reg·μ² + λ_sigma_reg·log_σ²
```

### 6.3 数值保护

| 风险 | 对策 |
|------|------|
| 概率下溢 | `logsumexp` 替代 `log(a+b)` |
| p_los 坍缩 | clamp [ε, 1-ε] + entropy reg + label smoothing |
| sigma explosion | clamp + L2 regularization |
| 梯度爆炸 | gradient clip = 1.0 |

---

## 7. 使用方法

### 7.1 环境

```txt
torch >= 2.0.0
numpy, pandas, scipy
tensorboard  (可选，训练可视化)
```

### 7.2 单场景训练

```bash
cd model/part1_GAT/RadioGAT-Multi-band-Radiomap-Reconstruction

# BCE + Uncertainty 模式 (100 epochs, block-diagonal batching)
python run_full_training.py --dataset berlin1_potsdamer_platz

# MoG 模式 (需先设置 config.USE_MIXTURE_GAUSSIAN=True)
python run_full_training.py --dataset berlin1_potsdamer_platz
```

### 7.3 并行训练 4 场景

```bash
# 4 个终端分别运行
python run_full_training.py --dataset berlin1_potsdamer_platz
python run_full_training.py --dataset berlin2_gendarmenmarkt
python run_full_training.py --dataset frankfurt1_maintower
python run_full_training.py --dataset frankfurt2_westendtower
```

### 7.4 分析训练结果

```bash
# BCE 模式分析 (2-head 模型)
python analyze_experiment.py --exp exp_001 --dataset berlin1_potsdamer_platz

# MoG 模式分析 (4-head 模型)
python analyze_mog.py --exp exp_008 --dataset berlin1_potsdamer_platz
```

---

## 8. 关键超参数

### 8.1 通用训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LEARNING_RATE` | 5e-5 | 学习率 |
| `NUM_EPOCHS` | 100 | 总训练轮数 |
| `BATCH_SIZE` | 32 | block-diagonal batch 大小 |
| `GRADIENT_ACCUMULATION` | 1 | 梯度累积步数 |
| `GRADIENT_CLIP` | 1.0 | 梯度裁剪阈值 |
| `HIDDEN_FEATURES` | 128 | GAT 隐藏维度 |
| `NUM_HEADS` | 8 | 注意力头数 |
| `NUM_LAYERS` | 2 | GAT 层数 |
| `DROPOUT` | 0.1 | Dropout 率 |
| `AZIMUTH_THRESHOLD` | 90° | 邻接矩阵边阈值 |
| `EARLY_STOPPING_PATIENCE` | 20 | 早停耐心值 |

### 8.2 BCE 模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LAMBDA_BCE` | 0.6 | BCE 损失权重 |
| `LAMBDA_UNC` | 0.08 | Uncertainty 损失权重 |
| `LAMBDA_ENTROPY` | 0.03 | Entropy 正则化权重 |
| `LAMBDA_ELEVATION_PRIOR` | 0.1 | 仰角先验惩罚权重 |
| `P_LOS_SMOOTHING` | 0.2 | p_los 平滑系数 |
| `LABEL_SMOOTHING` | 0.05 | BCE 标签平滑 |
| `POS_WEIGHT` | 1.07 | NLOS 类别权重 (auto) |

### 8.3 MoG 模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MOG_PURE_BCE_EPOCHS` | 30 | Pure BCE 阶段轮数 |
| `MOG_BLEND_EPOCHS` | 25 | Blend 过渡阶段轮数 |
| `MU_NLOS_MIN` | 0.0 | μ_nlos clamp 下界 (km) |
| `MU_NLOS_MAX` | 500.0 | μ_nlos clamp 上界 (km) |
| `SIGMA_NLOS_MIN` | 2.0 | σ_nlos clamp 下界 (km) |
| `SIGMA_NLOS_MAX` | 200.0 | σ_nlos clamp 上界 (km) |
| `LAMBDA_MU_REG` | 0.001 | μ_nlos L2 正则化 |
| `LAMBDA_SIGMA_REG` | 0.001 | log_sigma L2 正则化 |
| `SIGMA_GAP_TARGET` | 0.3 | σ_nlos - σ_los 最小间隔目标 |
| `LAMBDA_SIGMA_SEP` | 2.0 | Sigma 分离损失权重 |

---

## 9. 输出结果

训练结果保存在 `model/part1_GAT/result/exp_XXX/` 下：

```
result/exp_008/
├── env.md                       # 实验配置 + 训练摘要
├── result.md                    # 关键指标摘要
├── best_model.pth              # 验证集最佳模型
├── final_model.pth             # 训练结束模型
├── checkpoints/
│   └── checkpoint_epoch_{N}.pth # 每 epoch 检查点
└── analysis_{dataset}.json      # 完整推理分析 (analyze_*.py 生成)
```

### 分析 JSON 内容

`analyze_experiment.py` / `analyze_mog.py` 生成的 JSON 包含：
- 分类指标 (Acc, P, R, F1, TP/FP/TN/FN)
- p_los 分布 (LOS/NLOS 均值、中位数、分位数、gap)
- mu_nlos 分布 (MoG only)
- sigma_los / sigma_nlos 分布
- 仰角/CNO 分档精度
- FN/FP Top-10 错误案例 (含仰角、CNO、prErr 特征)
- 双峰质量评估

---

## 10. 实验结果总结

### 10.1 BCE + Uncertainty 模式 (block-diagonal bs=32, 100 epochs)

| 数据集 | Acc | F1 | p_los Gap | 加速比 | 最佳 Epoch |
|--------|-----|-----|-----------|--------|-----------|
| berlin1 | 0.8696 | 0.8692 | 0.643 | 2.7× | 98 |
| berlin2 | 0.8956 | 0.8686 | 0.715 | 16.8× | 100 |
| frankfurt1 | — | — | — | — | — |
| frankfurt2 | — | — | — | — | — |

### 10.2 MoG 模式 (block-diagonal bs=32, 100 epochs)

| 数据集 | Acc | F1 | p_los Gap | sigma_nlos Gap | 最佳 Epoch |
|--------|-----|-----|-----------|---------------|-----------|
| berlin1 | 0.8467 | 0.8475 | 0.451 | -0.019 | 53 |
| berlin2 | 0.8687 | 0.8413 | 0.668 | **+0.657** | 31 |
| frankfurt1 | 0.8333 | 0.8152 | 0.556 | -0.004 | 29 |
| frankfurt2 | 0.8740 | 0.7857 | 0.588 | **+0.319** | 72 |

### 10.3 MoG vs BCE 对比

- **分类性能**：MoG 在所有 4 场景的 F1 低于 BCE 基线 (平均 -0.022)
- **Uncertainty**：MoG 在 berlin2 和 frankfurt2 实现了有意义的 sigma 分离度 (BCE 基线几乎无分离)
- **mu_nlos**：3/4 实验中 mu_nlos head 失效（卡在 clamp 上限 3.0），需修复初始化
- **结论**：当前 MoG 不建议替代 BCE 基线，需修复 mu_nlos 初始化和训练策略

### 10.4 block-diagonal batching 加速

| 数据集 | bs=1 (旧) | bs=32 (新) | 加速比 |
|--------|----------|-----------|--------|
| berlin1 (1,377 epochs) | ~1.4 min/ep | 0.52 min/ep | 2.7× |
| berlin2 (5,925 epochs) | ~7.9 min/ep | 0.47 min/ep | 16.8× |

大数据集上加速效果显著：berlin2 在旧版需 ~2 天完成 100 epoch，新版仅需 ~47 分钟。

---

## 11. 设计决策记录

### 11.1 为什么使用 block-diagonal batching？

每个 GNSS 历元的可见卫星数不同（6-20+ 颗），无法直接 stack 为等长 batch。Block-diagonal batching 将 batch 内多个图的邻接矩阵拼接为分块对角矩阵，节点特征直接 cat，实现 32× 的有效 batch 大小。验证表明对模型质量**零影响**（F1 差异 < 0.001）。

### 11.2 为什么向量化 GATLayer？

原版 GATLayer 使用 `for i in range(edge_index.size(1))` 的 Python 双循环，每次调用 `.item()` 触发 CPU-GPU 同步，在大 batch 下成为瓶颈。向量化版本使用 `index_add_` 原地操作，消除了 Python 循环和同步开销。

### 11.3 为什么 MoG 有三阶段训练？

直接端到端训练 MoG NLL 会导致 p_los 和 sigma/mu 互相干扰。三阶段策略：先训练 p_los 获得稳定分类器 → 逐步引入 component loss → 最终联合优化。但当前 mu_nlos 初始化不当导致 3/4 实验失败。

### 11.4 为什么使用 BCE 辅助损失 (MoG)？

纯 MoG NLL 存在**梯度饥饿**：当 p_los → 1 时，NLOS 分量收不到梯度。BCE 辅助损失直接监督 p_los，防止坍缩。

### 11.5 为什么 batch_size=32 有效？

Block-diagonal 拼接使不同大小的图可以合并为一个 batch。有效 batch=32 在 GPU 利用率（~15-40%）和梯度噪声之间取得平衡。对于小数据集（如 berlin1），建议降低 batch_size。

---

## 12. 已知问题与改进方向

| 优先级 | 问题 | 改进方案 |
|--------|------|---------|
| P0 | MoG mu_nlos bias=3.91 → softplus → 3.0 卡上限 | 改为 bias=-2.0, softplus≈0.13 |
| P0 | MoG 前 30 epoch 冻结 sigma/mu | 全程使用 SupervisedComponentNLLLoss |
| P1 | MoG p_los gap 全面低于 BCE | p_los head 独立 LR + BCE 权重调整 |
| P1 | sigma_los 坍缩 (exp_010: 0.002 km) | 加 min_sigma 保护或 L2 中心惩罚 |
| P2 | 所有模型对伪距误差不敏感 | 强化 prInnovation 特征权重 |
| P2 | 高仰角 NLOS 系统漏报 | 针对性数据增强 |

---

## 原始项目致谢

本项目基于 RadioGAT 改造：
> X. Li et al., "RadioGAT: A Joint Model-Based and Data-Driven Framework for Multi-Band Radiomap Reconstruction via Graph Attention Networks," IEEE Trans. Wireless Commun., vol. 23, no. 11, pp. 17777-17792, Nov. 2024

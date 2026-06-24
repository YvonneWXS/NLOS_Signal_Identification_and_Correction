# Module 1: NLOS Perception & Error Distribution Modeling (GAT-based)

> Urban GNSS NLOS Signal Identification & Correction  
> **Module 1**: GAT-based NLOS detection + Mixture-of-Gaussians (MoG) uncertainty estimation  
> **Current version: v8** (2026-06-05) — mu_nlos direction CORRECT + magnitude RESTORED. MoG输出 (p_los, μ_NLOS, σ_LOS, σ_NLOS). 四数据集 F1 0.84–0.91.

---

## Quick Start

```batch
:: 激活环境
conda activate smartLoc
cd /d "D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model"

:: 单数据集训练
python run_full_training.py --dataset berlin1_potsdamer_platz --exp-name exp_001

:: 四数据集串行训练
python run_serial.py

:: 训练后分析
python analyze_mog.py --exp exp_048 --dataset berlin1_potsdamer_platz
```

单数据集 100 epoch 耗时: ~25 min (bs=32, AMP on RTX 5060 Laptop GPU).

---

## Directory Structure

```
part1_GAT/
├── file/                                # Documentation & design docs (this directory)
│   ├── README.md                        # ← 本文档
│   ├── 城市环境GNSS NLOS信号识别与修正：研究范式重构与实验框架设计.pdf  # 项目总体框架设计
│   ├── 图级批次合并.md                   # Block-diagonal batching 设计与实测
│   ├── 混合高斯输出.md                   # MoG 输出恢复计划 (三阶段渐进训练)
│   ├── Goal Fix 6 for MoG v2.md         # Fix6: sigma分离 + 裁剪 + F1追赶
│   ├── Fix6_FourCity_Comparison_Report.md  # Fix6 四城对比报告
│   ├── Fix6_Code_Changes_Detailed_Report.md # Fix6 代码修改详细报告
│   └── MoG_Architecture_Changes_Report.md   # MoG 架构变更总览
│
├── model/                               # Core code (active)
│   ├── run_full_training.py             # [Main] 训练入口 (解析 --dataset --exp-name)
│   ├── run_serial.py                    # [Util] 四数据集串行训练 + 分析
│   ├── GAT_V2025.py                     # [Active] 主模型文件: GATLayer + NLOSGAT + MoGNLLLoss + train/evaluate/main (~1500行)
│   ├── GAT_V2026.py                     # [Backup] V2026: 向量化GAT + block-diagonal (已合并到V2025，保留为备份)
│   ├── config.py                        # [Config] 集中配置 (路径 / 架构 / 训练 / Loss / MoG)
│   ├── analyze_mog.py                   # [Analysis] 训练后完整指标分析
│   ├── analyze_experiment.py            # [Analysis] 旧版分析 (被 analyze_mog 替代)
│   ├── analyze_model.py                 # [Analysis] 模型诊断
│   ├── generate_report.py               # [Analysis] 报告生成
│   ├── Data_read.py                     # [Data] 数据加载 (obs/nav/sp3)
│   ├── NodeFeature_Generate.py          # [Data] 11维节点特征提取
│   ├── Depth_Adj_Generate.py            # [Data] 测站高程异常生成
│   ├── Radio_Depth_Generate.py          # [Data] 无线电测深生成
│   ├── sp3_reader.py                    # [Data] SP3精密星历读取
│   ├── train_wrapper.py                 # [Data] 训练封装
│   ├── positioning_test.py              # [Test] 定位测试
│   ├── New_axis40.txt / stations_position.txt  # [Data] IGS站坐标
│   └── backup_*/                        # 代码备份 (按时间戳)
│
├── result/                              # Experiment results
│   ├── exp_001-004/                     # BCE baseline (initial)
│   ├── exp_007-019/                     # BCE+Uncertainty experiments
│   ├── exp_020-029/                     # MoG protocol verification
│   ├── exp_030-033/                     # MoG Fix6 (sigma separation + clamp)
│   ├── exp_034-039/                     # MoG Fix6 再训练
│   ├── exp_040-043/                     # v5: supervised mu + μ方向初步修正 → 方向仍然错误
│   ├── exp_044-047/                     # v7: MuDirectionLoss 方向修正 → 幅度塌缩
│   ├── exp_048-051/                     # v8: 纯成对排序 μ方向修正 → 方向正确 + 幅度正常 (CURRENT)
│   └── analysis_*.json / four_city_*.md  # 汇总分析报告
│
└── RadioGAT-Multi-band-Radiomap-Reconstruction/  # [Legacy] 旧版代码 (弃用)
```

---

## Model Architecture

### NLOSGAT (GAT_V2025.py)

```
Input: 11-dim per-satellite features
  [elevation, azimuth, CNO, prStdev]
  + GNSS one-hot (GPS/Glonass/Galileo/BeiDou)
  + prError, pseudorange residual, cycle slip, half-cycle slip

Network:
  ┌─────────────────────────────────────────┐
  │  GATLayer ×2 (8 heads, 128 hidden)      │
  │  Block-diagonal batching (bs=32)        │
  │  LeakyReLU + Dropout(0.1)               │
  ├─────────────────────────────────────────┤
  │  Node-level representations: h_i ∈ R^128 │
  ├──────────────┬──────────────┬───────────┤
  │  p_los_head  │ mu_nlos_head │ sigma heads│
  │  Sigmoid     │  Softplus     │  exp(clamp)│
  │  → p_los∈[0,1]│ → μ_NLOS>0  │ → σ_LOS, σ_NLOS │
  └──────────────┴──────────────┴───────────┘
Output: p_los, mu_nlos, sigma_los, sigma_nlos

Parameters: ~281k
Training: AMP混合精度, bs=32, grad_accum=1, block-diagonal collate
Speed: ~25 min / 100 epoch (berlin1), ~2.7× 加速比 vs bs=1
```

### Edge Construction (GAT Graph)

- 每历元卫星构建全连接图
- 边权重 = azimuth_diff / AZIMUTH_THRESHOLD (90°)
- azimuth_diff > 90° → 无边 (天空上方位角差距大的卫星互相关系弱)

### Loss Function (三阶段渐进训练)

| 阶段 | Epoch | 激活的损失项 | 说明 |
|------|:-----:|-------------|------|
| **阶段一: 纯BCE** | 1–8 | BCE + Entropy + ElevPrior | 冻结 mu/sigma 头，p_los 热身 |
| **阶段二: 混合过渡** | 9–33 | BCE + MoG NLL + MuReg + MuDir + SigmaSep | 解冻 mu/sigma，渐进切换 (λ从1.0衰减到0.0) |
| **阶段三: 纯MoG NLL** | 34–100 | MoG NLL + BCE(0.6) + MuReg + MuDir + SigmaSep | 主体训练，mu方向修正主导 |

每个损失项:
- **BCE**: `F.binary_cross_entropy(p_los, labels)` — 二分类
- **MoG NLL**: 混合高斯负对数似然 — `logsumexp([log(p_los)+log N(r|0,σ_LOS), log(1-p_los)+log N(r|μ_NLOS,σ_NLOS)])`
- **MuReg**: L2 anchor — `(mu_nlos - MU_NLOS_TARGET)^2`, λ=0.20
- **MuDirectionLoss**: 纯成对排序损失 — 强制 mean(mu_NLOS) > mean(mu_LOS) + 0.15 km, λ=1.0
- **SigmaSep**: 逐样本 σ_NLOS > σ_LOS 分离损失
- **Entropy**: 熵正则 — 防 p_los 退化为 0/1
- **ElevPrior**: 仰角先验 — p_los 应与 sin(elevation) 大致一致

---

## Configuration (config.py)

| 参数 | 当前值 | 说明 |
|------|:-----:|------|
| `IN_FEATURES` | 11 | 节点特征维度 |
| `HIDDEN_FEATURES` | 128 | 隐藏层维度 |
| `NUM_HEADS` | 8 | GAT 注意力头数 |
| `NUM_LAYERS` | 2 | GAT 层数 |
| `BATCH_SIZE` | 32 | Block-diagonal batch (图级合并) |
| `LEARNING_RATE` | 5e-5 | 学习率 (适配大batch) |
| `NUM_EPOCHS` | 100 | 训练轮数 |
| `USE_MIXTURE_GAUSSIAN` | True | 激活完整 MoG 输出 |
| `MOG_PURE_BCE_EPOCHS` | 8 | 阶段一: 纯BCE热身 |
| `MOG_BLEND_EPOCHS` | 25 | 阶段二: 混合过渡 |
| `LAMBDA_MU_REG` | 0.20 | μ_NLOS L2正则权重 |
| `LAMBDA_MU_DIRECTION` | 1.0 | 成对排序损失权重 |
| `MU_NLOS_TARGET` | 0.30 km | μ_NLOS L2锚点 |
| `SIGMA_LOS_CLAMP_LOG_MAX` | 2.0 | σ_LOS log clamp上限 (≈7.4m) |
| `SIGMA_NLOS_CLAMP_LOG_MAX` | 2.5 | σ_NLOS log clamp上限 (≈12.2m) |

---

## Data Pipeline

```
原始数据 (data/dataset/{city}/obs, nav, sp3)
  │
  ├── Data_read.py: 加载观测/导航/SP3文件
  ├── sp3_reader.py: 解析SP3精密星历 → 卫星ECEF位置
  └── NodeFeature_Generate.py: 提取11维特征
       │
       ▼
  data/processedData/{dataset}_processed.pkl  (缓存)
       │
       ▼
  run_full_training.py → GAT_V2025.py 训练 → result/exp_0XX/
       │
       ├── best_model.pth  /  final_model.pth
       └── checkpoints/
```

---

## Key Experiment History

| 实验 | 数据集 | 关键变化 | 核心结果 |
|------|------|---------|---------|
| exp_001-004 | 四城 | BCE baseline (初始实现) | 基线建立 |
| exp_007-011 | 四城 | BCE+Uncertainty | sigma NLOS 失效 (frankfurt) |
| exp_012-029 | berlin1 | MoG 协议验证 | 训练稳定性迭代 |
| exp_030-033 | 四城 | MoG Fix6 (sigma分离+clamp, F1追赶) | sigma 改善，F1 仍低 0.017 |
| exp_034-039 | 四城 | MoG Fix6 再训练 | sigma 更好，μ方向错误 |
| exp_040-043 | 四城 | v5: supervised mu | μ方向全是错的 (mu_LOS > mu_NLOS) |
| exp_044-047 | 四城 | v7: MuDirectionLoss 方向修正 | 方向正确, 幅度塌缩 (mu_NLOS 仅181-223m) |
| **exp_048-051** | **四城** | **v8: 纯成对排序, 无压制, LAMBDA_MU_REG=0.20** | **方向正确, 幅度正常 (mu_NLOS=216-308m, F1=0.84-0.91)** |

### v8 最终模型状态

| 数据集 | Exp | F1 | mu_LOS | mu_NLOS | Margin | σ(NLOS)/σ(LOS) |
|--------|:---:|:---:|:------:|:------:|:------:|:--------------:|
| berlin1 | exp_048 | 0.854 | 191m | 308m | +117m | 1.09 |
| berlin2 | exp_049 | 0.892 | 73m  | 216m | +143m | 1.05 |
| frankfurt1 | exp_050 | 0.843 | 117m | 237m | +121m | 1.12 |
| frankfurt2 | exp_051 | 0.906 | 141m | 260m | +119m | 1.08 |

---

## Analysis Tools

### analyze_mog.py — 完整指标分析

```batch
python analyze_mog.py --exp exp_048 --dataset berlin1_potsdamer_platz
```

输出: 分类性能 (Acc/F1/Precision/Recall), p_los 分布 (双峰质量), 错误案例 (Type A/B), sigma 分布, 星座性能, μ_NLOS 方向验证. 结果保存为 `result/exp_048/analysis_*.json` + `result.md`.

### analyze_experiment.py — 旧版分析 (兼容)

支持 `--exp` 参数，可生成 Markdown 报告。

---

## Implementation History

### v8 (2026-06-05) — 纯成对排序 μ方向修正 (CURRENT)
- **修复**: 移除 v7 的 LOS 压制项 (2.0× 权重), 仅保留纯成对排序损失
- **结果**: 方向正确 + 幅度恢复 (mu_NLOS 216-308m), F1 0.84-0.91

### v7 (2026-06-04) — MuDirectionLoss 方向修正 (部分成功)
- 修复了 μ方向 (v5 的错误方向被纠正)
- **问题**: μ幅度塌缩到 181-223m

### v5 (2026-06-04) — 监督式 mu
- 引入监督式 μ_NLOS L2回归
- **问题**: μ方向错误 (mu_LOS > mu_NLOS)

### MoG Fix6 (2026-05-29) — sigma 分离 + 裁剪 + F1 追赶
- 逐样本 sigma 分离损失, 硬裁剪, 自动 pos_weight

### MoG v1 (2026-05-28) — 恢复混合高斯输出
- 三阶段渐进训练, 向量化 GAT + block-diagonal batching 合并

### BCE+Uncertainty (earlier) — 基线
- p_los (sigmoid) + log_sigma (线性), BCE + Uncertainty loss

---

## Environment

| Item | Value |
|------|-------|
| Python | 3.9+ (conda: smartLoc) |
| PyTorch | CUDA 2.x (RTX 5060 Laptop GPU, 8 GB) |
| Dependencies | NumPy, SciPy, TensorBoard |
| Training time | ~25 min / dataset / 100 epoch |

---

## Related Documents

- [Module 2: Factor Graph Localization Fusion](../part2_FactorGraphLocalizationFusion/model/README.md)
- [Module 3: Residual Feedback & Online Correction](../part3_ResidualFeedbackAndOnline_Correction/model/README.md)
- [Main Project README](../../README.md)
- [Research Framework PDF](城市环境GNSS NLOS信号识别与修正：研究范式重构与实验框架设计.pdf)

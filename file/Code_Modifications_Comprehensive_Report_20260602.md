# NLOS Signal Identification and Correction — 代码修改综合报告

**日期**: 2026-06-02
**项目**: 城市环境 GNSS NLOS 信号识别与修正
**GitHub**: https://github.com/YvonneWXS/NLOS_Signal_Identification_and_Correction
**分支**: master (领先 origin/master 2 commits, 含未提交 v3 修改)

---

## 一、项目架构总览

本项目分为两大模块：

| 模块 | 功能 | 核心技术 | 状态 |
|------|------|---------|:----:|
| **Module 1** (part1_GAT) | NLOS 感知与误差分布建模 | GAT + 混合高斯输出 (MoG) | **已完成** (Fix 6) |
| **Module 2** (part2_FactorGraphLocalizationFusion) | 因子图定位融合 | FactorGraph + MoG 观测模型 + L-BFGS-B | **v2 完成, v3 进行中** |

数据流: 原始 GNSS 观测 → Module 1 (GAT 推断 p_los, mu_nlos, sigma) → Module 2 (因子图 + 6 方法定位评估)

---

## 二、Module 1: GAT NLOS 感知模型 — 修改历程

### 2.1 基线 (BCE+Uncertainty)

**初始架构**:
- 模型: 2层 8头 GAT, 128 hidden dim, 281K 参数
- 输出: p_los (Sigmoid, LOS概率) + log_sigma (整体不确定性)
- 损失: BCE + Uncertainty Loss
- 问题: frankfurt1/2 场景 sigma(NLOS) <= sigma(LOS), 不确定性估计失效

### 2.2 图级批次合并 (Block-Diagonal Batching) — 2.7x 加速

**修改文件**: `GAT_V2025.py` 的 `batch_collate_fn` + `GATLayer.forward`

**核心改动**:
1. **自定义 collate_fn**: 将多个历元的 edge_index 拼接为 block-diagonal (块对角) 形式, 正确偏移节点索引
2. **GATLayer 向量化**: 将 Python for-loop 实现替换为 `index_add_` 操作
3. **batch_size**: 1 → 32
4. **学习率**: 1e-4 → 5e-5 (配合大 batch)

**效果**:
- 每 epoch: 1.4 min → 0.52 min (2.7× 加速)
- 4 城 100 epoch: ~48h → ~3.5h
- **模型质量零损失** (F1 差异 < 0.001)

### 2.3 AMP 混合精度训练

**修改文件**: `GAT_V2025.py`, `config.py`

**核心改动**:
1. 添加 `torch.cuda.amp.autocast` + `GradScaler`
2. 修复 dtype 不匹配 (Half vs Float) 在 BCE loss 中
3. `GRADIENT_ACCUMULATION`: 4 → 8
4. 关闭逐 batch 打印, 仅输出 epoch 级别指标

### 2.4 恢复混合高斯输出 (MoG) — 核心架构升级

**修改文件**: `GAT_V2025.py`, `config.py`, `analyze_experiment.py`

**架构变更**:

| 组件 | BCE 基线 | MoG 最终版 |
|------|----------|-----------|
| 输出头 | 2 个 (p_los, log_sigma) | **4 个** (p_los, mu_nlos, log_sigma_los, log_sigma_nlos) |
| mu_nlos_head | 不存在 | Linear(128, 1) + Softplus (bias=-2.0) |
| log_sigma_los_head | 不存在 | Linear(128, 1) (bias=-2.0) |
| log_sigma_nlos_head | 统一 sigma | 改为 NLOS 专用, bias=-3.0 |

**三阶段训练策略**:

| 阶段 | Epoch | p_los 梯度来源 | sigma/mu 梯度来源 | 损失函数 |
|------|:---:|------|------|------|
| 阶段 1: 纯 BCE | 1-8 | BCE | mu_reg(L2) + sigma_warmup_reg | NLOSLoss |
| 阶段 2: Blend | 9-33 | BCE | SupervisedComponentNLLLoss (cosine 0→1) | lam*BCE + (1-lam)*CompNLL |
| 阶段 3: 纯 NLL | 34-100 | BCE (10:1 vs NLL) | MoGNLLLoss (p_los.detach()) | NLL*0.1 + BCE*1.0 |

**关键设计决策**:
- p_los 在所有阶段仅通过 BCE 训练, 不被 NLL 干扰
- NLL 阶段使用 `p_los.detach()` 切断梯度
- BCE:NLL = 10:1, 确保分类信号主导 backbone 梯度

**损失函数体系**:

1. **MoGNLLLoss** (完整混合高斯负对数似然):
   `
   log_mix = logsumexp([log(p_los) + log N(err|0, sigma_los),
                        log(1-p_los) + log N(err|mu_nlos, sigma_nlos)])
   `
   包含正则化: mu_reg (目标中心化 L2), sigma_center (拉向物理范围), sigma_sep (逐样本分离), entropy (防 p_los 坍缩)

2. **SupervisedComponentNLLLoss**: 使用真实标签分别拟合 LOS/NLOS 成分

**Optimizer 参数组**:

| 参数组 | 学习率倍数 | 说明 |
|--------|:---:|------|
| p_los_head | 6× (3e-4) | 最高 LR, 加速分类收敛 |
| mu_nlos_head | 1× (5e-5) | 标准 LR |
| log_sigma_los/nlos_head | 1× (5e-5) | 标准 LR |
| GAT backbone | 1× (5e-5) | 标准 LR |

### 2.5 MoG 稳定性修复链条 (Fix 1 → Fix 6)

**Fix 1**: mu_nlos head 梯度流修复 — 移除 freeze, 加强 mu_reg (0.001→0.03)
**Fix 2**: 零中心化 → 目标中心化 mu_reg — 将锚定目标从 0 改为 0.15 km
**Fix 3+4**: 正则化加强 — LAMBDA_MU_REG 0.10→0.30, sigma_center 加倍, BCE 3:1 权重
**Fix 5**: p_los LR 4×→10× (后调整为 6×), BCE:NLL 3:1→10:1
**Fix 6** (最终版):
- **6A** (逐样本 sigma 分离): 从 batch-statistic gap → per-sample contrastive loss
- **6B** (log-space 硬裁剪): 输出空间 clamp → log-space clamp, sigma head 梯度裁剪 0.5
- **6C** (动态 BCE 权重): BCE 权重从 1.5→0.6 线性衰减
- **6D** (自动 pos_weight): 根据 NLOS 比例自动计算 BCE pos_weight

**MoG 最终效果**: berlin1 F1=0.857, p_los gap 从 0.52 提升到 0.57

### 2.6 代码组织清理

- 删除 `RadioGAT-Multi-band-Radiomap-Reconstruction/` 子目录
- 核心文件平移到 `part1_GAT/model/`
- 删除 54+ 临时文件 (.bat, .bak, .ascii, _*.py)
- 保留备份: `backup_20260528/` 和 `backup_20260528_223115/`

---

## 三、Module 2: 因子图定位融合 — 修改历程

### 3.1 v1: 初始实现 (已提交: 7ca845b)

**新增文件** (`part2_FactorGraphLocalizationFusion/model/fusion/`):

| 文件 | 功能 |
|------|------|
| `baselines.py` | 6 种定位方法 (Standard LS, WLS-elev, WLS-CNO, WLS-MoG, WLS-MoG-lite, FactorGraph-MoG) |
| `factor_graph_fusion.py` | MoG 观测模型 + FactorGraph L-BFGS-B 优化 |
| `utils.py` | 坐标变换, 数据加载, MoG 推理, 特征提取, SP3 卫星位置计算 |
| `evaluate_fusion.py` | 6 方法评估框架, CEP50 指标 |
| `run_fusion.py` | 主运行脚本 |
| `debug_geometry.py` | 几何验证工具 |
| `motion_geometry_predictor.py` | 运动几何预测器 (为 2A prior 准备) |

**v1 关键 Bug 修复**:
1. **Jacobian 符号修复** (CRITICAL): H[:,:3] = -LOS (之前错误地使用 +LOS, 导致 LS 发散到 14.7km)
2. **SP3 时钟改正移除**: 移除后 RMS 从 88513m 降到 432m
3. **MoG 推理特征提取修复**: 补齐缺失的 3 维特征 (pr_mes/3e7, pseudorange_error, GNSS one-hot)
4. **Sigma 裁剪增强**: [0.01, inf) → [0.05, 50.0]

### 3.2 v2: 稳定性全面翻修 (已提交: 9c2c905)

**核心改进**:

| 改进 | 方法 | 效果 |
|------|------|------|
| **Fix A**: 鲁棒 MoG NLL | sigma 裁剪 [0.1, 10.0], p_los 裁剪 [0.02, 0.98], 逐分量裁剪 [-30, +10] | 消除数值溢出 |
| **Fix B**: Huber 化 NLL | 极值残差梯度缩放至 30%, smooth Huber 替代 hard clip | 梯度连续, 收敛稳定 |
| **Fix C**: 多起点优化 | 3 起点 (WLS-MoG, WLS-elevation, Standard LS) + 最佳选择 | berlin 场景改善 |
| **Fix D**: 梯度验证 | approx_fprime vs 解析梯度, 逐历元诊断 | 梯度方向正确, 量级偏差 2-44× |

**v2 关键 Bug 修复**:
- FactorGraph batch_solve 参数修复
- 状态向量边界约束 (Earth 表面 ±9556km)
- 回退策略: 爆炸时回退 WLS-MoG

**v2 结果** (exp_001):

| 方法 | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|:---:|:---:|:---:|:---:|
| Standard LS | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-MoG | 983.2 | 830.6 | 551.6 | 508.6 |
| FactorGraph-MoG | **937.0** | **793.8** | 551.6 | 508.6 |
| Delta FG vs WLS | +3.3% | +4.7% | 0.0% | 0.0% |

- Frankfurt 场景: NLL 曲面太平坦, L-BFGS-B 无法改善
- 0% epochs < 10m: 纯伪距定位在无改正条件下无法满足高精度需求

### 3.3 v3: P0/P1/P2 优化 (未提交, 进行中)

**修改文件**: `factor_graph_fusion.py` (v3), `utils.py` (v4), `evaluate_fusion.py` (v2)

**P0.1 — 平滑梯度**:
- `_smooth_clip(x, lo, hi, k=5.0)`: softplus 平滑替代 np.clip
- 梯度验证最大相对误差: 44× → 4.65×
- Smooth Huber: `smooth_huber = 0.5 * (mx + log(1 + exp(-|a - b|)))`

**P0.2 — p_los 温度缩放**:
- 推理时: `logit → scale(÷T) → sigmoid`, T=0.6
- `run_mog_inference()` 返回 `p_los_sharp` + `sigma_ratio`

**P1.1 — 逐历元诊断**:
- 每个 test epoch 输出: NLL 变化, p_los 均值, sigma LOS/NLOS 均值, 残差分布

**P1.2 — TCN 2A 先验训练** (部分完成):
- 新增 `train_tcn.py` (未提交)
- 使用 Module 1 MoG 输出作为软标签预测 NLOS
- 数据缓存已生成 (berlin1), Conv1d padding 需要修复

**P2 — 优化器比较**:
- L-BFGS-B: 正常工作
- trust-ncg: 在 MoG NLL 上完全失败 (非凸)

**Huber 离群值抑制**:
- Sigmoid-based: `damp = 0.3 + 0.7 / (1 + exp(clip(5*(z-3), -50, 50)))`
- z > 3 时梯度缩放至 30%, z=0 时保持 100%

**v3 最新结果** (exp_001):

| 方法 | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|:---:|:---:|:---:|:---:|
| Standard LS | 904.5 | 610.8 | 525.2 | 382.6 |
| WLS-MoG | 983.2 | 830.6 | 551.6 | 508.6 |
| FactorGraph-MoG | **937.0** | **793.8** | 551.6 | 508.6 |

- 全部 10 个 berlin1 test epoch 的 NLL 在 L-BFGS-B 后改善
- Temp scaling 使 WLS-MoG 在 berlin2 上变差 (716→831m), 需重新调参

---

## 四、训练性能基准

| 配置 | 单 epoch 时间 | 4 城 100 epoch | GPU 利用率 |
|------|:---:|:---:|:---:|
| 原始 (bs=1, 无 AMP) | 1.4 min | ~48 h | ~15% |
| Block-diag (bs=32, 无 AMP) | 0.52 min | ~3.5 h | ~40% |
| Block-diag + AMP + grad_accum=8 | 0.35 min | ~2.3 h | ~55% |

GPU: NVIDIA GeForce RTX 5060 Laptop GPU
环境: Python 3.x, PyTorch CUDA, smartLoc conda env

---

## 五、待完成事项 (优先级排序)

| 优先级 | 任务 | 模块 | 阻塞项 |
|:---:|------|:---:|------|
| **P0** | 修复 TCN Conv1d padding | Module 2 | dilation padding 不匹配 |
| **P1** | 训练 4 数据集 TCN 模型 | Module 2 | P0 完成后 |
| **P1** | 集成 TCN 到 evaluate_fusion.py (FactorGraph-MoG+2A) | Module 2 | TCN 训练完成后 |
| **P1** | 重新运行完整 pipeline + 更新报告 | Module 2 | 以上完成后 |
| **P2** | p_los 温度调参 (T=0.6 在 berlin2 上更差) | Module 2 | 可选 |
| **P2** | 提交 v3 修改到 git + push | 全局 | 代码稳定后 |
| **P3** | 运动模型集成 (motion_geometry_predictor.py) | Module 2 | 后续迭代 |

---

## 六、关键路径速查

`
Python: D:\1_developTool\4_conda\envs\smartLoc\python.exe
GPU: NVIDIA GeForce RTX 5060 Laptop GPU

Module 1 入口:
  model\part1_GAT\model\run_serial.py     (串行训练脚本)
  model\part1_GAT\model\run_full_training.py (单实验训练)
  model\part1_GAT\model\config.py          (训练配置)
  model\part1_GAT\model\GAT_V2025.py       (MoG 模型 + 训练逻辑)
  model\part1_GAT\model\analyze_mog.py     (MoG 分析工具)

Module 2 入口:
  model\part2_FactorGraphLocalizationFusion\model\run_fusion.py
  model\part2_FactorGraphLocalizationFusion\model\fusion\*

结果目录:
  model\part1_GAT\result\exp_001-037\
  model\part2_FactorGraphLocalizationFusion\result\exp_001\

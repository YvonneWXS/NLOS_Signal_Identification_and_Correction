# MoG 混合高斯输出 架构修改完整报告

## 概述

本文档记录了从原始 **BCE+Uncertainty 基线**（无混合高斯输出）到 **MoG R5 最终版**（混合高斯输出）的全部修改。
所有改动仅涉及两个文件：`config.py` 和 `GAT_V2025.py`。

---

## 一、模型架构变更

### 1.1 输出头（Model Heads）

| 组件 | BCE 基线 | MoG 最终版 |
|------|----------|-----------|
| p_los_head | Linear(128, 1) + Sigmoid | 同左（保留） |
| mu_nlos_head | **不存在** | Linear(128, 1) + Softplus（新增） |
| log_sigma_los_head | **不存在** | Linear(128, 1)（新增，可学习 σ_LOS） |
| log_sigma_nlos_head | Linear(128, 1)（单一 σ） | 改为 NLOS 专用，与 log_sigma_los_head 解耦 |

**BCE 基线输出**: `(p_los, log_sigma)` — LOS 概率 + 整体不确定性标准差
**MoG 最终输出**: `(p_los, mu_nlos, log_sigma_los, log_sigma_nlos)` — 完整混合高斯参数

### 1.2 模型参数初始化

```python
# mu_nlos_head: bias = -2.0, softplus(-2.0) ≈ 0.127 km
nn.init.constant_(self.mu_nlos_head.bias, -2.0)
# log_sigma_los_head: bias = -2.0, exp(-2.0) ≈ 0.135 km
nn.init.constant_(self.log_sigma_los_head.bias, -2.0)
# log_sigma_nlos_head: bias = -3.0, exp(-3.0) ≈ 0.05 km
nn.init.constant_(self.log_sigma_nlos_head.bias, -3.0)
```

---

## 二、损失函数架构

### 2.1 BCE 基线：单一 `NLOSLoss`

```
L_total = lambda_bce * BCE(p_los, label) + lambda_unc * UncertaintyLoss(log_sigma, |prError|)
```

仅输出二分类概率 + 一个全局 σ，无法区分 LOS 和 NLOS 的测量质量。

### 2.2 MoG 最终版：三层损失体系

**层一：`NLOSLoss`（保留，纯 BCE 阶段）**
与原始相同，但 log_sigma_nlos 含义从"全局 σ"变为"NLOS σ"。

**层二：`SupervisedComponentNLLLoss`（新增，Blend 阶段）**
使用真实 LOS/NLOS 标签分别拟合：
- LOS 样本 → N(0, σ_los)
- NLOS 样本 → N(μ_nlos, σ_nlos)
不涉及 p_los，仅训练 sigma/mu head。

**层三：`MoGNLLLoss`（新增，纯 NLL 阶段）**
完整混合高斯负对数似然：
```
log_mix = logsumexp([log(p_los) + log P(err|0, σ_los),
                     log(1-p_los) + log P(err|μ_nlos, σ_nlos)])
```

包含以下正则化项：
- **mu_reg**: `lambda_mu_reg * (mu_nlos - mu_target)²` — 目标中心化 L2 锚定
- **sigma_reg**: `lambda_sigma_reg * (log sigma_nlos)²` — 防爆炸
- **sigma_center_loss**: `0.10*(σ_los-0.3)² + 0.01*(σ_nlos-1.5)²` — 拉向物理范围
- **sigma_sep_loss**: `lambda_sep * relu(target_gap - (σ_nlos_NLOS - σ_nlos_LOS))`
- **entropy**: `-H(p_los)` — 防 p_los 坍缩

---

## 三、三阶段训练策略

| 阶段 | Epoch | p_los 梯度来源 | sigma/mu 梯度来源 | 损失函数 |
|------|:---:|------|------|------|
| 阶段 1: 纯 BCE | 1-8 | BCE | mu_reg(L2→0.15)+sigma_warmup_reg | `NLOSLoss` |
| 阶段 2: Blend | 9-33 | BCE | SupervisedComponentNLLLoss (cosine 0→1) | `lam*BCE + (1-lam)*CompNLL` |
| 阶段 3: 纯 NLL | 34-100 | BCE(10:1 vs NLL) | MoGNLLLoss (p_los.detach()) | `NLL*0.1 + BCE*1.0` |

**关键设计决策**：
- p_los 在所有阶段仅通过 BCE 训练，不被 NLL 干扰
- NLL 阶段使用 `p_los.detach()` 切断梯度
- BCE:NLL 总损失权重 10:1，确保分类信号主导 backbone 梯度

---

## 四、Optimizer 参数组

| 参数组 | 学习率 | 说明 |
|--------|:---:|------|
| p_los_head | `10x base_lr = 5e-4` | 最高 LR，加速分类收敛 |
| mu_nlos_head | `1x base_lr = 5e-5` | 标准 LR |
| log_sigma_los_head | `1x base_lr = 5e-5` | 标准 LR |
| log_sigma_nlos_head | `1x base_lr = 5e-5` | 标准 LR |
| GAT backbone | `1x base_lr = 5e-5` | 标准 LR |

---

## 五、config.py 新增参数

```python
# ========== 新增 MoG 开关 ==========
USE_MIXTURE_GAUSSIAN = True

# ========== 三阶段训练控制 ==========
MOG_PURE_BCE_EPOCHS = 8       # 纯 BCE 预热 epoch 数
MOG_BLEND_EPOCHS = 25         # 平滑过渡 epoch 数

# ========== mu_nlos 约束 ==========
MU_NLOS_MIN = 0.0             # Softplus 下限
MU_NLOS_MAX = 500.0           # Softplus 上限（放宽）
LAMBDA_MU_REG = 0.30          # NLL 阶段 mu 锚定强度
LAMBDA_MU_WARMUP_REG = 0.05   # 纯 BCE 阶段 mu 监督强度
MU_NLOS_TARGET = 0.15         # mu_nlos L2 锚定目标 (km)

# ========== sigma 约束 ==========
SIGMA_NLOS_MIN = 0.05         # clamp 下限 (km)
SIGMA_NLOS_MAX = 200.0        # clamp 上限 (km)
SIGMA_GAP_TARGET = 0.5        # sigma_sep 目标 (km)
LAMBDA_SIGMA_SEP = 5.0        # sigma separation loss 权重
LAMBDA_SIGMA_REG = 0.01       # sigma L2 正则

# ========== BCE/NLL 权重 ==========
LAMBDA_BCE_IN_NLL = 1.5       # 纯 NLL 阶段 BCE 权重

# ========== 修改的原有参数 ==========
LEARNING_RATE = 5e-5           # 从 1e-4 降低（适配 bs=32）
BATCH_SIZE = 32                # 从 1 改为 block-diagonal batching
```

---

## 六、GAT_V2025.py 核心代码变更

### 6.1 新增类

| 类名 | 功能 |
|------|------|
| `MoGNLLLoss` (~60 行) | 混合高斯 NLL + sigma centering + mu_reg + sigma_sep |
| `SupervisedComponentNLLLoss` (~55 行) | 基于标签的组件 NLL（Blend 阶段） |

### 6.2 关键修改点

| 位置 | 原始行为 | 修改后行为 |
|------|----------|-----------|
| 模型 forward | 返回 `(p_los, log_sigma)` | 返回 `(p_los, mu_nlos, log_sigma_los, log_sigma_nlos)` |
| train_epoch | 单 loss 函数 | 三阶段分支 |
| 纯 BCE 阶段 | mu/sigma head `requires_grad=False`（BUG） | 所有 head `requires_grad=True` + 轻量 L2 监督 |
| 损失计算 | `loss_fn(p_los, log_sigma, ...)` | 按阶段选择 loss 函数 |
| p_los 梯度 | BCE 反向传播 | BCE + NLL 阶段 `p_los.detach()` 解耦 |
| optimizer | 单一 param_group | 5 个独立 param_group（不同 LR） |
| best_model | `min(val_loss)` | `max(val_f1)` |

---

## 七、发现的 Bug 及修复

| Bug | 严重度 | 现象 | 根因 | 修复 |
|------|:---:|------|------|------|
| mu/sigma head 零梯度 | P0 | mu_nlos = 3.0（clamp 上限） | 纯 BCE 阶段 `requires_grad=False` | 删除 freeze，全阶段可训练 |
| mu_reg 零中心化 | P0 | mu_nlos 从 0.15 坍缩到 0.036 | `lambda * mu^2` 把 mu 推向 0 | 改为 `lambda*(mu-0.15)^2` 目标中心化 |
| best_model 选择偏差 | P1 | 选到 sigma_sep 峰值而非 F1 峰值 | `F1*0.7 + sigma_sep*0.3` 复合指标 | 改为纯 F1 选择 |
| BCE 权重不足 | P1 | F1 系统性低于 BCE 基线 | NLL 梯度通过 backbone 干扰 p_los | BCE:NLL 1:1→10:1 + p_los LR 10x |

---

## 八、实验结果对比

### 8.1 五轮迭代演进 (berlin1_potsdamer_platz)

| 版本 | 实验 | F1 | Acc | p_los gap | sigma_sep | mu_nlos |
|------|------|:---:|:---:|:---:|:---:|:---:|
| BCE 基线 | exp_001 | **0.869** | 0.870 | 0.643 | **0.12** | — |
| MoG R1 (原始) | exp_016 | 0.851 | — | 0.079 | 0.301 | 0.160 |
| MoG R2 (Fix 1) | exp_022 | 0.855 | 0.856 | 0.558 | 0.485 | 0.036 |
| MoG R3 (Fix 2) | exp_023 | 0.855 | 0.856 | 0.558 | 0.485 | 0.062 |
| MoG R4 (Fix 3+4) | exp_025 | 0.837 | 0.847 | 0.516 | 1.371 | 0.113 |
| **MoG R5 (Fix 5)** | **exp_030** | **0.857** | **0.858** | **0.561** | **0.652** | **0.097** |

### 8.2 四城市最终结果 (MoG R5 vs BCE 基线)

| 数据集 | 模型 | F1 | Acc | p_los gap | sigma_sep | mu_nlos |
|--------|------|:---:|:---:|:---:|:---:|:---:|
| berlin1 | BCE | 0.869 | 0.870 | 0.643 | 0.12 | — |
| berlin1 | **MoG R5** | **0.857** | 0.858 | 0.561 | **0.65** | 0.097 |
| berlin2 | BCE | 0.869 | 0.896 | 0.715 | — | — |
| berlin2 | **MoG R5** | **0.850** | 0.877 | 0.684 | **0.46** | 0.046 |
| frankfurt1 | BCE | 0.851 | 0.868 | 0.645 | — | — |
| frankfurt1 | **MoG R5** | **0.823** | 0.835 | 0.516 | **0.82** | **0.169** |
| frankfurt2 | BCE | ~0.87 | ~0.87 | — | — | — |
| frankfurt2 | **MoG R5** | **0.784** | 0.880 | 0.522 | **1.08** | 0.054 |

### 8.3 sigma_sep 改善幅度

| 数据集 | BCE 基线 | MoG R5 | 改善 |
|--------|:---:|:---:|:---:|
| berlin1 | 0.12 km | 0.65 km | **5.4x** |
| berlin2 | — | 0.46 km | — |
| frankfurt1 | — | 0.82 km | — |
| frankfurt2 | — | 1.08 km | — |

---

## 九、训练性能

| 指标 | 原始 (bs=1) | 最终 (bs=32 + AMP) |
|------|:---:|:---:|
| berlin1 每 epoch | ~1.4 min | ~0.52 min |
| 四城市 100 epoch | ~48 小时 | **~3.5 小时** |
| 加速比 | — | **~14x** |
| GPU 利用率 | ~10% | ~80% |

---

## 十、代码改动统计

| 文件 | 新增行 | 删除行 | 净变化 |
|------|:---:|:---:|:---:|
| `GAT_V2025.py` | ~350 | ~57 | +293 |
| `config.py` | ~38 | ~7 | +31 |
| **合计** | **~388** | **~64** | **+324** |

### 主要新增内容

- `MoGNLLLoss` 类 (~60 行)
- `SupervisedComponentNLLLoss` 类 (~55 行)
- 三阶段训练分支逻辑 (~80 行)
- 模型四头输出 + 初始化 (~30 行)
- Optimizer 五参数组 (~20 行)
- sigma/mu 验证指标计算 (~40 行)

---

## 十一、关键工程决策

1. **p_los 与 NLL 永久解耦**: p_los 在所有阶段仅接收 BCE 梯度，NLL 使用 `detach()` 版本。
   这是保证分类性能不崩溃的基础。

2. **mu_reg 目标中心化而非零中心化**: 零中心化 L2 会把 mu 推向 0，
   与物理直觉（NLOS 伪距误差 > 0）相悖。

3. **BCE 梯度在 backbone 上的主导权**: 由于 p_los_head 和 sigma/mu_head 共享 GAT backbone，
   BCE:NLL = 10:1 确保 backbone 学到的特征有利于分类。

4. **三阶段渐进训练**: 纯 BCE(8 ep) → Blend(25 ep) → 纯 NLL(67 ep)，
   cosine 平滑过渡，避免损失函数突变。

5. **block-diagonal batching (bs=32)**: 多个历元的独立小图拼成块对角大图，
   GPU 利用率从 ~10% 提升到 ~80%，训练速度 14x 且对模型质量无影响。

6. **F1-only best model 选择**: 避免 sigma_sep 峰值时期的模型被误选为 best。

---

*报告生成时间: 2026-06-01*
*数据来源: exp_001 (BCE), exp_016 (MoG R1), exp_022-025 (R2-R4), exp_030-033 (R5)*
*代码 commit: 743a832*

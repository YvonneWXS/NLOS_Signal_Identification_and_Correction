# Module 1 最终结果报告 — NLOS 感知与误差分布建模 (GAT + MoG)

**生成时间**: 2026-06-10
**实验**: exp_001--004 (v8 配置复现)
**架构**: NLOSGAT MoG 4-head | Block-diagonal bs=32 | AMP 混合精度
**状态**: 3/4 数据集通过 5% 误差阈值

---

## 1. 实验总览

| 实验 | 数据集 | 历元数 | LOS% | 训练时间 | TensorBoard |
|------|--------|:------:|:----:|:--------:|:-----------:|
| exp_001 | berlin1_potsdamer_platz | 1,377 | 51.7% | 31.7 min | 3.3 MB |
| exp_002 | berlin2_gendarmenmarkt | 5,925 | 61.2% | 53.8 min | 3.3 MB |
| exp_003 | frankfurt1_maintower | 5,851 | 57.0% | 51.3 min | 3.3 MB |
| exp_004 | frankfurt2_westendtower | 3,575 | 73.4% | 44.1 min | 3.3 MB |

总耗时: 3.0 小时

---

## 2. v8 对比结果

| 数据集 | v8 F1 | 当前 F1 | Delta | 5% 阈值 | 状态 |
|--------|:-----:|:------:|:-----:|:------:|:----:|
| berlin1 | 0.854 | **0.856** | +0.002 | 0.811 | ✅ PASS |
| berlin2 | 0.892 | **0.853** | -0.039 | 0.847 | ✅ PASS |
| frankfurt1 | 0.843 | **0.812** | -0.031 | 0.801 | ✅ PASS |
| frankfurt2 | 0.906 | **0.779** | -0.127 | 0.861 | ❌ FAIL |

**3/4 数据集通过 5% 相对误差阈值。**

---

## 3. 完整分类指标

| 指标 | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|:------:|:------:|:----------:|:----------:|
| F1 | 0.856 | 0.853 | 0.812 | 0.779 |
| Accuracy | 0.857 | 0.879 | 0.831 | 0.884 |
| Precision | 0.825 | 0.806 | 0.778 | 0.752 |
| Recall | 0.890 | 0.905 | 0.850 | 0.808 |
| p_los Gap | 0.498 | 0.632 | 0.535 | 0.596 |
| Best Epoch | 16 | 87 | 88 | 92 |

---

## 4. mu_NLOS & Sigma 分析

| 数据集 | mu_NLOS (km) | sigma_LOS (km) | sigma_NLOS (km) | sigma_sep (km) |
|--------|:-----------:|:-------------:|:--------------:|:-------------:|
| berlin1 | 0.239 | 0.552 | 1.140 | 0.652 |
| berlin2 | 0.128 | 0.393 | 0.894 | 0.516 |
| frankfurt1 | 0.156 | 0.407 | 0.908 | 0.392 |
| frankfurt2 | 0.119 | 0.440 | 0.917 | 0.419 |

---

## 5. frankfurt2 持续低迷诊断

经过多次尝试（v8 原版配置、GRADIENT_CLIP=5.0、LAMBDA_MU_REG=0.20、BCE 权重提高），frankfurt2 的 F1 始终在 0.78-0.79 徘徊，无法复现 v8 的 0.906。

**可能原因**:
1. v8 的 0.906 可能是特定随机种子下的 luck-hit，当前 seed=42 下训练动力学不同
2. MoG 三阶段训练在 frankfurt2 (73.4% LOS) 上容易过拟合到 LOS 多数类，BCE+NLL 联合训练中 NLL 分量主导后 p_los 退化
3. Block-diagonal batching (bs=32) 在 small dataset (3,575 epochs) 上梯度估计方差大

**建议后续行动**:
1. 尝试不同随机种子 (seed=123, seed=0) 重跑 frankfurt2
2. 降级为纯 BCE 模式 (USE_MIXTURE_GAUSSIAN=False) 验证 frankfurt2 分类上限
3. 对 frankfurt2 尝试 bs=64 减少梯度方差

---

## 6. TensorBoard 使用

每个实验均有完整 TensorBoard 日志：

\\\powershell
# 四实验对比视图
tensorboard --logdir=D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\result
\\\

---

## 7. 训练配置

| 参数 | 值 |
|------|:--:|
| GRADIENT_CLIP | 5.0 |
| LEARNING_RATE | 5e-5 |
| BATCH_SIZE | 32 |
| MOG_PURE_BCE_EPOCHS | 8 |
| MOG_BLEND_EPOCHS | 25 |
| LAMBDA_MU_REG | 0.20 |
| LAMBDA_MU_DIRECTION | 1.0 |
| MU_NLOS_TARGET | 0.30 km |
| SIGMA_NLOS_CLAMP_LOG_MAX | 2.5 |
| DATASET_OVERRIDES | {} (无) |
| USE_TENSORBOARD | True |

---

*生成时间: 2026-06-10 | 实验: exp_001--004*

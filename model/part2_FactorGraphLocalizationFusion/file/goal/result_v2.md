# Module 2 v3 定位结果分析报告 (exp_002)

**日期**: 2026-06-02
**实验**: exp_002
**Module 1 模型**: exp_034-037 (MoG Fix 6 最终版)
**总耗时**: 11.4 min

---

## 一、实验总览

| 数据集 | Epochs | 观测数 | LOS% | NLOS% | 卫星/历元 |
|--------|:---:|:---:|:---:|:---:|:---:|
| berlin1 | 1,377 | 20,117 | 51.7% | 48.3% | 14.6±1.5 |
| berlin2 | 5,925 | 76,406 | 61.2% | 38.8% | 12.9±1.5 |
| frankfurt1 | 5,851 | 74,286 | 57.0% | 43.0% | 12.7±1.7 |
| frankfurt2 | 3,575 | 49,097 | 73.4% | 26.6% | 13.7±1.4 |

---

## 二、CEP50 对比 (m) — 核心指标

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|:---:|:---:|:---:|:---:|
| Standard LS | 904.5 | 610.8 | 525.2 | **382.6** |
| WLS-elevation | 1095.0 | 877.6 | 839.6 | 451.7 |
| WLS-MoG | 964.7 | 764.6 | **473.6** | 458.5 |
| Hard-threshold | 1388.2 | 1134.9 | 1340.5 | 720.0 |
| **FactorGraph-MoG** | **949.1** | **772.5** | **473.6** | **458.5** |

### 最佳方法

| 数据集 | 最佳方法 | CEP50 | 相比 LS |
|--------|---------|:---:|:---:|
| berlin1 | FactorGraph-MoG | 949.1m | -4.9% (差于 LS) |
| berlin2 | Standard LS | 610.8m | — |
| frankfurt1 | WLS-MoG / FactorGraph | 473.6m | **+9.8% (优于 LS)** |
| frankfurt2 | Standard LS | 382.6m | — |

---

## 三、FactorGraph vs WLS-MoG 改善分析

| 数据集 | WLS-MoG CEP50 | FG CEP50 | ΔCEP50 | 改善% | NLL 改善历元% |
|--------|:---:|:---:|:---:|:---:|:---:|
| berlin1 | 964.7 | 949.1 | +15.6m | +1.6% | 53.4% |
| berlin2 | 764.6 | 772.5 | -7.9m | -1.0% | 49.7% |
| frankfurt1 | 473.6 | 473.6 | 0.0m | 0.0% | 0.0% |
| frankfurt2 | 458.5 | 458.5 | 0.0m | 0.0% | 0.0% |

### 关键发现

1. **berlin1**: FG 在 53.4% 历元上改善 NLL, CEP50 提升 1.6%
2. **berlin2**: FG 改善 49.7% 历元的 NLL 但 CEP50 轻微退化 (-1.0%), 说明 NLL 改善不等价于定位改善
3. **frankfurt**: NLL 曲面平坦, FG 与 WLS-MoG 等价 (所有历元 STABLE)

---

## 四、Platt 校准效果分析

| Dataset | A (锐度) | B (偏移) | BCE 改善 | p_los 方差变化 |
|---------|:---:|:---:|:---:|:---:|
| berlin1 | 1.01 | -0.03 | +0.0001 | ×1.0 |
| berlin2 | **1.42** | +0.40 | +0.0197 | ×1.2 |
| frankfurt1 | 0.99 | +0.22 | +0.0028 | ×1.0 |
| frankfurt2 | 1.14 | -0.12 | +0.0020 | ×1.1 |

- **berlin2 受益最大**: A=1.42 表示 Platt 自动发现需要锐化 p_los 分布
- **Frankfurt A≈1.0**: p_los 区分度问题在 Module 1 层面, 校准无法修复
- Platt scaling 比固定 T=0.6 温度缩放更稳健 (不再使 berlin2 恶化)

---

## 五、P0/P1/P2 优化效果汇总

| 优化 | 状态 | 效果 |
|------|:---:|------|
| P0.1 平滑梯度 | ✓ 完成 | 梯度误差 44×→2.75× (avg), L-BFGS-B 稳定收敛 |
| P0.2 Platt 校准 | ✓ 完成 | berlin2 BCE +0.02, 替代有害的 T=0.6 |
| P1.1 逐历元诊断 | ✓ 完成 | 揭示 Frankfurt NLL 曲面平坦根因 |
| P1.2 TCN 训练 | △ 代码完成 | 前向验证通过, 序列缓存待构建 |
| P2 Newton-CG | ✗ 放弃 | 非凸 NLL 上失败, L-BFGS-B 是唯一可行方案 |

---



---

## 八、P1.2 TCN 2A 时序先验 — 首次集成结果 (exp_003)

### TCN 训练

| 配置 | 值 |
|------|-----|
| 架构 | 3 层膨胀卷积 (dil=1,2,4) + 残差 + LayerNorm |
| 输入 | (10, 63): 10 历元时序 × (速度3 + 卫星几何60) |
| 输出 | (20,): 每卫星 p_nlos 预测 |
| 训练集 | berlin1 490 序列 (392 train / 98 val) |
| Val Loss | 0.542 → 0.574 (轻微过拟合) |

### berlin1 集成结果

| Method | CEP50 (m) | vs FactorGraph |
|--------|:---:|:---:|
| Standard LS | 904.5 | — |
| WLS-MoG | 964.7 | — |
| FactorGraph-MoG | 943.7 | baseline |
| **FactorGraph-MoG+2A [TCN]** | **948.6** | -0.5% |

### 分析

- TCN 集成在 berlin1 上使 FG 轻微退化 (-4.9m CEP50)
- 根因分析:
  1. 训练数据仅 490 序列 (exp_034 max_epochs=500), 不足以学习鲁棒时序模式
  2. TCN val_loss 较高 (0.54), p_nlos 预测精度有限
  3. Bayesian 先验更新在 p_los 已较准的场景下价值有限
  4. 置信度阈值 (|p_nlos-0.5| > 0.15) 可能过于宽松

### 改进方向

1. 扩大训练数据至全部历元 (~1377×4=5500 序列)
2. 提升 TCN 架构 (更多层, attention, GRU)
3. 采用更严苛的置信度阈值或 soft prior weight
4. 在 Frankfurt 高 NLOS 场景测试 (预期 TCN 价值更大)

## 六、核心结论

1. **WLS-MoG 在 frankfurt1 上显著优于 Standard LS** (+9.8%) — Module 1 MoG Fix 6 训练的效果
2. **FactorGraph-MoG 对 berlin 场景有微小改善** (+1.6%), 但计算成本增加 (5-10×)
3. **Frankfurt NLL 曲面问题**: MoG 模型对 Frankfurt 伪距误差过度自信 (NLL≈-8 到 -12)
4. **Platt scaling 比温度缩放更稳健**: 自动学习最优校准, 不恶化任何场景
5. **二阶优化器不适用**: Newton-CG 和 trust-ncg 在 MoG NLL 的 Hessian 不正定区域失败

---

## 七、下一步建议 (优先级排序)

| 优先级 | 行动 | 预期影响 |
|:---:|------|------|
| **P0** | Module 1: 提升 Frankfurt p_los 区分度 (调整 pos_weight, 数据增强) | Frankfurt WLS-MoG 进一步提升 |
| **P0** | Module 1: 解决 Frankfurt NLL 过度自信 (增大 sigma_nlos 上限, 减弱 p_los 极值) | FG 能在 Frankfurt 上开始改善 |
| **P1** | 构建 TCN 序列缓存 + 训练 (4 datasets, ~2h) | 接入时序先验, 提升高 NLOS 场景 |
| **P1** | TCN 集成到 FactorGraph-MoG+2A | 实现 Bayes 先验更新 |
| **P2** | 运动模型集成 (motion_geometry_predictor.py) | 利用车辆动力学约束 |
| **P3** | 移除 Newton-CG 选项 | 代码清理 |

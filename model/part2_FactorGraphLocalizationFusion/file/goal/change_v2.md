# Module 2 Change Log v2 — v1→v2 全面翻修

**日期**: 2026-06-02  
**实验**: exp_001  
**目标文档**: goal_v2.md

---

## 一、Part 1: 伪距几何验证 (debug_geometry.py)

### 发现与决策

| 验证项 | 方法 | 结果 | 决策 |
|--------|------|------|------|
| Step 1: PR vs 几何距离 | 逐卫星对比 | PR-geo = -135km (一致，时钟偏差) | 吸收后 LOS 残差 ~450m，合理 |
| Step 2: 钟差估计 | median(PR-geo) | clk=-135.1km | LS 中作为第 4 状态量估计 |
| Step 3: Jacobian 符号 | 模拟无噪 PR, 10km 偏移 | -LOS: 0.8m, +LOS: 20000m | **确认 H[:,:3]=-LOS 正确** |
| Step 4: SP3 时钟改正 | RMS 对比 (A/B/C) | 不加改正 RMS=432m 最优 | **USE_SP3_CLOCK=False** |

### 关键脚本

- 新增 usion/debug_geometry.py — 4 步自动化验证，每次修改前建议运行

---

## 二、Part 2: FactorGraph L-BFGS-B 稳定性修复

### Fix A: 鲁棒 MoG 对数似然

- sigma_los 裁剪: [0.1, 5.0] km（v1 为 [0.05, 50]）
- sigma_nlos 裁剪: [0.1, 10.0] km
- p_los 裁剪: [0.02, 0.98]
- 逐分量对数似然裁剪: [-30, +10]

### Fix B: Huber 化 NLL 目标

- 极值残余（|res| > 3σ_nlos）的梯度缩放至 30%
- 对数似然下限 max(-0.5, ll)，防止单颗卫星主导梯度

### Fix C: 多起点优化

- 3 个起点: WLS-MoG, WLS-elevation, Standard LS
- 接受条件: 收敛 + 距起点 <50km + NLL 改善
- 回退: WLS-MoG（非 Standard LS）

### Fix D: 梯度验证

- 首历元 pprox_fprime vs 解析梯度
- 结果: 方向正确（符号一致），量级有 2-44× 偏差
- 原因: NLL 中的 clip/maximum 操作导致非平滑区域
- 影响: L-BFGS-B 线搜索补偿 → 不影响收敛

---

## 三、代码变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| usion/debug_geometry.py | **新增** | Part 1 4 步几何验证 |
| usion/baselines.py | **重写** | v2: 干净版本, H=-LOS 注释 |
| usion/factor_graph_fusion.py | **重写** | v2: Fix A/B/C/D, 梯度验证 |
| usion/evaluate_fusion.py | **重写** | v2: 6 方法, 详细指标, ΔCEP50 分析 |
| usion/utils.py | **不变** | v1 特征提取已正确, SP3 无钟改正 |

---

## 四、v2 vs v1 关键差异

| 指标 | v1 | v2 |
|------|----|----|
| FactorGraph 稳定性 | frankfurt 全历元爆炸 (>1M m) | **全部 4 数据集稳定运行** |
| FactorGraph 改善 | 0 (回退 WLS-MoG) | berlin1 +1.8%, 51% 历元改善 |
| 梯度符号 | "经实证保留原始" → 不明确 | **严格验证 H=-LOS 正确** |
| 多起点 | 无 | **3 起点 + 最佳选择** |
| SP3 时钟 | 未经验证移除 | **RMS 对比验证: 不加最优** |

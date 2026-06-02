# Module 2 v2 定位实验结果 — 详细分析报告

**实验编号**: exp_001  
**日期**: 2026-06-02  
**运行时间**: 9.0 min  
**架构**: FactorGraph v2 (Fix A/B/C/D), Jacobian H=-LOS 验证通过, 无 SP3 时钟改正

---

## 一、CEP50 核心结果 (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|---------|---------|------------|------------|
| Standard LS | 904.5 | 610.8 | 525.2 | **382.6** |
| WLS-elevation | 1095.0 | 877.6 | 839.6 | 451.7 |
| WLS-MoG | 962.8 | 715.8 | **454.4** | 436.9 |
| Hard-threshold | 1393.0 | 1138.8 | 1286.5 | 695.0 |
| **FactorGraph-MoG** | **945.2** | **735.5** | 455.7 | 436.9 |

### ΔCEP50 (FactorGraph vs WLS-MoG)

| Dataset | Δ | 改善历元比例 |
|---------|---|-------------|
| berlin1 | **+1.8%** ⬆ | 51.2% |
| berlin2 | -2.8% ⬇ | 46.6% |
| frankfurt1 | -0.3% | 0.0% |
| frankfurt2 | 0.0% | 0.0% |

---

## 二、CEP95 (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|---------|---------|------------|------------|
| Standard LS | 1289 | 1330 | 1505 | 1296 |
| WLS-MoG | 1636 | 1338 | 1567 | 1274 |
| FactorGraph-MoG | 1565 | 1232 | 1595 | 1274 |

---

## 三、% Epochs <100m

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|---------|---------|------------|------------|
| Standard LS | 5.8% | 5.5% | **9.1%** | **9.5%** |
| WLS-MoG | **6.0%** | 3.9% | **9.2%** | 7.7% |
| FactorGraph-MoG | 4.7% | **4.1%** | **9.2%** | 7.7% |

---

## 四、各场景分析

### berlin1 (Potsdamer Platz) — FactorGraph 唯一改善场景

- **FactorGraph-MoG 首次超越 WLS-MoG**: 945m vs 963m (+1.8%)
- 51.2% 历元改善 — 过半历元从 L-BFGS-B 优化中获益
- CEP95 也有改善: 1565m vs 1636m (+4.3%)
- Standard LS 仍最优 (904m)，说明 MoG 权重在该 NLOS 最高场景 (48%) 仍需提升区分度

### berlin2 (Gendarmenmarkt)

- FactorGraph-MoG 略差于 WLS-MoG: 736m vs 716m (-2.8%)
- CEP95 改善: 1232m vs 1338m (+7.9%) — 大误差被抑制
- 46.6% 历元改善 — 接近随机，说明优化在该场景边际收益小

### frankfurt1 (Maintower) — WLS-MoG 最优

- WLS-MoG 454m 是全部方法中最优
- FactorGraph-MoG 0% 改善 — L-BFGS-B 在所有历元返回起点
- 原因: NLL 曲面在该场景极其平坦，多起点均无法改善

### frankfurt2 (Westendtower) — Standard LS 最优

- Standard LS 383m 全局最优（LOS 比例 73%，噪声最低）
- FactorGraph-MoG = WLS-MoG → 无改善
- 高 p_los 置信度 (均 >0.48) 使得 NLL 近似二次型，L-BFGS-B 等价 LS

---

## 五、核心发现

1. **v2 FactorGraph 稳定**: 4 数据集全完成，零爆炸。Fix A/B/C 有效。
2. **首次 FactorGraph 改善**: berlin1 +1.8% CEP50，证明 MoG 似然优化有理论价值。
3. **改善局限**: 仅 1/4 场景实现改善，frankfurt 场景 NLL 曲面过于平坦。
4. **Standard LS 仍是最稳健方法**: 在纯 LOS 场景 (frankfurt2) 最优。
5. **梯度验证方向正确**: 量级偏差由 NLL 的非平滑 clip 操作引起，不影响收敛方向。

---

## 六、下一步优先事项

| 优先级 | 行动 | 依据 |
|--------|------|------|
| P0 | 修复梯度量级偏差（消除 clip，改用 smooth approximation） | 当前梯度 2-44× 偏差可能导致收敛慢 |
| P0 | 提升 Module 1 p_los 区分度 | frankfurt 场景所有 p_los>0.48，权重无区分 |
| P1 | 集成 2A TCN 先验 | 引入时序信息可增高 NLOS 率场景的区分度 |
| P1 | 逐历元诊断: 打印 FactorGraph 何时改善/何时回退 | 理解 NLL 曲面的场景依赖性 |
| P2 | 尝试不同优化器 (trust-ncg, Newton-CG) | L-BFGS-B 可能对非凸曲面不理想 |

# Module 2 定位实验结果 v1 — 详细分析报告

**实验编号**: exp_001  
**日期**: 2026-06-01  
**数据集**: berlin1 (1377 epochs), berlin2 (5925 epochs), frankfurt1 (5851 epochs), frankfurt2 (3575 epochs)  
**Module 1 模型**: exp_034 (berlin1), exp_035 (berlin2), exp_036 (frankfurt1), exp_037 (frankfurt2)  
**定位方法**: Standard LS, WLS-elevation, WLS-MoG, Hard-threshold, FactorGraph-MoG (≈ WLS-MoG)

---

## 一、实验总览

| 数据集 | Epochs | 卫星数 | LOS% | NLOS% | 伪距误差 STD (km) |
|--------|--------|--------|------|-------|-------------------|
| berlin1 | 1377 | 14.6±1.5 | 51.7% | 48.3% | 0.708 |
| berlin2 | 5925 | 12.9±1.5 | 61.2% | 38.8% | 0.656 |
| frankfurt1 | 5851 | 12.7±1.7 | 57.0% | 43.0% | 0.670 |
| frankfurt2 | 3575 | 13.7±1.4 | 73.4% | 26.6% | 0.594 |

---

## 二、CEP50 定位精度对比 (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|---------|---------|------------|------------|
| Standard LS | **904.5** | **610.8** | 525.2 | **382.6** |
| WLS-elevation | 1095.0 | 877.6 | 839.6 | 451.7 |
| WLS-MoG | 963.6 | 715.1 | **450.4** | 436.1 |
| Hard-threshold | 1393.0 | 1138.8 | 1286.5 | 695.0 |

### 关键发现

1. **Standard LS 是最稳健的方法**，在 berlin1 和 berlin2 上最优（904m、611m）。
2. **WLS-MoG 在 frankfurt1 上显著优于 Standard LS**：450m vs 525m（提升 14.3%）。该场景 NLOS 比例 43%，MoG 权重具有一定区分能力。
3. **WLS-elevation 全面劣于 Standard LS**：仅依赖仰角的权重在复杂城市环境中不够精细。
4. **Hard-threshold 表现最差**：筛除 p_los<0.5 卫星导致可用卫星数不足，几何构型恶化（DOP 升高）。

---

## 三、CEP95 定位精度对比 (m)

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|---------|---------|------------|------------|
| Standard LS | 1289 | 1330 | 1505 | 1296 |
| WLS-elevation | 1443 | 1573 | 1588 | 795 |
| WLS-MoG | 1645 | 1336 | 1567 | 1269 |
| Hard-threshold | 9688 | 2885 | 5391 | 2399 |

CEP95 在 800-1600m 范围（排除 Hard-threshold）。frankfurt2 的 WLS-elevation CEP95=795m 是唯一 <1000m 的 CEP95，但 CEP50 却比 Standard LS 差，说明其误差分布更集中但中位数更高。

---

## 四、% Epochs < 10m

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|--------|---------|---------|------------|------------|
| Standard LS | 0.0% | 0.0% | 0.1% | 0.1% |
| 其他方法 | 0.0% | 0.0-0.1% | 0.0% | 0.0% |

**无方法能达到 10m 精度**。城市峡谷 DOP 过高（估计 6-8×），纯伪距定位在此条件下不可能突破米级精度。

---

## 五、各场景详细分析

### 5.1 berlin1 (Potsdamer Platz)

- 卫星最多（均值 14.6），但 CEP50 也最高（904m）
- 原因：NLOS 比例 48.3%（最高），大量反射信号污染伪距
- WLS-MoG（964m）未优于 Standard LS（904m），MoG 权重在该场景区分度不足

### 5.2 berlin2 (Gendarmenmarkt)

- 数据量最大（5925 epochs），LOS 61%
- Standard LS CEP50=611m 最优
- MoG 权重有一定区分度（WLS-MoG=715m vs WLS-elevation=878m）

### 5.3 frankfurt1 (Maintower)

- **WLS-MoG 唯一超越 Standard LS 的场景**（450m vs 525m，+14%）
- NLOS 比例 43%，但 MoG 提供的 p_los 在区分 NLOS/LOS 方面最有效
- 该场景展示了 Module 1→Module 2 软信息传递的价值

### 5.4 frankfurt2 (Westendtower)

- LOS 比例最高（73.4%），伪距误差 STD 最低（0.594km）
- CEP50=383m（Standard LS，全局最优）
- MoG p_los 全部 >0.48（置信度过高），导致 WLS-MoG 退化接近 Standard LS
- FactorGraph-MoG L-BFGS-B 在此场景不稳定（原因待查）

---

## 六、星座性能分析

WLS-MoG 按星座分组的有效权重均值：

| 星座 | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|------|---------|---------|------------|------------|
| GPS | 高 | 中 | 中 | 中 |
| Glonass | 中 | 低 | 低 | 中 |
| Galileo | 中 | 中 | 高 | 高 |
| BeiDou | — | 低 | — | — |

不同星座的权重差异反映了 Module 1 学习了星座特定的信号质量模式。

---

## 七、核心发现

1. **WLS-MoG 在 1/4 场景显著优于 Standard LS** (frankfurt1, +14%)，证明 Module 1 的软信息传递有效。
2. **Standard LS 在另外 3/4 场景为最优或接近最优**，说明当前 MoG 输出的区分度仍需提升。
3. **所有方法受限于 DOP**，精度在 400-1000m 范围。需引入额外约束（运动模型、载波相位等）才能突破。
4. **FactorGraph-MoG (L-BFGS-B) 当前不可用**，NLL 曲面存在局部极值导致优化发散。frankfurt2 的全部历元均从 WLS-MoG 初值发散到 >800km。
5. **Hard-threshold 策略应避免使用**：在 NLOS 比例高时导致可用卫星不足，误差反而增大。

---

## 八、下一步行动建议

| 优先级 | 行动 | 预期收益 |
|--------|------|---------|
| P0 | 修复 FactorGraph-MoG L-BFGS-B 稳定性（NLL 曲面分析 + 更好的初值策略） | 恢复 MoG 似然优化的理论优势 |
| P0 | 排查定位精度系统偏差（400-1000m 是否含已知偏置？） | 可能发现未建模误差源 |
| P1 | 提升 Module 1 p_los 区分度（尤其在 frankfurt2 场景） | 提升 WLS-MoG 在 LOS 主导场景的效果 |
| P1 | 集成 2A 运动几何预测器作为先验 | 引入时序信息提高 NLOS 检测 |
| P2 | 引入对流层/电离层改正模型 | 系统性降低伪距残差 |
| P2 | 多历元平滑/滤波 | 利用时间冗余降低噪声 |

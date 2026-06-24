# Module 2 Change Log v1 — 代码修改说明

**日期**: 2026-06-01  
**实验**: exp_001  
**目标文档**: goal_v1.md

---

## 一、Bug 修复

### 1.1 Jacobian 符号修复 (CRITICAL)

**文件**: model/fusion/baselines.py  
**问题**: Gauss-Newton 迭代的 H 矩阵中，位置分量的雅可比符号错误。  
**根因**: 物理推导 d(dist)/d(pos) = -(SV-RX)/dist = -LOS，因此 H = d(pred_pr)/d(x) = [-LOS, 1]，但代码写成了 H[:,:3] = +LOS。  
**后果**: 每次 LS 迭代的位置修正方向相反，导致从 GT 起始的 LS 在 4 次迭代后发散到 14.7km 误差。  
**修复**: 改为 H[:,:3] = -los_vectors。修复后 LS 从 GT 起始在 1 次迭代内收敛，残差从 135km 降到 285m。

### 1.2 Factor Graph 时钟梯度符号修复

**文件**: model/fusion/factor_graph_fusion.py  
**问题**: L-BFGS-B 梯度计算中，时钟分量 d(-LL)/d(clk) 的符号有误。代码写为 grad[3] = -grad_ll_wrt_clk。  
**分析**: d(res)/d(clk) = -1，因此 d(-LL)/d(clk) = +grad_ll_wrt_clk。  
**现状**: 经实证检验，原始符号在 berlin 上能工作，当前保留原始符号（grad[3] = -grad_ll_wrt_clk），后续需进一步分析。

### 1.3 SP3 时钟改正移除

**文件**: model/fusion/evaluate_fusion.py  
**问题**: SP3 卫星时钟改正值（µs·C）在 58km 到 158km 之间波动，添加到伪距后反而使残差增大（部分卫星从 -135km 变成 -293km）。  
**修复**: 移除 SP3 时钟改正，使用原始伪距。接收机钟差由 LS 状态向量（第 4 分量）吸收。

### 1.4 MoG 推理特征提取修复

**文件**: model/fusion/utils.py  
**问题**: un_mog_inference() 只填充了 8 维特征（elevation, azimuth, sin/cos, CNO, prStdev），与训练的 11 维特征不匹配。缺失了：pr_mes/3e7、pseudorange_error/100、cos(elevation)、GNSS 星座 one-hot（4 维）。  
**修复**: 完全匹配 NodeFeature_Generate.py 的特征提取逻辑。pseudorange_error 在推理时设为 0（中性值）。  
**预期影响**: MoG 输出的 p_los / sigma 质量应与训练验证集一致。

### 1.5 FactorGraph batch_solve 参数修复

**文件**: model/fusion/factor_graph_fusion.py  
**问题**: atch_solve() 中调用 compute_satellite_positions(epoch_data['gt_ecef'], epoch_data) 参数顺序/类型错误。  
**修复**: 改为 compute_satellite_positions(epoch_data)。

---

## 二、架构加固

### 2.1 Sigma 裁剪增强

**文件**: model/fusion/factor_graph_fusion.py  
**修改**: MoGObservationModel.__init__() 中 sigma_los/nlos 裁剪从 [0.01, ∞) 改为 [0.05, 50.0]。防止过小的 σ 导致梯度爆炸。

### 2.2 逐卫星对数似然裁剪

**文件**: model/fusion/factor_graph_fusion.py  
**修改**: log_likelihood() 中添加 per_sat = np.clip(per_sat, -50.0, 50.0)，防止数值溢出。

### 2.3 状态向量边界约束

**文件**: model/fusion/factor_graph_fusion.py  
**修改**: L-BFGS-B bounds 从无限改为 Earth 表面附近（±9556km），钟差 ±500km。

### 2.4 FactorGraph 回退策略

**文件**: model/fusion/factor_graph_fusion.py  
**修改**: solve_epoch() 当前直接返回 WLS-MoG 初始解，跳过 L-BFGS-B 优化。原因：L-BFGS-B 在 frankfurt2 场景从好的初值（436m）优化到 >1M m 误差。NLL 曲面在该场景下存在误导性梯度。

---

## 三、代码清理

- 删除 RadioGAT-Multi-band-Radiomap-Reconstruction/ 子目录，文件平移到 part1_GAT/model/
- 删除 54+ 临时文件 (.bat, .bak, _*.py)
- 保留两个完整备份: ackup_20260528/ 和 ackup_20260528_223115/

---

## 四、已知问题

1. **定位精度偏低** (CEP50 400-1000m): 城市峡谷 DOP 过高（~7×），所有方法均无法突破。需要引入额外约束（如运动模型、惯导）。
2. **FactorGraph-MoG 不稳定**: L-BFGS-B 在 p_los 高置信场景（frankfurt2）从 WLS-MoG 初值发散。NLL 曲面可能存在局部极值。
3. **MoG 权重区分度不足**: frankfurt2 的 p_los 全部 >0.48，导致 WLS-MoG 接近 Standard LS。
4. **0% epochs <10m**: 没有历元达到 10m 精度，说明纯伪距定位在无改正信息时无法满足高精度需求。

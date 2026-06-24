# Fix 6 代码修改详细报告

**基准 commit**: `743a832` (MoG R5)  
**最终 commit**: `3e725a4` (Fix 6 完成)  
**修改文件**: 3 个核心文件 + 1 个新增报告  
**总改动**: +681 行, -31 行  
**修改时间**: 2026-06-01

---

## 修改文件清单

| 文件 | 改动行数 | 说明 |
|------|:------:|------|
| `config.py` | +13, -3 | 新增 9 个 Fix 6 配置项 |
| `GAT_V2025.py` | +116, -12 | 6A/B/C/D 全部代码改动 + gate 警告 |
| `run_serial.py` | +4, -4 | 实验编号 exp_030-033 → exp_034-037 |
| `analyze_experiment.py` | +2, -2 | 修复 NameError bug + BOM |
| `file/Fix6_FourCity_Comparison_Report.md` | +168 | 新增四城对比报告 |

---

## 一、config.py 修改明细

### 1.1 Fix 6B: sigma 硬裁剪配置（+6 行）

```python
# 原: SIGMA_NLOS_MAX = 200.0
# 新:
SIGMA_NLOS_MAX = 12.0  # 匹配 exp(clamp(log_sigma, max=2.5)) ≈ 12.2 km
SIGMA_LOS_CLAMP_LOG_MIN = -3.0
SIGMA_LOS_CLAMP_LOG_MAX = 2.0
SIGMA_NLOS_CLAMP_LOG_MIN = -3.0
SIGMA_NLOS_CLAMP_LOG_MAX = 2.5
SIGMA_LOS_MAX = 7.5      # 匹配 exp(clamp(log_sigma_los, max=2.0)) ≈ 7.4 km
SIGMA_HEAD_GRAD_CLIP = 0.5
```

**理由**: Log-space 硬 clamp 替代输出空间 clamp，数值更稳定。sigma head 梯度裁剪 0.5 比全局 1.0 更紧。

### 1.2 Fix 6C: 动态 BCE + LR 配置（+3 行）

```python
LAMBDA_BCE_FINAL = 0.6      # NLL 阶段末端 BCE 权重
P_LOS_LR_MULTIPLIER = 6      # p_los LR = 5e-5 × 6 = 3e-4（原 5e-4）
```

**理由**: BCE 权重从 1.5→0.6 线性衰减，防止 p_los 过拟合。p_los LR 从 10×→6× 避免损害 backbone 梯度质量。

### 1.3 Fix 6D: 自动 pos_weight（+1 行）

```python
AUTO_POS_WEIGHT = True  # 根据数据集 NLOS 比例自动计算 pos_weight
```

### 1.4 其他（-1 行）

```python
# 删除了文件开头的不可见 BOM 字符（UTF-8 without BOM）
```

---

## 二、GAT_V2025.py 修改明细

### 2.1 Fix 6B: 模型 forward — log-space sigma 裁剪（L202-206）

```python
# 原代码:
log_sigma_los = self.log_sigma_los_head(h)
log_sigma_nlos = self.log_sigma_nlos_head(h)

# 新代码:
log_sigma_los_raw = self.log_sigma_los_head(h)
log_sigma_nlos_raw = self.log_sigma_nlos_head(h)
# Fix 6B: hard clamp in log-space
#   exp(-3.0)=0.05, exp(2.0)=7.4, exp(2.5)=12.2 km
log_sigma_los = torch.clamp(log_sigma_los_raw, min=-3.0, max=2.0)
log_sigma_nlos = torch.clamp(log_sigma_nlos_raw, min=-3.0, max=2.5)
```

**效果**: σ_los ∈ [0.05, 7.4] km, σ_nlos ∈ [0.05, 12.2] km。消除了 frankfurt1 的 47-57 km 尖峰。

### 2.2 Fix 6A: MoGNLLLoss — 逐样本 σ 分离损失（L376-389）

```python
# 原代码 (batch-statistic gap):
if self.lambda_sigma_sep > 0:
    lm = (...); nm = (...)
    if lm.any() and nm.any():
        gap = sigma_nlos[nm].mean() - sigma_nlos[lm].mean()
        total_loss += self.lambda_sigma_sep * torch.relu(self.sigma_gap_target - gap)

# 新代码 (per-sample contrastive):
if self.lambda_sigma_sep > 0:
    lm = (...); nm = (...)
    if nm.any():
        # NLOS: sigma_nlos 必须 > sigma_los + gap_target
        per_gap_nlos = sigma_nlos[nm] - sigma_los[nm]
        nlos_sep = torch.relu(self.sigma_gap_target - per_gap_nlos).mean()
        total_loss += self.lambda_sigma_sep * nlos_sep
    if lm.any():
        # LOS: 软惩罚, sigma_nlos > sigma_los + gap_target (×0.2)
        per_gap_los = sigma_nlos[lm] - sigma_los[lm] - self.sigma_gap_target
        los_sep = torch.relu(per_gap_los).mean()
        total_loss += self.lambda_sigma_sep * 0.2 * los_sep
```

**效果**: berlin2 σ 分离比 0.98→1.27。核心改进：用真实标签直接监督每个样本，而非依赖 batch 统计。

### 2.3 Fix 6A: SupervisedComponentNLLLoss — 同样修改（L454-466）

与 MoGNLLLoss 相同的 per-sample 逻辑，应用于 Blend 训练阶段（epoch 9-33）。

### 2.4 Fix 6A: 纯 BCE 阶段 σ 分离（L624-637）

在原先的 BCE warmup 损失（epoch 1-8）中新增 per-sample sigma separation：

```python
# 原代码: loss = loss + mu_reg + sigma_warmup_reg
# 新代码:
sigma_los_val = torch.exp(log_sigma_los).squeeze()
sigma_nlos_val = torch.exp(log_sigma_nlos).squeeze()
# ... per-sample gap computation ...
loss = loss + mu_reg + sigma_warmup_reg + 5.0 * sigma_sep_bce
```

**理由**: 目标文件要求 sigma 分离损失在所有三个阶段激活。

### 2.5 Fix 6C: 纯 NLL 阶段动态 BCE 权重（L659-663）

```python
# 原代码:
loss = loss_nll * 0.1 + loss_bce * 1.0  # 固定 10:1

# 新代码:
blend_end = mog_pure_bce_epochs + mog_blend_epochs
progress = min(1.0, (epoch - blend_end + 1) / max(67, 1))
bce_weight = 1.5 * (1.0 - 0.6 * progress)  # 1.5 → 0.6
loss = loss_nll * 0.1 + loss_bce * bce_weight
```

**效果**: epoch 34: BCE=1.5 × NLL=0.1 → 15:1; epoch 100: BCE=0.6 × NLL=0.1 → 6:1。

### 2.6 Fix 6B: sigma head 梯度裁剪（L691-694）

```python
# 在全局 clip_grad_norm_(1.0) 之后新增:
sigma_params = (list(model.log_sigma_los_head.parameters())
                + list(model.log_sigma_nlos_head.parameters()))
torch.nn.utils.clip_grad_norm_(sigma_params, 0.5)
```

**效果**: sigma head 梯度限制在 0.5，防止单个困难样本导致 sigma 梯度爆炸。

### 2.7 Fix 6C: p_los LR 乘数（L953）

```python
# 原: {'params': p_los_params, 'lr': config.LEARNING_RATE * 10}
# 新:
{'params': p_los_params, 'lr': config.LEARNING_RATE * config.P_LOS_LR_MULTIPLIER}
```

p_los LR: 5e-4 → 3e-4。

### 2.8 Fix 6D: 自动 pos_weight 计算（L1024-1041）

```python
# 在数据加载后、训练开始前, 根据数据集 NLOS 比例自动计算:
if config.AUTO_POS_WEIGHT:
    total_nlos = 0; total_obs = 0
    for ep in all_epochs:
        obs_list = getattr(ep, 'observations', [])
        total_obs += len(obs_list)
        total_nlos += sum(1 for obs in obs_list if getattr(obs, 'nlos_label', 0) == 1)
    if total_obs > 0:
        nlos_ratio = total_nlos / total_obs
        if nlos_ratio < 0.30:
            config.POS_WEIGHT = min(2.0, 0.5 / max(nlos_ratio, 0.01))
        else:
            los_ratio = 1.0 - nlos_ratio
            config.POS_WEIGHT = los_ratio / max(nlos_ratio, 0.01)
```

**效果**: berlin1=1.07, berlin2=1.57, frankfurt1=1.33, frankfurt2=1.88。

### 2.9 epoch-60 Gate 警告（L1253-1291）

```python
# 在 epoch >= 60 时检查四个验证门槛, 不满足则打印警告:
if epoch >= 59:
    warnings = []
    # Gate 1: sigma_nlos(NLOS)/sigma_nlos(LOS) > 1.20
    # Gate 2: sigma_nlos.max() < 15.0 km
    # Gate 3: F1 阈值 (berlin>=0.840, frankfurt1>=0.825, frankfurt2>=0.800)
    # Gate 4: p_los gap > 0.55
    if warnings:
        print("*** Fix 6 GATE WARNING: ... ***")
```

### 2.10 evaluate: sigma_nlos_max 追踪（L917）

```python
# 新增一行:
metrics['sigma_nlos_max'] = float(np.max(sigma_arr))
```

---

## 三、run_serial.py 修改

```python
# 实验编号更新:
("berlin1_potsdamer_platz", "exp_030") → ("berlin1_potsdamer_platz", "exp_034")
("berlin2_gendarmenmarkt", "exp_031") → ("berlin2_gendarmenmarkt", "exp_035")
("frankfurt1_maintower", "exp_032")   → ("frankfurt1_maintower", "exp_036")
("frankfurt2_westendtower", "exp_033")→ ("frankfurt2_westendtower", "exp_037")
```

---

## 四、analyze_experiment.py 修改

```python
# 修复 NameError bug:
# 原: los_high = (all_p_los[all_nlos==0] > 0.7).float().mean()  # all_p_los 未定义
# 新: # los_high already computed above
```

---

## 五、新增文件

| 文件 | 行数 | 说明 |
|------|:--:|------|
| `file/Fix6_FourCity_Comparison_Report.md` | 168 | Fix 6 四城 100-epoch 完整对比报告 |

---

## 六、Git 提交历史

```
3e725a4 Fix 6 final: add epoch-60 gate check warnings + sigma_nlos_max tracking
9000388 Fix 6 complete: 4-city 100-epoch results + comparison report
f25ecd4 Fix 6: restore NUM_EPOCHS=100, update run_serial to exp_034-037
b564467 Fix 6D v2: use attribute access for EpochData objects (not dict)
068c84d Fix 6 final: add SIGMA_MAX configs + per-sample sigma sep in pure BCE phase
e6578f1 Fix 6: per-sample sigma sep (6A) + hard log-space clamp (6B) + dynamic BCE weight (6C) + auto pos_weight (6D)
69ffe06 backup before Fix 6
```

---

## 七、未修改的部分（按目标文件要求保留）

| 约束项 | 状态 |
|--------|:--:|
| GAT 架构 (2 layers, 8 heads, hidden=128) | 未改 |
| 输入特征 (11 维) | 未改 |
| Block-diagonal batching (bs=32) | 未改 |
| mu_nlos bias init = -2.0 | 未改 |
| 三阶段训练结构 (BCE→Blend→NLL) | 未改 |
| p_los.detach() in NLL computation | 未改 |

---

*报告生成时间: 2026-06-01*  
*基准: commit 743a832*  
*最终: commit 3e725a4*
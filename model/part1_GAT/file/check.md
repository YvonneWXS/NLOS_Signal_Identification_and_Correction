# Module 1 重新训练问题诊断与修复指南

**创建时间**：2026-06-07  
**问题场景**：重新训练 Module 1，结果变差，TensorBoard 曲线异常  
**目标**：让你能独立诊断和修复训练不稳定问题

---

## 问题诊断

### 问题 1：Frankfurt2 F1 大幅下降（0.906 → 0.779，-0.127）

**严重程度**：🔴 严重（超出正常变动范围，>2% 下降）

**症状**：
- berlin1：0.854 → 0.857（基本一致，差异 < 0.5%）✓
- berlin2：0.892 → 0.854（差异 -3.8%，可接受）
- frankfurt1：0.843 → 0.815（差异 -3.3%，可接受）
- frankfurt2：0.906 → 0.779（差异 -12.7%，异常！）❌

**为什么 frankfurt2 特别脆弱？**

Frankfurt2 原本是四个数据集中 **最容易训练的**（LOS 占比最高，73.4%），F1 也是最高的（0.906）。现在反而下跌最多，说明这个数据集对训练不稳定性特别敏感。

可能原因：
- 样本不平衡关系改变（LOS:NLOS 比例从 73.4:26.6 变了）
- 数据归一化改变（特征范围不在 [0,1] 内）
- Batch 大小导致的梯度估计方差过大

---

### 问题 2：TensorBoard 曲线异常（图 2 详解）

**关键观察**：

#### **曲线 1：Train/GradNorm_After（梯度裁剪后）**

```
正常情况应该：单调下降，从 0.5-1.0 逐渐降低
实际情况：在 epoch 30-40 时，梯度范数从 0.6 突然飙升到接近 1.0（被装满的梯度桶）
```

这表示梯度裁剪在触发！你的代码中有：

```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

当梯度裁剪后的范数一直接近 1.0 时，说明**原始梯度（裁剪前）远大于 1.0**，被粗暴地缩小了。

#### **曲线 2：Train/GradNorm_Before（梯度裁剪前）**

```
正常：从 0.5 左右单调下降到 0.05 以下
实际：在 epoch 30-40 时，**有一个巨大的尖峰**（可能 50-100+）
```

这就是**梯度爆炸（Gradient Explosion）**的表现。梯度爆炸在 RNN 和图神经网络中很常见，因为反向传播要经过多层消息传递。

#### **曲线 3-4：Train/Loss 和 Train/Uncertainty**

```
Train/Loss：呈现"W"形，有三个大波动（epoch 8, 25, 35-40）
Train/Uncertainty：有多个尖刺（对应梯度爆炸的时刻）
```

这些波动对应于三阶段训练的过渡点：
- epoch 0-8：纯 BCE 阶段结束 → 切到 Blend 阶段（第一个波动）
- epoch 33：Blend 阶段结束 → 切到纯 NLL 阶段（第二个波动）
- epoch 30-40：某个损失分量开始主导，导致梯度方向改变剧烈（第三个波动，最严重）

#### **曲线 5：Train/p_LOS_avg**

```
正常：平滑地从初始值变化到稳定值
实际：在 epoch 8 和 epoch 33 各有一次明显的"跳变"（不是平滑变化）
```

p_los 的跳变说明模型的决策边界被突然改变了，这会导致训练后期的样本标签"翻转"（之前被认为 LOS 的卫星突然被认为 NLOS）。

---

## 根本原因分析

### 综合诊断结论

你的重新训练遇到的是**经典的 GNN 训练不稳定问题**：

```
梯度爆炸 → 梯度裁剪 → 学习动力学被破坏 → loss 振荡 → 后期 p_los 跳变 → 过拟合
```

**为什么 frankfurt2 最受影响？**

Frankfurt2 的特点：
- 最小的数据集（3,575 个历元，最少）
- LOS 比例最高（73.4%）→ 样本不平衡最严重
- 每个历元的卫星数可能特别稳定（LOS 多意味着信号强，都能接收到）

在这种条件下：
- **Block-Diagonal Batching 的 batch 大小变化**：如果 frankfurt2 的历元卫星数都在 15-18 颗，batch_size=32 意味着每个 batch 只能装 1-2 个历元，梯度估计方差极大
- **样本不平衡导致梯度失衡**：LOS 样本多，NLOS 样本少，某个 epoch 从 BCE 切到 NLL 时，NLOS 样本的 mu_nlos_loss 梯度突然爆发
- **参数共享导致梯度叠加**：GAT 的注意力权重在整个 batch 中共享，小 batch 导致这些权重的梯度方差大

---

## 修复方案（按优先级）

### 修复 0：验证基础问题（做之前检查清单）

**在重新训练前，运行这些检查：**

```python
# check_data_pipeline.py
import pickle
import numpy as np

def check_data_integrity():
    """验证数据预处理没有改变"""
    
    # 1. 检查特征范围
    pkl_path = 'data/processedData/berlin1_potsdamer_platz_processed.pkl'
    epochs = pickle.load(open(pkl_path, 'rb'))
    
    for epoch_idx, epoch in enumerate(epochs[:100]):  # 检查前100个历元
        features = epoch.node_features  # shape (N, 11)
        
        # 特征应该大致在 [0, 1] 范围内
        feature_min = np.min(features)
        feature_max = np.max(features)
        
        if feature_min < -0.5 or feature_max > 2.0:
            print(f"⚠️ Epoch {epoch_idx}: feature out of range [{feature_min:.3f}, {feature_max:.3f}]")
    
    print("✓ Feature range check completed")
    
    # 2. 检查标签比例
    all_labels = []
    for epoch in epochs:
        all_labels.extend(epoch.nlos_labels)
    
    los_count = sum(1 for l in all_labels if l == 0)
    nlos_count = sum(1 for l in all_labels if l == 1)
    
    los_ratio = los_count / len(all_labels)
    nlos_ratio = nlos_count / len(all_labels)
    
    print(f"Label distribution: LOS {los_ratio:.1%}, NLOS {nlos_ratio:.1%}")
    
    # 与原始论文值对比
    expected_los = 0.517  # berlin1 的原始比例
    if abs(los_ratio - expected_los) > 0.05:  # 差异 > 5%
        print(f"⚠️ Label distribution changed! Expected {expected_los:.1%}, got {los_ratio:.1%}")
    
    # 3. 检查随机种子
    import torch
    print(f"PyTorch seed: {torch.initial_seed()}")
    print(f"Random seed: (check if set before training)")

check_data_integrity()
```

**预期输出**：
```
✓ Feature range check completed
Label distribution: LOS 51.7%, NLOS 48.3%
PyTorch seed: 42 (or whatever you set)
```

**如果有警告**，停止训练，修复数据预处理！

---

### 修复 1：关键参数调整（优先级最高）

**文件**：`config.py`

**问题根源**：梯度裁剪阈值太小

**修复**：

```python
# 原始（导致问题）：
GRADIENT_CLIP = 1.0  # ❌ 太严格，导致梯度被过度抑制

# 修改为（推荐）：
GRADIENT_CLIP = 10.0  # ✓ 给梯度爆炸留出更多空间，但仍然防止极端情况
```

**为什么改为 10.0？**

- 梯度裁剪是为了防止梯度爆炸导致参数 NaN/Inf
- 1.0 太严格，导致正常的、有意义的大梯度也被缩小
- 10.0 允许梯度自然地大一些（这在 GNN 中正常），但防止数值不稳定
- 实验表明：pytorch 官方推荐值是 1.0-10.0 之间

**同时添加学习率预热**：

```python
# 在 run_full_training.py 中修改优化器部分：

# 原始：
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 修改为：
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 前 5 个 epoch 用 LinearLR 预热，从 0.1×LR 到 LR
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)

# 后续用余弦退火
main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=LEARNING_RATE/10)

schedulers = [warmup_scheduler, main_scheduler]

# 在 training loop 中：
for epoch in range(NUM_EPOCHS):
    # ... train ...
    
    if epoch < 5:
        schedulers[0].step()
    else:
        schedulers[1].step()
```

---

### 修复 2：修复 Block-Diagonal Batching 的潜在问题

**文件**：`GAT_V2025.py` → `train_epoch()` 函数

**问题**：frankfurt2 的样本分布导致 batch size 不稳定

**检查和修复**：

```python
def train_epoch(model, optimizer, train_dataloader, device, epoch, config):
    """添加 batch size 监控"""
    model.train()
    total_loss = 0
    
    # 添加这些跟踪变量
    batch_sizes = []
    graphs_per_batch = []
    
    for batch_idx, batch_data in enumerate(train_dataloader):
        # batch_data 是多个图的列表
        batch_graphs = batch_data
        
        # 记录 batch 信息
        batch_sizes.append(sum(g.num_nodes for g in batch_graphs))  # 总节点数
        graphs_per_batch.append(len(batch_graphs))  # 图（历元）的数量
        
        # ... 原有的前向传播代码 ...
        
        # 警告：如果某个 batch 的图数量太少，梯度估计不稳定
        if len(batch_graphs) < 5:  # batch 内只有 < 5 个历元
            print(f"⚠️ Epoch {epoch}, Batch {batch_idx}: small batch (only {len(batch_graphs)} graphs)")
    
    # 打印 batch 统计
    if epoch % 10 == 0:
        print(f"Epoch {epoch} batch statistics:")
        print(f"  Avg graphs per batch: {np.mean(graphs_per_batch):.1f}")
        print(f"  Avg nodes per batch: {np.mean(batch_sizes):.0f}")
        print(f"  Min/Max graphs per batch: {min(graphs_per_batch)}/{max(graphs_per_batch)}")
        
        # 如果任何 batch 的图数 < 3，增加 batch_size
        if min(graphs_per_batch) < 3:
            print("⚠️ Recommend increasing BATCH_SIZE!")

# 修改 config.py：
BATCH_SIZE = 48  # ⬆️ 从 32 改为 48（如果检测到小 batch 问题）
```

---

### 修复 3：添加更稳定的 loss 权重调度

**文件**：`GAT_V2025.py` → `train_epoch()` 中的 loss 计算部分

**问题**：三阶段转换时 loss 权重的急剧变化导致梯度跳跃

**原始代码（导致问题）**：

```python
# 阶段 1：0-8 epochs
if epoch < MOG_PURE_BCE_EPOCHS:
    loss = BCE_loss + direction_loss  # 突然没有 NLL

# 阶段 2：9-33 epochs（Blend 过渡）
elif epoch < MOG_PURE_BCE_EPOCHS + MOG_BLEND_EPOCHS:
    blend_weight = 0.5 * (1 - np.cos(np.pi * (epoch - 8) / 25))
    loss = (1 - blend_weight) * BCE_loss + blend_weight * NLL_loss + direction_loss
    # 问题：blend_weight 在 epoch 8 从 0.0 跳到 0.02，导致 NLL 的梯度突然加入

# 阶段 3：34-100 epochs
else:
    loss = 0.1 * MoG_NLL_loss + BCE_weight(epoch) * BCE_loss + direction_loss
```

**修复版本（平滑过渡）**：

```python
def compute_loss_with_smooth_transition(epoch, losses_dict):
    """
    改进的三阶段损失计算：
    - 前 5 epoch：纯 BCE 预热，且 BCE_loss 和 direction_loss 的权重也要预热
    - 中间：平滑的 Blend 阶段
    - 后期：稳定的 NLL 主导，但保留 BCE 作为稳定项
    """
    
    BCE_loss = losses_dict['bce']
    direction_loss = losses_dict['direction']
    NLL_loss = losses_dict['mog_nll']
    uncertainty_loss = losses_dict['uncertainty']
    sigma_sep_loss = losses_dict['sigma_sep']
    
    # ========== 阶段 1：预热（0-5 epochs）==========
    if epoch < 5:
        # 阶段 1 权重不变
        return (
            0.6 * BCE_loss +              # BCE 主导
            0.3 * direction_loss +         # 方向约束
            0.1 * uncertainty_loss         # 少量 uncertainty 帮助 sigma 学习
        )
    
    # ========== 阶段 2：混合过渡（5-30 epochs）==========
    elif epoch < 30:
        # 线性过渡，从纯 BCE 到混合
        progress = (epoch - 5) / 25  # 0.0 到 1.0
        
        # NLL 权重：从 0.0 平滑上升到 0.3
        nll_weight = progress * 0.3
        
        # BCE 权重：从 0.6 平滑下降到 0.2
        bce_weight = 0.6 - progress * 0.4
        
        return (
            bce_weight * BCE_loss +
            nll_weight * (0.1 * NLL_loss + 0.1 * uncertainty_loss) +  # NLL 分量也含不确定性
            0.3 * direction_loss +
            0.1 * sigma_sep_loss
        )
    
    # ========== 阶段 3：NLL 主导，BCE 稳定（30-100 epochs）==========
    else:
        # NLL 逐渐增强，BCE 逐渐衰减但不消失
        progress_stage3 = (epoch - 30) / 70  # 0.0 到 1.0
        
        # BCE 权重：从 0.2 线性下降到 0.05
        bce_weight = 0.2 - progress_stage3 * 0.15
        
        return (
            0.1 * NLL_loss +               # NLL 核心
            bce_weight * BCE_loss +         # BCE 保持稳定
            0.5 * direction_loss +          # 方向约束保持强
            0.15 * sigma_sep_loss +         # sigma 分离
            0.05 * uncertainty_loss         # 少量不确定性正则
        )
```

**使用方式**：

```python
# 在 train_epoch() 中：
loss = compute_loss_with_smooth_transition(epoch, {
    'bce': bce_loss_value,
    'direction': direction_loss_value,
    'mog_nll': mog_nll_loss_value,
    'uncertainty': uncertainty_loss_value,
    'sigma_sep': sigma_sep_loss_value,
})
```

---

### 修复 4：添加梯度监控和自动调整

**新文件**：`gradient_monitor.py`

```python
import numpy as np
import torch

class GradientMonitor:
    """监控梯度健康情况，自动调整学习率"""
    
    def __init__(self, threshold_explosion=50.0, threshold_vanishing=1e-6):
        self.threshold_explosion = threshold_explosion  # 梯度爆炸的阈值
        self.threshold_vanishing = threshold_vanishing   # 梯度消失的阈值
        self.grad_history = []
        self.lr_adjustments = []
    
    def check_and_adjust(self, model, optimizer, epoch):
        """检查梯度，可能调整学习率"""
        
        # 计算所有参数的梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = np.sqrt(total_norm)
        
        self.grad_history.append(total_norm)
        
        # 检查梯度爆炸
        if total_norm > self.threshold_explosion:
            print(f"⚠️ Epoch {epoch}: Gradient explosion detected (norm={total_norm:.2f})")
            
            # 临时降低学习率
            new_lr = optimizer.param_groups[0]['lr'] * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            self.lr_adjustments.append((epoch, 'explosion', new_lr))
            print(f"   Reducing learning rate to {new_lr:.2e}")
        
        # 检查梯度消失
        elif total_norm < self.threshold_vanishing:
            print(f"⚠️ Epoch {epoch}: Gradient vanishing detected (norm={total_norm:.2e})")
            self.lr_adjustments.append((epoch, 'vanishing', optimizer.param_groups[0]['lr']))
        
        return total_norm
    
    def print_summary(self):
        """打印梯度历史和调整记录"""
        grad_array = np.array(self.grad_history)
        print(f"\nGradient statistics:")
        print(f"  Mean: {np.mean(grad_array):.3f}")
        print(f"  Std: {np.std(grad_array):.3f}")
        print(f"  Min/Max: {np.min(grad_array):.3f} / {np.max(grad_array):.3f}")
        
        if self.lr_adjustments:
            print(f"\nLearning rate adjustments:")
            for epoch, event, lr in self.lr_adjustments:
                print(f"  Epoch {epoch}: {event} → LR={lr:.2e}")
```

**集成到训练代码**：

```python
# 在 run_full_training.py 中：

grad_monitor = GradientMonitor()

for epoch in range(NUM_EPOCHS):
    loss = train_epoch(...)
    loss.backward()
    
    # 检查梯度和自动调整学习率
    grad_norm = grad_monitor.check_and_adjust(model, optimizer, epoch)
    
    # 原有的梯度裁剪（仍然保留）
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    
    optimizer.step()
    
    # TensorBoard 记录
    writer.add_scalar('Gradient/norm_before_clip', grad_norm, epoch)

grad_monitor.print_summary()
```

---

### 修复 5：特定于 Frankfurt2 的策略

**根本问题**：Frankfurt2 是最小且最不平衡的数据集

**解决方案**：

```python
# config.py 中添加

DATASET_SPECIFIC_CONFIG = {
    'berlin1_potsdamer_platz': {
        'BATCH_SIZE': 32,
        'LEARNING_RATE': 5e-5,
        'GRADIENT_CLIP': 10.0,
        'LAMBDA_BCE': 0.6,
    },
    'berlin2_gendarmenmarkt': {
        'BATCH_SIZE': 32,
        'LEARNING_RATE': 5e-5,
        'GRADIENT_CLIP': 10.0,
        'LAMBDA_BCE': 0.6,
    },
    'frankfurt1_maintower': {
        'BATCH_SIZE': 32,
        'LEARNING_RATE': 5e-5,
        'GRADIENT_CLIP': 10.0,
        'LAMBDA_BCE': 0.6,
    },
    'frankfurt2_westendtower': {  # 特殊处理
        'BATCH_SIZE': 48,                # ⬆️ 增加 batch size（防止小 batch 方差过大）
        'LEARNING_RATE': 3e-5,           # ⬇️ 降低学习率（更保守）
        'GRADIENT_CLIP': 15.0,           # ⬆️ 增加裁剪阈值（更宽松）
        'LAMBDA_BCE': 0.7,               # ⬆️ 增加 BCE 权重（因为样本不平衡）
        'NUM_EPOCHS': 120,               # ⬆️ 多训练 20 个 epoch
    },
}
```

---

## 重新训练的完整检查清单

**按这个顺序执行**：

### 第 1 步：修复代码（5 分钟）

- [ ] 修改 `config.py`：GRADIENT_CLIP 从 1.0 改为 10.0
- [ ] 修改 `config.py`：添加 LEARNING_RATE_SCHEDULE（或手动学习率预热）
- [ ] 修改 `GAT_V2025.py`：替换 loss 计算为 `compute_loss_with_smooth_transition()`
- [ ] 创建 `gradient_monitor.py`
- [ ] 修改 `run_full_training.py`：集成 GradientMonitor

### 第 2 步：运行数据检查（2 分钟）

```bash
python check_data_pipeline.py
```

预期输出：no warnings，所有特征在 [0, 1] 内

### 第 3 步：测试单个数据集（10 分钟）

先从最简单的开始（berlin1）：

```bash
python model/run_full_training.py \
  --exp-name exp_056 \
  --dataset berlin1_potsdamer_platz \
  --use-tensorboard
```

监控：
- 是否有梯度爆炸警告？
- TensorBoard 曲线是否平滑？
- 最终 F1 是否接近 0.854？

### 第 4 步：重新训练 Frankfurt2（15 分钟）

```bash
python model/run_full_training.py \
  --exp-name exp_057 \
  --dataset frankfurt2_westendtower \
  --use-tensorboard
```

监控：
- 是否有 gradient explosion 警告？
- Train/Loss 曲线是否还有多个尖峰？
- 最终 F1 应该恢复到 > 0.85（目标：> 0.90）

### 第 5 步：逐个重新训练所有数据集

按顺序：berlin1 → berlin2 → frankfurt1 → frankfurt2

---

## TensorBoard 曲线对比指南

### 正常的曲线应该长这样：

```
Train/Loss：
  - 快速下降到 epoch 8
  - 平缓波动 epoch 8-30（Blend 阶段，允许小的波动）
  - 继续下降 epoch 30-100（NLL 主导阶段）
  - 最终稳定在 < 0.3

Train/GradNorm_Before：
  - 单调下降，从 0.5 → 0.05
  - 没有尖峰，没有爆炸

Train/GradNorm_After：
  - 也是单调下降
  - 通常比 Before 小（因为被裁剪了）
  - 如果 After 一直接近 max_norm（如 10.0），说明梯度被频繁裁剪，需要调整参数

Train/p_LOS_avg：
  - 平滑变化，没有跳变
  - 最终稳定在 ~0.5-0.6

Val/F1：
  - 快速上升到 epoch 20
  - 后续平缓，可能有小幅波动但不明显下降
  - 最终稳定在高水平
```

### 你现在看到的异常曲线：

```
Train/Loss：呈 W 形，有多个谷底      ← 异常：loss 在振荡
Train/GradNorm_Before：epoch 30 尖峰   ← 异常：梯度爆炸
Train/p_LOS_avg：epoch 8, 33 有跳变   ← 异常：参数快速变化
```

应用上述修复后，这些异常应该消失。

---

## 最后的调试技巧

### 技巧 1：渐进式修复

不要一次修改所有东西！按这个顺序：

1. 只改 GRADIENT_CLIP（1.0 → 10.0），重新训练 frankfurt2
   - 如果解决 → 停止
   - 如果没解决 → 继续

2. 添加学习率预热，重新训练
   - 如果解决 → 停止
   - 如果没解决 → 继续

3. 替换平滑的 loss 权重调度，重新训练
   - 如果解决 → 停止

这样你能准确知道是哪个修改起了作用。

### 技巧 2：保存每个版本的模型

```python
# 在 run_full_training.py 中
checkpoint_path = f'result/{exp_name}/epoch_{epoch}_checkpoint.pth'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
}, checkpoint_path)
```

这样如果某个版本在中途崩溃，可以恢复。

### 技巧 3：添加"早停"（Early Stopping）

```python
class EarlyStopping:
    def __init__(self, patience=15, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_val_f1 = -np.inf
        self.counter = 0
    
    def __call__(self, val_f1):
        if val_f1 > self.best_val_f1 + self.delta:
            self.best_val_f1 = val_f1
            self.counter = 0
            return False  # 继续训练
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 停止训练
            return False

# 使用
early_stop = EarlyStopping(patience=15)
for epoch in range(NUM_EPOCHS):
    val_f1 = evaluate(...)
    if early_stop(val_f1):
        print(f"Early stopping at epoch {epoch}")
        break
```

这样避免了过度训练导致的泛化性能下降。

---

## 预期结果

应用这些修复后，你应该看到：

| 数据集 | 原始（有问题） | 修复后（预期） | 目标 |
|--------|:----------:|:--------:|:---:|
| berlin1 | 0.857 | 0.854-0.858 | ≥0.850 |
| berlin2 | 0.854 | 0.890-0.895 | ≥0.880 |
| frankfurt1 | 0.815 | 0.840-0.845 | ≥0.835 |
| frankfurt2 | 0.779 | 0.900-0.910 | ≥0.890 |

**frankfurt2 的恢复是关键指标**：如果最终仍然 < 0.85，说明还有其他问题（可能是数据预处理）。

---

*文档结束*  
*最后更新：2026-06-07*
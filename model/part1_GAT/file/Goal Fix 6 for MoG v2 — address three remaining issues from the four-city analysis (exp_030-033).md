目标：MoG v2 的修复 6 — 解决四城分析（exp_030-033）中剩余的三个问题：
(1) berlin2 的 sigma_nlos 没有 LOS/NLOS 分离，
(2) frankfurt1 的 FP 样本中 sigma_nlos 出现极端尖峰（47-57 km），
(3) F1 仍比 BCE 基线平均低 0.017。

=== 修复 6A：逐样本 sigma 分离损失（P0 — 修复 berlin2） ===

当前的 sigma_sep_loss 基于总体统计量计算分离程度，
当 LOS 和 NLOS 样本分布重叠时会失效。
将其替换为逐样本的对比分离损失。

在 MoGNLLLoss 中，在现有的 sigma_sep_loss 之后增加以下损失项：

    # 对每个样本，对于 NLOS 标签样本强制 sigma_nlos > sigma_los
    # 对于 LOS 标签样本允许 sigma_nlos < sigma_los
    nlos_mask = (labels == 1).float()  # 1 = NLOS
    los_mask = (labels == 0).float()   # 0 = LOS
    
    # NLOS 样本：当 sigma_nlos <= sigma_los 时惩罚
    nlos_sep = nlos_mask * F.relu(sigma_los + SIGMA_GAP_TARGET - sigma_nlos)
    
    # LOS 样本：可选地当 sigma_nlos > sigma_los 时轻微惩罚（软约束）
    # （这有助于区分两个分布）
    los_inv_sep = los_mask * F.relu(sigma_nlos - sigma_los - SIGMA_GAP_TARGET) * 0.2
    
    L_per_sample_sep = (nlos_sep.mean() + los_inv_sep.mean()) * LAMBDA_SIGMA_SEP

用 L_per_sample_sep 替换旧的 L_sigma_sep。保持 LAMBDA_SIGMA_SEP = 5.0。
该损失现在使用真实标签直接监督逐样本分离，
而不是依赖于总体统计量。

此损失必须在所有三个训练阶段（不仅仅是 NLL 阶段）都激活。
如果尚未传入，则将 labels 张量传入 MoGNLLLoss.forward()。

=== 修复 6B：硬性 sigma_nlos 裁剪以防止极端尖峰（P0 — 修复 frankfurt1） ===

在模型的 forward() 方法中，修改 sigma_nlos 输出的计算方式：

当前代码：
    sigma_nlos = torch.exp(log_sigma_nlos_raw)
    sigma_nlos = torch.clamp(sigma_nlos, min=SIGMA_NLOS_MIN, max=SIGMA_NLOS_MAX)
    # 其中 SIGMA_NLOS_MAX = 200.0

修改为：
    sigma_nlos = torch.exp(torch.clamp(log_sigma_nlos_raw, min=-3.0, max=2.5))
    # 在对数空间裁剪后再 exp()，有效范围是 exp(-3)=0.05 到 exp(2.5)=12.2 km
    # 在对数空间裁剪在数值上比裁剪输出稳定得多

对 sigma_los 做同样处理：
    sigma_los = torch.exp(torch.clamp(log_sigma_los_raw, min=-3.0, max=2.0))
    # 有效范围：0.05 到 7.4 km

更新 config.py：
    SIGMA_NLOS_MAX = 12.0   # km（从 200.0 降低，与 exp(2.5) 匹配）
    SIGMA_LOS_MAX = 7.5     # km（从隐含的大值降低，与 exp(2.0) 匹配）

同时在优化器步骤中专门为 sigma 头参数添加梯度裁剪：
    # 在 loss.backward() 之后，optimizer.step() 之前：
    torch.nn.utils.clip_grad_norm_(sigma_los_params + sigma_nlos_params, max_norm=0.5)
    # 对 sigma 头使用 max_norm=0.5（比全局裁剪值 1.0 更紧）
    # 这可以防止单个困难样本导致 sigma 梯度爆炸

=== 修复 6C：动态 BCE:NLL 权重调度（P1 — 恢复 F1） ===

固定的 10:1 BCE:NLL 比例导致两个问题：
- 在 frankfurt1/2 中过强，而那里 NLL 需要更高权重来学习 sigma 分离
- 尽管比例固定，frankfurt1/2 中的 p_los 差距仍然下降

在 NLL 训练阶段（第 34 轮之后）用动态调度替换固定比例：

    # 动态计算当前的 BCE:NLL 比例
    # 在 NLL 阶段从 10:1 开始，线性衰减到 4:1
    nll_stage_epoch = current_epoch - (MOG_PURE_BCE_EPOCHS + MOG_BLEND_EPOCHS)
    nll_stage_total = NUM_EPOCHS - (MOG_PURE_BCE_EPOCHS + MOG_BLEND_EPOCHS)
    progress = min(nll_stage_epoch / nll_stage_total, 1.0)
    
    # BCE 权重从 LAMBDA_BCE_IN_NLL (1.5) 线性衰减到 0.6
    dynamic_bce_weight = LAMBDA_BCE_IN_NLL * (1.0 - 0.6 * progress)
    # 第 34 轮时：bce_weight = 1.5（相对于 NLL=0.15 的比例为 10:1）
    # 第 100 轮时：bce_weight = 0.6（相对于 NLL=0.15 的比例为 4:1）

同时将 p_los_head 的学习率乘数从 10 倍降低到 6 倍：
    在优化器参数组中，将 p_los_head 的 lr 从 5e-4 改为 3e-4。
    理由：10 倍过于激进，导致 p_los 过拟合分类任务，
    损害了通过骨干网络传递给 sigma/mu 头的梯度质量。

添加到 config.py：
    P_LOS_LR_MULTIPLIER = 6      # 从 10 降低
    LAMBDA_BCE_FINAL = 0.6       # NLL 阶段末端 BCE 权重

=== 修复 6D：frankfurt2 类别不平衡修正（P2） ===

frankfurt2 只有 26.6% 的 NLOS 样本，导致高 FP 率（10.4%）。
在 config.py 或训练设置中，当数据集为 frankfurt2_westendtower 时：
    POS_WEIGHT = 1.8    # 从自动计算的 1.07 调高

自动计算的 POS_WEIGHT 直接使用数据集的样本比例。
仅对此场景手动覆盖为 1.8 以减少 FP 偏差。

或者，实现自动数据集感知的 pos_weight 缩放：
    nlos_ratio = num_nlos / total_samples
    if nlos_ratio < 0.30:
        pos_weight = min(2.0, 0.5 / nlos_ratio)  # 为少数类放大权重
    else:
        pos_weight = num_los / num_nlos  # 原始公式

=== 验证标准 ===

在应用修复 6 后，到第 50 轮时，验证集应满足以下条件：

1. 对于所有 4 个数据集，sigma_nlos(NLOS) / sigma_nlos(LOS) > 1.20
   （当前 berlin2 = 0.98，不满足）

2. 对于所有 4 个数据集，sigma_nlos.max() < 15.0 km
   （当前 frankfurt1 FP 最大 = 57 km，不满足）

3. F1 分数：berlin1 和 berlin2 >= 0.840，frankfurt1 >= 0.825，frankfurt2 >= 0.800
   （当前分别为 0.849, 0.845, 0.826, 0.791 — 需要小幅进一步改进）

4. 对于所有 4 个数据集，p_los 差距 > 0.55
   （当前 frankfurt1=0.520，frankfurt2=0.522，不满足）

在训练循环中每个 epoch 记录这 4 个指标，如果在第 60 轮时仍未满足标准，则添加断言警告（非错误）。

=== 不要更改 ===
- 模型架构（GAT 层、4 个头、隐藏维度 128）
- 输入特征（11 维）
- 块对角批处理逻辑
- mu_nlos 初始化（偏置 = -2.0）— 此设置工作正常
- 三阶段训练结构（BCE → 混合 → NLL）
- NLL 计算中的 p_los.detach() — 保持原样

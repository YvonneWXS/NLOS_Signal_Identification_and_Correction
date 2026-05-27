# train.py (修改版)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
import seaborn as sns
import os
import time

# --- 导入自定义模块 ---
from featureEngineering import SmartLocFeatureEngineer
from datasetGenerator import WindowGenerator, GNSSSlidingWindowDataset
from model import NLOS_CNN_LSTM

# --- 绘图风格设置 ---
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper")

# --- 配置参数 ---
CONFIG = {
    'csv_path': '../dataset/berlin2_gendarmenmarkt/RXM-RAWX_processed.csv',
    'model_save_path': 'nlos_model_best.pth',
    'seq_len': 10,
    'max_gap': 1.5,
    'batch_size': 128,
    'learning_rate': 0.001,
    'epochs': 50,
    'hidden_size': 64,
    'num_filters': 64,
    'test_split_ratio': 0.2,
    'num_chunks': 10,  # 分段切分数量
    'weight_decay': 1e-4  # 新增L2正则化参数
}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = running_loss / len(loader)
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    preds = (all_probs > 0.5).astype(int)
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, preds),
        'f1': f1_score(all_labels, preds, average='binary'),
        'probs': all_probs,
        'labels': all_labels,
        'preds': preds
    }
    return metrics


# --- 分段切分函数（实际调用）---
def split_data_by_chunks(df, time_col, num_chunks=10, val_ratio=0.2):
    """将数据按时间排序后分段切分，保证验证集均匀分布"""
    print(f"正在进行分段切分 (Chunks={num_chunks}, Val_Ratio={val_ratio})...")
    unique_times = np.sort(df[time_col].unique())
    total_len = len(unique_times)
    chunk_size = int(np.ceil(total_len / num_chunks))
    train_times_list = []
    val_times_list = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_len)
        if start_idx >= end_idx:
            break
        chunk_times = unique_times[start_idx:end_idx]
        split_point = int(len(chunk_times) * (1 - val_ratio))
        train_times_list.extend(chunk_times[:split_point])
        val_times_list.extend(chunk_times[split_point:])

    train_time_set = set(train_times_list)
    val_time_set = set(val_times_list)
    print(f"  时间点划分 -> 训练集: {len(train_time_set)}, 验证集: {len(val_time_set)}")

    train_df = df[df[time_col].isin(train_time_set)].copy()
    val_df = df[df[time_col].isin(val_time_set)].copy()
    return train_df, val_df


def plot_training_history(train_losses, val_losses, val_f1s):
    """绘制 Loss 和 F1 曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # Loss
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red', linestyle='--')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    # F1 Score
    ax2.plot(val_f1s, label='Val F1 Score', color='green')
    ax2.set_title('Validation F1 Score ')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('viz_training_history.png')
    print("已保存: viz_training_history.png")


def plot_advanced_metrics(labels, probs):
    """绘制 ROC, PR 曲线和概率分布"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    ax2.plot(recall, precision, color='purple', lw=2, label=f'AP = {pr_auc:.2f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    # 3. Probability Distribution
    sns.histplot(probs[labels == 0], color='blue', label='True LOS', kde=True, ax=ax3, alpha=0.6, stat='density')
    sns.histplot(probs[labels == 1], color='red', label='True NLOS', kde=True, ax=ax3, alpha=0.6, stat='density')
    ax3.set_title('Prediction Probability Distribution')
    ax3.set_xlabel('Predicted Probability (NLOS)')
    ax3.legend()
    plt.tight_layout()
    plt.savefig('viz_advanced_metrics.png')
    print("已保存: viz_advanced_metrics.png")


def plot_confusion_matrix_heatmap(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred LOS', 'Pred NLOS'],
                yticklabels=['True LOS', 'True NLOS'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('viz_confusion_matrix.png')
    print("已保存: viz_confusion_matrix.png")


def calculate_permutation_importance(model, dataset_loader, criterion, device, feature_names):
    """修复特征名称报错，计算特征重要性"""
    print("\n正在计算特征重要性 (Permutation Importance)...")
    model.eval()
    # 1. 获取基准 Loss
    original_metrics = evaluate(model, dataset_loader, criterion, device)
    baseline_loss = original_metrics['loss']
    importances = {}

    # 重建完整数据集
    all_inputs = []
    all_labels = []
    for x, y in dataset_loader:
        all_inputs.append(x)
        all_labels.append(y)
    X_full = torch.cat(all_inputs).cpu()  # [N, Seq, Feat]
    y_full = torch.cat(all_labels).cpu()

    # 2. 逐个特征进行打乱测试（修复特征名称索引）
    num_features = X_full.shape[2]
    for i in range(num_features):
        # 修复：使用传入的 feature_names，避免索引越界
        feat_name = feature_names[i] if i < len(feature_names) else f"Feat_{i}"
        # 复制数据并打乱当前特征
        X_permuted = X_full.clone()
        perm_idx = torch.randperm(X_permuted.size(0))
        X_permuted[:, :, i] = X_permuted[perm_idx, :, i]

        # 重新封装 loader
        perm_dataset = TensorDataset(X_permuted, y_full)
        perm_loader = DataLoader(perm_dataset, batch_size=CONFIG['batch_size'])

        # 评估
        perm_metrics = evaluate(model, perm_loader, criterion, device)
        loss_increase = perm_metrics['loss'] - baseline_loss
        importances[feat_name] = loss_increase
        print(f"  特征 '{feat_name}': Loss 增加 {loss_increase:.6f}")

    # 绘图
    plt.figure(figsize=(10, 6))
    sorted_imps = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_imps]
    values = [x[1] for x in sorted_imps]
    sns.barplot(x=values, y=names, hue=names, palette="viridis", legend=False)
    plt.title("Feature Importance (Permutation Method)")
    plt.xlabel("Increase in Loss (Model degradation)")
    plt.tight_layout()
    plt.savefig('viz_feature_importance.png')
    print("已保存: viz_feature_importance.png")


# --- 主程序 ---
def main():
    start_time = time.time()
    # 1. 设置设备
    if not torch.cuda.is_available():
        print("⚠️ 警告：未检测到可用CUDA设备，将使用CPU训练（速度可能很慢）")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        # 打印CUDA设备详情
        print(f"✅ 使用CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   设备数量: {torch.cuda.device_count()}")
        print(f"   初始内存: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

    # 2. 读取数据
    print(f"\n正在读取数据: {CONFIG['csv_path']} ...")
    if not os.path.exists(CONFIG['csv_path']):
        print("错误: 找不到数据文件！")
        return
    df = pd.read_csv(CONFIG['csv_path'], sep=';')

    # 3. 改进的数据划分（使用分段切分，避免分布不均）
    time_col = 'GPSSecondsOfWeek [s]'
    train_df, test_df = split_data_by_chunks(
        df, time_col,
        num_chunks=CONFIG['num_chunks'],
        val_ratio=CONFIG['test_split_ratio']
    )
    print(f"数据划分完成 -> 训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")

    # 4. 特征工程
    engineer = SmartLocFeatureEngineer()
    engineer.fit(train_df)
    engineer.save_scalers('nlos_scalers.pkl')  # 保存训练集scaler
    train_df_proc = engineer.transform(train_df)
    test_df_proc = engineer.transform(test_df)
    feature_names = engineer.feature_columns  # 获取特征名称列表

    # 5. 生成滑动窗口
    win_gen = WindowGenerator(sequence_length=CONFIG['seq_len'], max_gap_seconds=CONFIG['max_gap'])
    X_train, y_train = win_gen.make_dataset(train_df_proc)
    X_test, y_test = win_gen.make_dataset(test_df_proc)
    print(f"滑动窗口生成完成 -> 训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

    # 6. 构建DataLoader
    train_dataset = GNSSSlidingWindowDataset(X_train, y_train)
    test_dataset = GNSSSlidingWindowDataset(X_test, y_test)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        pin_memory=True,  # 配合CUDA加速
        num_workers=4  # 增加数据加载线程（根据CPU核心数调整）
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    # 7. 初始化模型（添加L2正则化）
    input_dim = X_train.shape[2]
    model = NLOS_CNN_LSTM(
        input_size=input_dim,
        hidden_size=CONFIG['hidden_size'],
        num_filters=CONFIG['num_filters'],
        dropout=0.5  # 增大dropout率，增强正则化
    ).to(device)

    # 8. 类别权重与优化器（添加weight_decay）
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    print(f"类别权重 (pos_weight): {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']  # L2正则化
    )

    # 9. 训练循环
    print("\n--- 开始训练 ---")
    train_losses = []
    val_losses = []
    val_f1s = []
    best_f1 = 0.0

    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        val_f1s.append(val_metrics['f1'])

        epoch_dur = time.time() - epoch_start
        print(f"Epoch [{epoch + 1}/{CONFIG['epochs']}] ({epoch_dur:.1f}s) "
              f"Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        # 保存最佳模型（基于F1分数）
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"  🔄 保存最佳模型 (F1: {best_f1:.4f})")

    print(f"\n训练结束! 总耗时: {(time.time() - start_time) / 60:.1f} min")
    print(f"最佳验证集F1分数: {best_f1:.4f}")

    # 10. 最终评估与可视化
    print("\n加载最佳模型进行详细评估...")
    model.load_state_dict(torch.load(CONFIG['model_save_path']))
    final_metrics = evaluate(model, test_loader, criterion, device)

    print("\n=== 最终测试集报告 ===")
    print(classification_report(
        final_metrics['labels'],
        final_metrics['preds'],
        target_names=['LOS', 'NLOS']
    ))

    # 生成可视化图表
    plot_training_history(train_losses, val_losses, val_f1s)
    plot_confusion_matrix_heatmap(final_metrics['labels'], final_metrics['preds'])
    plot_advanced_metrics(final_metrics['labels'], final_metrics['probs'])

    # 计算特征重要性
    calculate_permutation_importance(model, test_loader, criterion, device, feature_names)

    print("\n✅ 所有任务完成。请查看生成的 .png 图片文件。")


if __name__ == "__main__":
    main()
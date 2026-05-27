import os
import platform
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score, accuracy_score, f1_score)

# 导入自定义模块
from featureEngineering import SmartLocFeatureEngineer
from datasetGenerator import WindowGenerator, GNSSSlidingWindowDataset
from model import NLOS_CNN_LSTM

# 配置
CONFIG = {
    'csv_path': '../dataset/berlin1_potsdamer_platz/RXM-RAWX_processed.csv',
    'model_path': 'nlos_model_best.pth',
    'seq_len': 10,
    'max_gap': 1.5,
    'batch_size': 32,

    'hidden_size': 64,
    'num_filters': 64,

    'output_csv': 'test_predictions.csv'  # 预测结果输出路径
}



# --- 中文绘图配置 ---
def configure_chinese_font():
    system = platform.system()
    if system == 'Windows':
        font_list = ['SimHei', 'Microsoft YaHei']
    elif system == 'Darwin':
        font_list = ['Arial Unicode MS', 'PingFang SC']
    else:
        font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = font_list + plt.rcParams['font.sans-serif']

# 初始化中文显示
configure_chinese_font()

"""
Visualization: Select a segment of data to show prediction probability over time.
"""
def plot_single_satellite_timeline(y_true, y_pred_prob, limit=500):
    """
    Visualization: Select a segment of data to show prediction probability over time.
    """
    plt.figure(figsize=(12, 4))

    # 截取前N个点
    n = min(len(y_true), limit)
    t = range(n)

    # 绘制真实标签背景
    plt.fill_between(t, 0, 1, where=(y_true[:n] == 1), color='red', alpha=0.3, label='True NLOS Region')
    plt.fill_between(t, 0, 1, where=(y_true[:n] == 0), color='green', alpha=0.1, label='True LOS Region')

    # 绘制预测概率曲线
    plt.plot(t, y_pred_prob[:n], color='blue', linewidth=1.5, label='NLOS Probability')

    # 绘制决策阈值线
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold (0.5)')

    plt.title(f'Timeline Prediction Analysis (First {n} Samples)')
    plt.xlabel('Time Steps')
    plt.ylabel('Probability of NLOS')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('test_viz_timeline.png')
    print("Saved: test_viz_timeline.png")

"""Plot ROC, PR Curve, and Confusion Matrix in English"""
def plot_test_metrics(labels, probs, preds):
    """Plot ROC, PR Curve, and Confusion Matrix in English"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 混淆矩阵
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Pred LOS', 'Pred NLOS'],
                yticklabels=['True LOS', 'True NLOS'])
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('Actual Label')
    ax1.set_xlabel('Predicted Label')

    # 2. ROC曲线
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax2.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax2.set_title('ROC Curve')
    ax2.set_xlabel('False Positive Rate (FPR)')
    ax2.set_ylabel('True Positive Rate (TPR)')
    ax2.legend(loc="lower right")

    # 3. 精确率-召回率曲线
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    ax3.plot(recall, precision, color='purple', lw=2, label=f'AP = {pr_auc:.2f}')
    ax3.set_title('Precision-Recall Curve')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig('test_viz_metrics.png')
    print("Saved: test_viz_metrics.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 步骤 1: 读取处理好的测试集数据 ---
    print("\n[1/5] 正在读取处理好的测试集数据...")
    # 指定处理好的测试集文件路径
    processed_test_file = '../dataset/berlin1_potsdamer_platz/RXM-RAWX_processed.csv'
    if not os.path.exists(processed_test_file):
        print(f"错误: 找不到处理好的测试集文件 {processed_test_file}")
        return
    # 直接读取处理好的数据（已包含角度等特征）
    df_full = pd.read_csv(processed_test_file, sep=';')

    # --- 步骤 2: 特征工程 ---
    print("\n[2/5] 正在进行特征工程...")
    engineer = SmartLocFeatureEngineer()
    # 加载训练时保存的scaler参数（重要：保持与训练一致）
    if os.path.exists('nlos_scalers.pkl'):
        engineer.load_scalers('nlos_scalers.pkl')
    else:
        print("警告: 找不到 nlos_scalers.pkl，测试结果可能不准确！(仅供调试)")
        engineer.fit(df_full)  # 仅在调试时使用，正式测试需加载训练好的scaler

    df_processed = engineer.transform(df_full)

    # --- 步骤 3: 生成滑动窗口 ---
    print("\n[3/5] 生成测试数据滑动窗口...")
    win_gen = WindowGenerator(sequence_length=CONFIG['seq_len'], max_gap_seconds=CONFIG['max_gap'])
    X_test, y_test = win_gen.make_dataset(df_processed)

    test_dataset = GNSSSlidingWindowDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # --- 步骤 4: 加载模型与推理 ---
    print("\n[4/5] 加载模型并推理...")
    if not os.path.exists(CONFIG['model_path']):
        print(f"错误: 找不到模型文件 {CONFIG['model_path']}，请先运行 train.py")
        return

    # 自动推断输入特征维度
    input_dim = X_test.shape[2]
    model = NLOS_CNN_LSTM(
        input_size=input_dim,
        hidden_size=CONFIG['hidden_size'],
        num_filters=CONFIG['num_filters']
    ).to(device)

    # 加载权重
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)  # 手动加Sigmoid
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())  # 标签在CPU上

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    all_preds = (all_probs > 0.5).astype(int)

    # --- 步骤 5: 结果输出与可视化 ---
    print("\n[5/5] 生成报告与可视化...")

    # 1. 打印分类报告
    print("\n=== 分类性能报告 ===")
    print(classification_report(all_labels, all_preds, target_names=['LOS (视距)', 'NLOS (非视距)']))

    # 2. 保存结果到CSV
    results_df = pd.DataFrame({
        'True_Label': all_labels,
        'Predicted_Label': all_preds,
        'Probability_NLOS': all_probs
    })
    results_df.to_csv(CONFIG['output_csv'], index=False)
    print(f"预测结果已保存至: {CONFIG['output_csv']}")

    # 3. 绘图
    plot_test_metrics(all_labels, all_probs, all_preds)
    plot_single_satellite_timeline(all_labels, all_probs)

    print("\n✅ 测试流程结束！")

if __name__ == "__main__":
    main()
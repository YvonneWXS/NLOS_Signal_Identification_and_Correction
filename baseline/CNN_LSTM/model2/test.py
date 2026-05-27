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
import shutil

# 导入自定义模块
from featureEngineering import SmartLocFeatureEngineer
from datasetGenerator import WindowGenerator, GNSSSlidingWindowDataset
from model import NLOS_CNN_LSTM

# --- 绘图风格设置 ---
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

# --- 配置 ---
CONFIG = {
    # 1. 基础配置
    'model_path': 'results/nlos_model_balanced.pth',
    'scaler_path': 'results/global_scalers.pkl',
    'test_set_path': 'results/test_set_balanced.pt', 
    'output_dir': 'test_results',
    'log_file': 'test_report.txt', # 新增：日志文件
    
    # 2. 可视化配置
    'viz_csv_path': '../dataset/berlin1_potsdamer_platz/RXM-RAWX_processed.csv', 
    'viz_satellite_id': 'G08', 
    
    # 3. 模型参数
    'seq_len': 10,
    'max_gap': 1.5,
    'hidden_size': 64,
    'num_filters': 64,
    'batch_size': 32
}

if os.path.exists(CONFIG['output_dir']):
    shutil.rmtree(CONFIG['output_dir'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# --- 日志辅助函数 ---
def log_msg(msg, log_path):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

class TestVisualizer:
    @staticmethod
    def plot_confusion_matrix_extended(y_true, y_pred, save_dir):
        cm = confusion_matrix(y_true, y_pred)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

        fig, axs = plt.subplots(1, 2, figsize=(16, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0], cbar=False, annot_kws={"size": 14})
        axs[0].set_title('Confusion Matrix (Counts)')
        axs[0].set_xlabel('Predicted')
        axs[0].set_ylabel('True')
        axs[0].set_xticklabels(['LOS', 'NLOS'])
        axs[0].set_yticklabels(['LOS', 'NLOS'])

        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axs[1], cbar=False, annot_kws={"size": 14})
        axs[1].set_title('Confusion Matrix (Normalized)')
        axs[1].set_xlabel('Predicted')
        axs[1].set_yticklabels(['', '']) 
        axs[1].set_xticklabels(['LOS', 'NLOS'])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '1_confusion_matrix.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_roc_pr_detailed(y_true, y_probs, save_dir):
        fig, axs = plt.subplots(1, 2, figsize=(16, 7))
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        axs[0].plot(fpr, tpr, color='darkorange', lw=3, label=f'AUC = {roc_auc:.4f}')
        axs[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axs[0].set_title('ROC Curve')
        axs[0].set_xlabel('FPR')
        axs[0].set_ylabel('TPR')
        axs[0].legend(loc="lower right")

        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        axs[1].plot(recall, precision, color='purple', lw=3, label=f'AP = {pr_auc:.4f}')
        axs[1].set_title('Precision-Recall Curve')
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')
        axs[1].legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '2_roc_pr_curves.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_confidence_distribution(y_true, y_probs, save_dir):
        plt.figure(figsize=(10, 6))
        plt.hist([y_probs[y_true==0], y_probs[y_true==1]], 
                 bins=20, stacked=True, color=['blue', 'red'], 
                 label=['True LOS', 'True NLOS'], alpha=0.7, edgecolor='black')
        plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold 0.5')
        plt.title('Prediction Confidence Histogram')
        plt.xlabel('Predicted Probability of NLOS')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '3_confidence_histogram.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_timeline_analysis(df_viz, seq_len, save_dir, sat_id):
        plt.figure(figsize=(18, 6))
        times = df_viz['time'].values
        probs = df_viz['prob'].values
        true_labels = df_viz['true'].values
        limit = min(len(times), 1200)
        t_idx = range(limit)
        
        plt.fill_between(t_idx, 0, 1, where=(true_labels[:limit] == 1), 
                         color='red', alpha=0.2, transform=plt.gca().get_xaxis_transform(), label='True NLOS')
        plt.fill_between(t_idx, 0, 1, where=(true_labels[:limit] == 0), 
                         color='green', alpha=0.1, transform=plt.gca().get_xaxis_transform(), label='True LOS')

        plt.plot(t_idx, probs[:limit], color='#2c3e50', linewidth=2, label='NLOS Prob')
        plt.axhline(0.5, color='orange', linestyle='--', label='Threshold')

        plt.title(f'Timeline Analysis: Satellite {sat_id}')
        plt.xlabel('Time Steps')
        plt.ylabel('Probability')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'4_timeline_{sat_id}.png'), dpi=300)
        plt.close()
    
    @staticmethod
    def plot_metrics_summary(metrics, save_dir):
        plt.figure(figsize=(8, 6))
        keys = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        vals = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#f1c40f']
        bars = plt.bar(keys, vals, color=colors, alpha=0.8)
        plt.ylim(0, 1.1)
        plt.title('Overall Model Performance Metrics')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '5_metrics_summary.png'), dpi=300)
        plt.close()

def load_model(device):
    input_dim = 7 
    model = NLOS_CNN_LSTM(
        input_size=input_dim,
        hidden_size=CONFIG['hidden_size'],
        num_filters=CONFIG['num_filters']
    ).to(device)
    
    if os.path.exists(CONFIG['model_path']):
        # weights_only=False 确保可以加载包含numpy array的完整checkpoint
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device, weights_only=False))
        print(f"✅ 模型已加载: {CONFIG['model_path']}")
    else:
        raise FileNotFoundError(f"找不到模型文件: {CONFIG['model_path']}")
    model.eval()
    return model

def run_balanced_evaluation(model, device, log_path):
    """阶段一：使用平衡测试集进行统计评估"""
    log_msg("\n--- 阶段 1: 统计评估 (Balanced Test Set) ---", log_path)
    if not os.path.exists(CONFIG['test_set_path']):
        log_msg(f"❌ 找不到测试集文件 {CONFIG['test_set_path']}，跳过阶段一。", log_path)
        return

    try:
        data = torch.load(CONFIG['test_set_path'], weights_only=False)
    except Exception as e:
        log_msg(f"加载测试集失败: {e}", log_path)
        return

    X_test, y_test = data['X'], data['y']
    test_loader = DataLoader(GNSSSlidingWindowDataset(X_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)
    
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    y_true = np.array(all_labels).flatten()
    y_pred = np.array(all_preds).flatten()
    y_probs = np.array(all_probs).flatten()
    
    report = classification_report(y_true, y_pred, target_names=['LOS', 'NLOS'])
    log_msg(report, log_path)
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, target_names=['LOS', 'NLOS'])
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': report_dict['NLOS']['precision'],
        'recall': report_dict['NLOS']['recall'],
        'f1': f1_score(y_true, y_pred)
    }

    TestVisualizer.plot_confusion_matrix_extended(y_true, y_pred, CONFIG['output_dir'])
    TestVisualizer.plot_roc_pr_detailed(y_true, y_probs, CONFIG['output_dir'])
    TestVisualizer.plot_confidence_distribution(y_true, y_probs, CONFIG['output_dir'])
    TestVisualizer.plot_metrics_summary(metrics, CONFIG['output_dir'])

def run_timeline_visualization(model, device, log_path):
    """阶段二：加载原始CSV，画时间线"""
    log_msg(f"\n--- 阶段 2: 时间线可视化 (Raw CSV) ---", log_path)
    csv_path = CONFIG['viz_csv_path']
    
    if not os.path.exists(csv_path):
        log_msg(f"❌ 找不到CSV文件 {csv_path}，跳过阶段二。", log_path)
        return

    log_msg(f"读取文件: {csv_path}", log_path)
    df = pd.read_csv(csv_path, sep=';')
    
    engineer = SmartLocFeatureEngineer()
    if os.path.exists(CONFIG['scaler_path']):
        engineer.load_scalers(CONFIG['scaler_path'])
    else:
        log_msg("⚠️ 警告: 找不到 Scaler，使用当前数据 Fit", log_path)
        engineer.fit(df)
        
    df_proc = engineer.transform(df)
    sv_col = 'Satellite identifier (svId) []'
    if sv_col not in df_proc.columns:
        log_msg(f"❌ 列名错误: 找不到 {sv_col}", log_path)
        return

    top_sv = df_proc[sv_col].value_counts().idxmax()
    log_msg(f"选取数据量最多的卫星进行绘图: SV_ID = {top_sv}", log_path)
    
    df_sat = df_proc[df_proc[sv_col] == top_sv].copy()
    df_sat = df_sat.sort_values(by='GPSSecondsOfWeek [s]') 
    
    win_gen = WindowGenerator(sequence_length=CONFIG['seq_len'], max_gap_seconds=CONFIG['max_gap'])
    X_seq, y_seq = win_gen.make_dataset(df_sat)
    
    if len(X_seq) == 0:
        log_msg("该卫星生成的窗口数量为0，无法绘图。", log_path)
        return

    loader = DataLoader(GNSSSlidingWindowDataset(X_seq, y_seq), batch_size=CONFIG['batch_size'], shuffle=False)
    probs_list = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            probs_list.extend(probs.cpu().numpy())
            
    probs_seq = np.array(probs_list).flatten()
    preds_seq = (probs_seq > 0.5).astype(int)
    
    viz_data = pd.DataFrame({
        'time': range(len(y_seq)), 
        'true': y_seq.flatten(),
        'prob': probs_seq,
        'pred': preds_seq
    })
    
    TestVisualizer.plot_timeline_analysis(viz_data, CONFIG['seq_len'], CONFIG['output_dir'], sat_id=top_sv)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_path = os.path.join(CONFIG['output_dir'], CONFIG['log_file'])
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== Test Report ===\n")
        f.write(f"Config: {CONFIG}\n\n")

    log_msg(f"使用设备: {device}", log_path)
    
    try:
        model = load_model(device)
        run_balanced_evaluation(model, device, log_path)
        run_timeline_visualization(model, device, log_path)
        log_msg(f"\n✅ 测试流程结束！所有结果已保存至: {CONFIG['output_dir']}/", log_path)
    except Exception as e:
        log_msg(f"\n❌ 发生严重错误: {e}", log_path)

if __name__ == "__main__":
    main()
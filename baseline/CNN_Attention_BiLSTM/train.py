import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             confusion_matrix, classification_report, roc_curve, auc, 
                             precision_recall_curve, average_precision_score)
import os
import time
import shutil

# --- 导入自定义模块 ---
from featureEngineering import SmartLocFeatureEngineer
from datasetGenerator import WindowGenerator, GNSSSlidingWindowDataset, DataProcessor
from model import NLOS_CNN_Attention_LSTM

# --- 绘图风格设置 ---
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False 
except:
    pass

# --- 配置参数 ---
CONFIG = {
    'csv_paths': [
        'D:/3_document/4_research/smartLoc/dataset/berlin1_potsdamer_platz/RXM-RAWX_processed.csv',
        'D:/3_document/4_research/smartLoc/dataset/berlin2_gendarmenmarkt/RXM-RAWX_processed.csv',
        'D:/3_document/4_research/smartLoc/dataset/frankfurt1_maintower/RXM-RAWX_processed.csv',
        'D:/3_document/4_research/smartLoc/dataset/frankfurt2_westendtower/RXM-RAWX_processed.csv',
        ''
    ],
    'output_dir': 'CNN_Attention_BiLSTM/results',
    'model_name': 'nlos_model_balanced.pth',
    'log_file': 'training_log.txt', # 新增：日志文件名
    'seq_len': 10,
    'max_gap': 1.5,
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 2,
    'hidden_size': 64,
    'num_filters': 64,
    'dropout': 0.5,
    'weight_decay': 1e-4
}

# 确保输出目录存在
if os.path.exists(CONFIG['output_dir']):
    shutil.rmtree(CONFIG['output_dir'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# --- 日志辅助函数 ---
def log_msg(msg, log_path):
    """同时打印到控制台和写入文件"""
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

class Visualizer:
    """专门负责绘图的类"""
    
    @staticmethod
    def plot_training_dashboard(history, save_path):
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Loss
        axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color='navy')
        axs[0, 0].plot(epochs, history['val_loss'], label='Val Loss', color='crimson', linestyle='--')
        axs[0, 0].set_title('Loss Curve')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # 2. Accuracy
        axs[0, 1].plot(epochs, history['val_acc'], label='Val Accuracy', color='darkgreen')
        axs[0, 1].set_title('Validation Accuracy')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)

        # 3. F1 Score
        axs[1, 0].plot(epochs, history['val_f1'], label='Val F1 Score', color='darkorange')
        axs[1, 0].set_title('Validation F1 Score')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        # 4. Precision & Recall
        axs[1, 1].plot(epochs, history['val_precision'], label='Precision', color='purple')
        axs[1, 1].plot(epochs, history['val_recall'], label='Recall', color='teal')
        axs[1, 1].set_title('Precision & Recall')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'viz_1_training_dashboard.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_confusion_matrices(y_true, y_pred, save_path):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0], cbar=False,
                    xticklabels=['Pred LOS', 'Pred NLOS'], yticklabels=['True LOS', 'True NLOS'])
        axs[0].set_title('Confusion Matrix (Counts)')
        
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axs[1], cbar=False,
                    xticklabels=['Pred LOS', 'Pred NLOS'], yticklabels=['True LOS', 'True NLOS'])
        axs[1].set_title('Confusion Matrix (Normalized)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'viz_2_confusion_matrices.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_roc_pr_curves(y_true, y_probs, save_path):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        axs[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        axs[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axs[0].set_title('ROC Curve')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        axs[1].plot(recall, precision, color='purple', lw=2, label=f'AP = {pr_auc:.3f}')
        axs[1].set_title('Precision-Recall Curve')
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'viz_3_roc_pr_curves.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_prob_distribution(y_true, y_probs, save_path):
        plt.figure(figsize=(10, 6))
        probs_los = y_probs[y_true == 0]
        probs_nlos = y_probs[y_true == 1]
        
        sns.kdeplot(probs_los, fill=True, color="blue", label="True LOS", alpha=0.3)
        sns.kdeplot(probs_nlos, fill=True, color="red", label="True NLOS", alpha=0.3)
        
        plt.axvline(0.5, color='black', linestyle='--', label='Threshold 0.5')
        plt.title('Prediction Probability Density Distribution')
        plt.xlabel('Predicted Probability of NLOS')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'viz_4_prob_distribution.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_dataset_stats(counts, save_path):
        plt.figure(figsize=(8, 6))
        names = list(counts.keys())
        values = list(counts.values())
        
        bars = plt.bar(names, values, color=['#3498db', '#9b59b6', '#2ecc71'])
        plt.title('Dataset Split Statistics (After Balancing)')
        plt.ylabel('Number of Windows')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'viz_5_data_stats.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_feature_importance(model, loader, criterion, device, feature_names, save_path):
        print("\n正在计算特征重要性 (Permutation Importance)...")
        model.eval()
        baseline_loss = 0
        
        all_x = []
        all_y = []
        for x, y in loader:
            all_x.append(x)
            all_y.append(y)
        
        X_full = torch.cat(all_x).to(device)
        y_full = torch.cat(all_y).to(device)
        
        with torch.no_grad():
            logits = model(X_full)
            baseline_loss = criterion(logits, y_full).item()
            
        importances = {}
        num_features = X_full.shape[2]
        
        for i in range(num_features):
            X_permuted = X_full.clone()
            idx = torch.randperm(X_permuted.size(0))
            X_permuted[:, :, i] = X_permuted[idx, :, i]
            
            with torch.no_grad():
                logits = model(X_permuted)
                loss = criterion(logits, y_full).item()
            
            feat_name = feature_names[i] if i < len(feature_names) else f"F{i}"
            importances[feat_name] = loss - baseline_loss 

        plt.figure(figsize=(10, 6))
        sorted_imps = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_imps]
        values = [x[1] for x in sorted_imps]
        
        # --- 修改开始 ---
        # 1. 去掉了 legend=False 参数，避免报错
        # 2. 获取 ax 对象以便后续手动控制图例
        ax = sns.barplot(x=values, y=names, palette='viridis', hue=names)
        
        # 3. 手动移除自动生成的图例 (如果存在)
        if ax.legend_ is not None:
            ax.legend_.remove()
        # --- 修改结束 ---

        plt.title('Feature Importance (Permutation Method)')
        plt.xlabel('Increase in Loss (Higher is more important)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'viz_6_feature_importance.png'), dpi=300)
        plt.close()


def train_epoch(model, loader, criterion, optimizer, device):
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
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    metrics = {
        'loss': running_loss / len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='binary'),
        'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
        'probs': np.array(all_probs).flatten(),
        'preds': np.array(all_preds).flatten(),
        'labels': np.array(all_labels).flatten()
    }
    return metrics

def load_and_process_data():
    all_dfs = []
    print("[1/4] 正在读取所有数据集文件...")
    for path in CONFIG['csv_paths']:
        if os.path.exists(path):
            print(f"  读取: {os.path.basename(path)}")
            all_dfs.append(pd.read_csv(path, sep=';'))
        else:
            print(f"  ⚠️ 文件不存在: {path}")

    if not all_dfs: raise ValueError("无数据！")

    print("[2/4] 全局特征工程 Fitting...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    engineer = SmartLocFeatureEngineer()
    engineer.fit(full_df)
    engineer.save_scalers(os.path.join(CONFIG['output_dir'], 'global_scalers.pkl'))
    
    feature_names = engineer.feature_columns 
    del full_df

    print("[3/4] 生成滑动窗口并合并...")
    win_gen = WindowGenerator(sequence_length=CONFIG['seq_len'], max_gap_seconds=CONFIG['max_gap'])
    X_pool, y_pool = [], []
    
    for df in all_dfs:
        df_proc = engineer.transform(df)
        X_chunk, y_chunk = win_gen.make_dataset(df_proc)
        if len(X_chunk) > 0:
            X_pool.append(X_chunk)
            y_pool.append(y_chunk)
            
    X_total = np.concatenate(X_pool, axis=0)
    y_total = np.concatenate(y_pool, axis=0)
    print(f"  原始总窗口数: {len(y_total)}")
    
    return X_total, y_total, feature_names

def main():
    start_time = time.time()
    
    # 初始化日志文件
    log_path = os.path.join(CONFIG['output_dir'], CONFIG['log_file'])
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== Training Log ===\n")
        f.write(f"Config: {CONFIG}\n\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_msg(f"使用设备: {device}", log_path)

    # 1. 数据准备
    X_total, y_total, feature_names = load_and_process_data()
    
    # 2. 数据平衡
    log_msg("[4/4] 数据平衡与划分...", log_path)
    X_bal, y_bal = DataProcessor.balance_dataset(X_total, y_total)
    
    # 3. 划分数据集
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = DataProcessor.split_dataset(
        X_bal, y_bal, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20
    )
    
    test_set_path = os.path.join(CONFIG['output_dir'], 'test_set_balanced.pt')
    torch.save({'X': X_test, 'y': y_test}, test_set_path)
    log_msg(f"✅ 测试集已保存至: {test_set_path}", log_path)
    
    Visualizer.plot_dataset_stats({
        'Train': len(y_train), 'Validation': len(y_val), 'Test': len(y_test)
    }, CONFIG['output_dir'])

    # 4. DataLoader
    train_loader = DataLoader(GNSSSlidingWindowDataset(X_train, y_train), 
                            batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(GNSSSlidingWindowDataset(X_val, y_val), 
                            batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(GNSSSlidingWindowDataset(X_test, y_test), 
                           batch_size=CONFIG['batch_size'], shuffle=False)

    # 5. 模型初始化
    model = NLOS_CNN_Attention_LSTM(
        input_size=X_train.shape[2],
        hidden_size=CONFIG['hidden_size'],
        num_filters=CONFIG['num_filters'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

    # 6. 训练循环
    log_msg("\n--- 开始训练 ---", log_path)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
    best_f1 = 0.0
    model_save_path = os.path.join(CONFIG['output_dir'], CONFIG['model_name'])

    for epoch in range(CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        msg = (f"Epoch {epoch+1}/{CONFIG['epochs']} | "
               f"Loss: {train_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        log_msg(msg, log_path)
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), model_save_path)
            log_msg("  🌟 Best Model Saved", log_path)

    # 7. 训练后可视化与评估
    log_msg("\n--- 训练完成，生成可视化报告 ---", log_path)
    
    Visualizer.plot_training_dashboard(history, CONFIG['output_dir'])
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    log_msg("\n=== 最终测试集报告 (Balanced) ===", log_path)
    report = classification_report(test_metrics['labels'], test_metrics['preds'], target_names=['LOS', 'NLOS'])
    log_msg(report, log_path)
    
    Visualizer.plot_confusion_matrices(test_metrics['labels'], test_metrics['preds'], CONFIG['output_dir'])
    Visualizer.plot_roc_pr_curves(test_metrics['labels'], test_metrics['probs'], CONFIG['output_dir'])
    Visualizer.plot_prob_distribution(test_metrics['labels'], test_metrics['probs'], CONFIG['output_dir'])
    Visualizer.plot_feature_importance(model, test_loader, criterion, device, feature_names, CONFIG['output_dir'])
    
    end_msg = f"\n✅ 所有任务完成！结果已保存在: {CONFIG['output_dir']}/\n总耗时: {(time.time() - start_time)/60:.1f} min"
    log_msg(end_msg, log_path)

if __name__ == "__main__":
    main()
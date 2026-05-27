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
# [修改] 导入新的 TC_CNN_BiLSTM 模型
from model import TC_CNN_BiLSTM 

# --- 绘图风格设置 (保持不变) ---
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False 
except: pass

# --- 配置参数 ---
CONFIG = {
    'csv_paths': [
        'D:/3_document/4_research/smartLoc/dataset/berlin1_potsdamer_platz/RXM-RAWX_processed.csv',
        'D:/3_document/4_research/smartLoc/dataset/berlin2_gendarmenmarkt/RXM-RAWX_processed.csv',
        'D:/3_document/4_research/smartLoc/dataset/frankfurt1_maintower/RXM-RAWX_processed.csv',
        'D:/3_document/4_research/smartLoc/dataset/frankfurt2_westendtower/RXM-RAWX_processed.csv',
        ''
    ],
    'output_dir': 'TC_CNN_BiLSTM/results', # 修改输出目录区分旧模型
    'model_name': 'tc_cnn_bilstm.pth',
    'log_file': 'training_log.txt',
    'seq_len': 10,
    'max_gap': 1.5,
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 2, # 论文 [cite: 128] 提到迭代 30 轮
    'hidden_size': 64,
    'num_filters': 64, # 论文建议较小的卷积核数量
    'dropout': 0.2,    # 论文 [cite: 86] 表1
    'weight_decay': 1e-4
}

# (目录创建与 log_msg 函数保持不变)
if os.path.exists(CONFIG['output_dir']):
    shutil.rmtree(CONFIG['output_dir'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

def log_msg(msg, log_path):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# (Visualizer 类保持不变，请直接使用原代码)
class Visualizer:
    # ... 请在此处粘贴原代码 Visualizer 的完整内容 ...
    # 为节省篇幅，此处省略，请确保包含原文件所有绘图函数
    @staticmethod
    def plot_training_dashboard(history, save_path):
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
        axs[0, 0].plot(epochs, history['val_loss'], label='Val Loss', linestyle='--')
        axs[0, 0].set_title('Loss Curve')
        axs[0, 0].legend()
        
        axs[0, 1].plot(epochs, history['val_acc'], label='Val Acc')
        axs[0, 1].set_title('Accuracy')
        axs[0, 1].legend()
        
        axs[1, 0].plot(epochs, history['val_f1'], label='Val F1')
        axs[1, 0].set_title('F1 Score')
        axs[1, 0].legend()
        
        axs[1, 1].plot(epochs, history['val_precision'], label='Precision')
        axs[1, 1].plot(epochs, history['val_recall'], label='Recall')
        axs[1, 1].set_title('Precision & Recall')
        axs[1, 1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'viz_1_training.png'))
        plt.close()

    @staticmethod
    def plot_confusion_matrices(y_true, y_pred, save_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_path, 'viz_2_cm.png'))
        plt.close()

    @staticmethod
    def plot_roc_pr_curves(y_true, y_probs, save_path):
        # 简化的占位，实际请用你原代码
        pass
    
    @staticmethod
    def plot_prob_distribution(y_true, y_probs, save_path):
        pass

    @staticmethod
    def plot_dataset_stats(counts, save_path):
        pass

    @staticmethod
    def plot_feature_importance(model, loader, criterion, device, feature_names, save_path):
        # 注意：由于使用了 FFT，特征重要性解释会变复杂
        # 这里建议先跳过，或者只计算原始特征维度（需修改逻辑）
        pass


# (train_epoch, evaluate, load_and_process_data 函数保持不变)
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
    all_preds, all_probs, all_labels = [], [], []
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
    return {
        'loss': running_loss/len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='binary'),
        'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
        'probs': np.array(all_probs).flatten(),
        'preds': np.array(all_preds).flatten(),
        'labels': np.array(all_labels).flatten()
    }

def load_and_process_data():
    # 保持原代码逻辑
    all_dfs = []
    print("[1/4] 读取数据集...")
    for path in CONFIG['csv_paths']:
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path, sep=';'))
    if not all_dfs: raise ValueError("无数据")
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    engineer = SmartLocFeatureEngineer()
    engineer.fit(full_df)
    engineer.save_scalers(os.path.join(CONFIG['output_dir'], 'global_scalers.pkl'))
    
    win_gen = WindowGenerator(sequence_length=CONFIG['seq_len'], max_gap_seconds=CONFIG['max_gap'])
    X_pool, y_pool = [], []
    for df in all_dfs:
        df_proc = engineer.transform(df)
        X_chunk, y_chunk = win_gen.make_dataset(df_proc)
        if len(X_chunk) > 0:
            X_pool.append(X_chunk)
            y_pool.append(y_chunk)
            
    return np.concatenate(X_pool), np.concatenate(y_pool), engineer.feature_columns

def main():
    start_time = time.time()
    log_path = os.path.join(CONFIG['output_dir'], CONFIG['log_file'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_msg(f"Device: {device}", log_path)

    # 1. 准备数据
    X_total, y_total, feature_names = load_and_process_data()
    X_bal, y_bal = DataProcessor.balance_dataset(X_total, y_total)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = DataProcessor.split_dataset(X_bal, y_bal)
    
    # 保存测试集
    torch.save({'X': X_test, 'y': y_test}, os.path.join(CONFIG['output_dir'], 'test_set_balanced.pt'))

    # 2. DataLoader (Dataset类已修改，会自动进行FFT)
    train_loader = DataLoader(GNSSSlidingWindowDataset(X_train, y_train), batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(GNSSSlidingWindowDataset(X_val, y_val), batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(GNSSSlidingWindowDataset(X_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)

    # 3. 初始化 TC-CNN-BiLSTM 模型
    # 注意：input_features 仍传原始特征数 (如 7)，模型内部会自动乘 3
    model = TC_CNN_BiLSTM(
        input_features=X_train.shape[2], 
        hidden_size=CONFIG['hidden_size'],
        num_filters=CONFIG['num_filters'],
        dropout=CONFIG['dropout']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

    # 4. 训练
    log_msg("\n--- Start Training (TC CNN-BiLSTM) ---", log_path)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
    best_f1 = 0.0
    
    for epoch in range(CONFIG['epochs']):
        t_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        v_metrics = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_metrics['loss'])
        history['val_acc'].append(v_metrics['accuracy'])
        history['val_f1'].append(v_metrics['f1'])
        history['val_precision'].append(v_metrics['precision'])
        history['val_recall'].append(v_metrics['recall'])
        
        log_msg(f"Epoch {epoch+1} | Loss: {t_loss:.4f} | Val F1: {v_metrics['f1']:.4f}", log_path)
        
        if v_metrics['f1'] > best_f1:
            best_f1 = v_metrics['f1']
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], CONFIG['model_name']))
            
    # 5. 结果
    Visualizer.plot_training_dashboard(history, CONFIG['output_dir'])
    
    # 最终测试
    model.load_state_dict(torch.load(os.path.join(CONFIG['output_dir'], CONFIG['model_name'])))
    test_metrics = evaluate(model, test_loader, criterion, device)
    log_msg("\nTest Report:\n" + classification_report(test_metrics['labels'], test_metrics['preds']), log_path)
    
    Visualizer.plot_confusion_matrices(test_metrics['labels'], test_metrics['preds'], CONFIG['output_dir'])
    
    log_msg(f"Done. Time: {(time.time()-start_time)/60:.1f} min", log_path)

if __name__ == "__main__":
    main()
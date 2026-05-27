import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             confusion_matrix, roc_auc_score)
import os
import time
import shutil

# --- 导入自定义模块 ---
from featureEngineering import SmartLocFeatureEngineer
from datasetGenerator import WindowGenerator, DataProcessor
from model import SmartLocRandomForest

# --- 绘图风格 ---
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
    'output_dir': 'Random_Forest/results_random_forest',
    'model_name': 'smartloc_rf.pkl',
    'seq_len': 10,   # 滑动窗口长度
    'max_gap': 1.5,
    
    # 随机森林超参数
    'rf_params': {
        'n_estimators': 150,     # 树的数量
        'max_depth': 20,         # 限制深度防止过拟合
        'min_samples_split': 5,
        'random_state': 42
    }
}

if os.path.exists(CONFIG['output_dir']):
    shutil.rmtree(CONFIG['output_dir'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

class Visualizer:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.savefig(os.path.join(save_path, 'viz_cm.png'))
        plt.close()

    @staticmethod
    def plot_feature_importance(model, feature_names_base, seq_len, save_path):
        """
        绘制随机森林特征重要性 (Top 20)
        需要还原 Flatten 后的特征名称
        """
        # 生成特征名: t-9_CNO, t-8_CNO ... t-0_CNO
        flat_feature_names = []
        for i in range(seq_len):
            # t-0 代表当前时刻, t-9 代表最久远的时刻
            prefix = f"t-{seq_len-1-i}" 
            flat_feature_names.extend([f"{prefix}_{name}" for name in feature_names_base])
        
        importances = model.feature_importances_
        
        # 排序
        indices = np.argsort(importances)[::-1]
        top_n = 20
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=importances[top_indices], y=[flat_feature_names[i] for i in top_indices], palette='viridis')
        plt.title('Random Forest - Top 20 Feature Importances')
        plt.xlabel('Importance Score (Gini Impurity Decrease)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'viz_feature_importance.png'))
        plt.close()

def load_and_process_data():
    all_dfs = []
    print("[1] 读取数据...")
    for path in CONFIG['csv_paths']:
        if os.path.exists(path):
            print(f"  - Loading: {path}")
            all_dfs.append(pd.read_csv(path, sep=';'))
            
    if not all_dfs: raise ValueError("无数据文件")
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # 特征工程
    engineer = SmartLocFeatureEngineer()
    engineer.fit(full_df)
    
    win_gen = WindowGenerator(sequence_length=CONFIG['seq_len'], max_gap_seconds=CONFIG['max_gap'])
    
    X_pool, y_pool = [], []
    for df in all_dfs:
        df_proc = engineer.transform(df)
        X_chunk, y_chunk, feat_cols = win_gen.make_dataset(df_proc)
        if len(X_chunk) > 0:
            X_pool.append(X_chunk)
            y_pool.append(y_chunk)
            
    return np.concatenate(X_pool), np.concatenate(y_pool), feat_cols

def evaluate_model(model, X, y, dataset_name="Test"):
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, probs)
    
    print(f"\n--- {dataset_name} Set Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(classification_report(y, preds))
    
    return preds, probs

def main():
    start_time = time.time()
    
    # 1. 准备数据
    X_total, y_total, feature_names = load_and_process_data()
    X_bal, y_bal = DataProcessor.balance_dataset(X_total, y_total)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = DataProcessor.split_dataset(X_bal, y_bal)
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 保存测试集供 test.py 使用
    np.savez(os.path.join(CONFIG['output_dir'], 'test_set.npz'), X=X_test, y=y_test)
    
    # 2. 训练模型
    model = SmartLocRandomForest(**CONFIG['rf_params'])
    model.fit(X_train, y_train)
    
    # 3. 保存
    model.save(os.path.join(CONFIG['output_dir'], CONFIG['model_name']))
    
    # 4. 评估
    evaluate_model(model, X_val, y_val, "Validation")
    preds_test, _ = evaluate_model(model, X_test, y_test, "Test")
    
    # 5. 可视化
    Visualizer.plot_confusion_matrix(y_test, preds_test, CONFIG['output_dir'])
    Visualizer.plot_feature_importance(model, feature_names, CONFIG['seq_len'], CONFIG['output_dir'])
    
    print(f"Done. Time: {(time.time()-start_time)/60:.1f} min")

if __name__ == "__main__":
    main()
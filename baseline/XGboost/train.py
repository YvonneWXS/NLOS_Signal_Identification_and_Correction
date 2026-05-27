import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             confusion_matrix, classification_report)
import os
import time
import shutil

# --- 导入自定义模块 ---
from featureEngineering import SmartLocFeatureEngineer
from datasetGenerator import WindowGenerator, DataProcessor
from model import SmartLocXGBoost

# --- 绘图风格 ---
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
try:
    plt.rcParams['font.family'] = 'sans-serif' # 避免中文乱码问题
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
    'output_dir': 'XGboost/results_xgboost',
    'model_name': 'smartloc_xgboost.pkl',
    'seq_len': 10,  # 保持滑动窗口，利用时间上下文
    'max_gap': 1.5,
    
    # XGBoost 参数 (可根据论文微调)
    'xgb_params': {
        'n_estimators': 200,    # 树的数量
        'max_depth': 8,         # 树深，稍微深一点以捕捉复杂特征
        'learning_rate': 0.05,  # 较低的学习率配合较多的树
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_gpu': True         # 如果有 GPU 可开启
    }
}

# 清理/创建输出目录
if os.path.exists(CONFIG['output_dir']):
    shutil.rmtree(CONFIG['output_dir'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

class Visualizer:
    @staticmethod
    def plot_training_curve(model, save_path):
        results = model.model.evals_result()
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        
        plt.figure(figsize=(10, 5))
        plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
        plt.plot(x_axis, results['validation_1']['logloss'], label='Val')
        plt.legend()
        plt.ylabel('Log Loss')
        plt.xlabel('Estimators')
        plt.title('XGBoost Training Loss')
        plt.savefig(os.path.join(save_path, 'viz_1_loss_curve.png'))
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_path, 'viz_2_confusion_matrix.png'))
        plt.close()

    @staticmethod
    def plot_feature_importance(model, feature_names_base, seq_len, save_path):
        """
        绘制特征重要性
        由于特征被 Flatten 展开了 (Seq * Feats)，我们需要处理一下名字
        """
        # 生成展开后的特征名
        flat_feature_names = []
        for i in range(seq_len):
            prefix = f"t-{seq_len-1-i}" # t-9, t-8 ... t-0
            flat_feature_names.extend([f"{prefix}_{name}" for name in feature_names_base])
        
        importances = model.feature_importances_
        
        # 创建 DataFrame 方便排序
        df_imp = pd.DataFrame({'feature': flat_feature_names, 'importance': importances})
        df_imp = df_imp.sort_values('importance', ascending=False).head(20) # 只看前20个
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=df_imp, palette='viridis')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'viz_3_feature_importance.png'))
        plt.close()

def load_and_process_data():
    all_dfs = []
    print("[1/4] 读取数据集 CSV 文件...")
    for path in CONFIG['csv_paths']:
        if os.path.exists(path):
            print(f"  - Loading: {path}")
            all_dfs.append(pd.read_csv(path, sep=';'))
        else:
            print(f"  [Warning] File not found: {path}")
            
    if not all_dfs: raise ValueError("没有加载到任何数据！")
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # 特征工程 (标准化等)
    engineer = SmartLocFeatureEngineer()
    engineer.fit(full_df) # 计算均值方差
    engineer.save_scalers(os.path.join(CONFIG['output_dir'], 'global_scalers.pkl'))
    
    # 生成窗口并 Flatten
    win_gen = WindowGenerator(sequence_length=CONFIG['seq_len'], max_gap_seconds=CONFIG['max_gap'])
    
    X_pool = []
    y_pool = []
    
    for df in all_dfs:
        df_proc = engineer.transform(df) # 应用标准化
        X_chunk, y_chunk, feat_cols = win_gen.make_dataset(df_proc)
        if len(X_chunk) > 0:
            X_pool.append(X_chunk)
            y_pool.append(y_chunk)
            
    return np.concatenate(X_pool), np.concatenate(y_pool), feat_cols

def evaluate_model(model, X, y, prefix="Test"):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    print(f"\n{prefix} Set Evaluation:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(classification_report(y, preds))
    return preds, acc, f1

def main():
    start_time = time.time()
    
    # 1. 准备数据
    X_total, y_total, feature_names_base = load_and_process_data()
    
    # 2. 平衡数据 (下采样)
    X_bal, y_bal = DataProcessor.balance_dataset(X_total, y_total)
    
    # 3. 划分数据集 (Train/Val/Test)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = DataProcessor.split_dataset(X_bal, y_bal)
    
    print(f"\n数据集划分完成:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # 保存测试集用于 test.py
    # 使用 numpy 保存而非 torch
    np.savez(os.path.join(CONFIG['output_dir'], 'test_set.npz'), X=X_test, y=y_test)

    # 4. 初始化模型
    model = SmartLocXGBoost(**CONFIG['xgb_params'])

    # 5. 训练
    model.fit(X_train, y_train, X_val, y_val)
    
    # 6. 保存模型
    model_save_path = os.path.join(CONFIG['output_dir'], CONFIG['model_name'])
    model.save(model_save_path)

    # 7. 评估与可视化
    Visualizer.plot_training_curve(model, CONFIG['output_dir'])
    
    # 验证集评估
    evaluate_model(model, X_val, y_val, prefix="Validation")
    
    # 测试集评估
    preds_test, _, _ = evaluate_model(model, X_test, y_test, prefix="Test")
    
    # 混淆矩阵
    Visualizer.plot_confusion_matrix(y_test, preds_test, CONFIG['output_dir'])
    
    # 特征重要性
    Visualizer.plot_feature_importance(model, feature_names_base, CONFIG['seq_len'], CONFIG['output_dir'])
    
    print(f"\nDone. Total Time: {(time.time()-start_time)/60:.1f} min")

if __name__ == "__main__":
    main()
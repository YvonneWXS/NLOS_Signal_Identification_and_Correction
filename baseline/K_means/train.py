import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             confusion_matrix)
from sklearn.decomposition import PCA
import os
import time
import shutil

# --- 导入自定义模块 ---
from featureEngineering import SmartLocFeatureEngineer
from datasetGenerator import WindowGenerator, DataProcessor
from model import SmartLocKMeans

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
    'output_dir': 'K_means/results_kmeans',
    'model_name': 'smartloc_kmeans.pkl',
    'seq_len': 10,
    'max_gap': 1.5,
    'n_clusters': 2  # LOS 和 NLOS 两类
}

if os.path.exists(CONFIG['output_dir']):
    shutil.rmtree(CONFIG['output_dir'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

class Visualizer:
    @staticmethod
    def plot_clusters_pca(model, X, y_true, save_path):
        """
        使用 PCA 将高维数据降维到 2D 并绘制聚类结果
        """
        print("正在绘制聚类可视化 (PCA)...")
        # 随机采样 2000 个点避免绘图过慢
        if len(X) > 2000:
            idx = np.random.choice(len(X), 2000, replace=False)
            X_sample = X[idx]
            y_sample = y_true[idx]
        else:
            X_sample = X
            y_sample = y_true

        # PCA 降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sample)
        
        # 预测簇
        y_pred = model.predict(X_sample)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 图1: 真实标签分布
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_sample, palette='coolwarm', ax=axes[0], alpha=0.6)
        axes[0].set_title('Ground Truth (Labels)')
        axes[0].set_xlabel('PCA Component 1')
        axes[0].set_ylabel('PCA Component 2')
        
        # 图2: K-means 聚类结果
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred, palette='viridis', ax=axes[1], alpha=0.6)
        axes[1].set_title('K-means Clustering Results')
        axes[1].set_xlabel('PCA Component 1')
        
        # 绘制聚类中心 (也需要 PCA 变换)
        centers_pca = pca.transform(model.cluster_centers_)
        axes[1].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X', label='Centroids')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'viz_pca_clusters.png'))
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted (Mapped)')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_path, 'viz_cm.png'))
        plt.close()

def load_and_process_data():
    all_dfs = []
    print("[1] 读取数据...")
    for path in CONFIG['csv_paths']:
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path, sep=';'))
            
    if not all_dfs: raise ValueError("无数据文件")
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # 特征工程 (K-means 对 Scale 极度敏感，这里必须确保数据被标准化)
    engineer = SmartLocFeatureEngineer()
    engineer.fit(full_df)
    
    win_gen = WindowGenerator(sequence_length=CONFIG['seq_len'], max_gap_seconds=CONFIG['max_gap'])
    
    X_pool, y_pool = [], []
    for df in all_dfs:
        df_proc = engineer.transform(df)
        X_chunk, y_chunk, _ = win_gen.make_dataset(df_proc)
        if len(X_chunk) > 0:
            X_pool.append(X_chunk)
            y_pool.append(y_chunk)
            
    return np.concatenate(X_pool), np.concatenate(y_pool)

def main():
    start_time = time.time()
    
    # 1. 准备数据
    X_total, y_total = load_and_process_data()
    X_bal, y_bal = DataProcessor.balance_dataset(X_total, y_total)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = DataProcessor.split_dataset(X_bal, y_bal)
    
    print(f"数据形状: Train {X_train.shape}, Test {X_test.shape}")
    
    # 保存测试集
    np.savez(os.path.join(CONFIG['output_dir'], 'test_set.npz'), X=X_test, y=y_test)
    
    # 2. 训练 K-means
    # 注意：fit 时传入 y_train 仅用于建立 '簇 -> 标签' 的映射，不参与聚类过程
    model = SmartLocKMeans(n_clusters=CONFIG['n_clusters'])
    model.fit(X_train, y_train) 
    
    # 3. 保存
    model.save(os.path.join(CONFIG['output_dir'], CONFIG['model_name']))
    
    # 4. 评估 (Test Set)
    print("\n--- Testing Model ---")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nDetailed Report:\n", classification_report(y_test, y_pred))
    
    # 5. 可视化
    Visualizer.plot_confusion_matrix(y_test, y_pred, CONFIG['output_dir'])
    Visualizer.plot_clusters_pca(model, X_test, y_test, CONFIG['output_dir'])
    
    print(f"Done. Time: {(time.time()-start_time)/60:.1f} min")

if __name__ == "__main__":
    main()
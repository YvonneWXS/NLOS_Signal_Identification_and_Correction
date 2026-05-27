import pandas as pd
import numpy as np
from tqdm import tqdm

class WindowGenerator:
    """
    K-means 版本：生成滑动窗口并展平 (Flatten) 为 2D 向量
    """
    def __init__(self, sequence_length=10, max_gap_seconds=1.5):
        self.seq_len = sequence_length
        self.max_gap = max_gap_seconds 

    def make_dataset(self, df_processed):
        print(f"正在生成滑动窗口数据 (窗口大小={self.seq_len}) 用于 K-means...")
        
        id_cols = ['GNSS identifier (gnssId) []', 'Satellite identifier (svId) []']
        time_col = 'GPSSecondsOfWeek [s]'
        label_col = 'label'
        
        all_cols = df_processed.columns.tolist()
        exclude_cols = id_cols + ['GPSWeek [weeks]', time_col, label_col]
        feature_cols = [c for c in all_cols if c not in exclude_cols]
        
        print(f"特征列 ({len(feature_cols)}): {feature_cols}")

        X_list = []
        y_list = []
        
        grouped = df_processed.groupby(id_cols)
        
        for name, group in tqdm(grouped, desc="处理卫星序列"):
            group = group.sort_values(by=time_col)
            data_feats = group[feature_cols].values
            data_times = group[time_col].values
            data_labels = group[label_col].values
            
            num_records = len(group)
            if num_records < self.seq_len:
                continue

            for i in range(num_records - self.seq_len + 1):
                end_idx = i + self.seq_len
                
                # 时间连续性检查
                window_times = data_times[i: end_idx]
                if np.any(np.diff(window_times) > self.max_gap):
                    continue 
                
                window_X = data_feats[i: end_idx, :] 
                window_y = data_labels[end_idx - 1] # 取窗口最后一个点的标签

                if np.isnan(window_y):
                    continue 
                
                # --- 关键步骤：展平 (Flatten) ---
                # 将 (10, 7) -> (70,)
                # 这样 K-means 计算欧氏距离时会考虑整个时间窗的变化
                window_X_flat = window_X.flatten()

                X_list.append(window_X_flat)
                y_list.append(window_y)

        if len(X_list) == 0:
            raise ValueError("未生成任何有效窗口！")

        X_final = np.array(X_list)
        y_final = np.array(y_list)
        
        print(f"生成完成。矩阵形状 X: {X_final.shape}, y: {y_final.shape}")
        return X_final, y_final, feature_cols

class DataProcessor:
    @staticmethod
    def balance_dataset(X, y):
        """
        K-means 对类别不平衡非常敏感，容易将大类拆分。
        因此强烈建议进行平衡处理。
        """
        print("正在平衡数据集...")
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        
        min_count = min(len(idx_0), len(idx_1))
        
        np.random.seed(42)
        idx_0_sel = np.random.choice(idx_0, min_count, replace=False)
        idx_1_sel = np.random.choice(idx_1, min_count, replace=False)
        
        selected = np.concatenate([idx_0_sel, idx_1_sel])
        np.random.shuffle(selected)
        
        return X[selected], y[selected]

    @staticmethod
    def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        # K-means 只有 Train 和 Test 即可，不需要 Validation 做 Early Stopping
        # 但为了保持代码结构兼容，保留划分
        total = len(y)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        return (X[:train_end], y[:train_end]), (X[train_end:val_end], y[train_end:val_end]), (X[val_end:], y[val_end:])
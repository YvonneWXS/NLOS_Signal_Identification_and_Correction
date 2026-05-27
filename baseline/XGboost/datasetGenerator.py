import pandas as pd
import numpy as np
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from featureEngineering import SmartLocFeatureEngineer 

class GNSSSlidingWindowDataset(Dataset):
    """
    修改版 Dataset：实现论文中的三通道 (Time, FFT-Real, FFT-Imag) 机制
    参考论文 Section 3.2 [cite: 113]
    """

    def __init__(self, features, labels, sequence_length=10):
        self.features = features # Numpy array [N, Seq, Feat]
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 1. 获取原始时域序列 (Original) [Seq, Feat]
        # x_time shape: (10, 7)
        x_time = self.features[idx]
        
        # 2. 进行快速傅里叶变换 (FFT)
        # axis=0 表示对时间维度(Seq)进行变换
        x_fft = np.fft.fft(x_time, axis=0)
        
        # 3. 提取实部和虚部 [cite: 113]
        x_real = x_fft.real
        x_imag = x_fft.imag
        
        # 4. 拼接三通道
        # 我们将不同性质的数据拼接在特征维度 (Axis 1)
        # 结果形状: [Seq, Feat * 3] -> [10, 21]
        x_combined = np.concatenate([x_time, x_real, x_imag], axis=1)
        
        # 转为 Tensor
        return torch.tensor(x_combined, dtype=torch.float32), self.labels[idx]

class WindowGenerator:
    """
    XGBoost 版本：生成滑动窗口并展平 (Flatten) 为 2D 矩阵
    """
    def __init__(self, sequence_length=10, max_gap_seconds=1.5):
        self.seq_len = sequence_length
        self.max_gap = max_gap_seconds 

    def make_dataset(self, df_processed):
        print(f"正在生成滑动窗口数据 (窗口大小={self.seq_len}) 用于 XGBoost...")
        
        # 1. 识别列名
        id_cols = ['GNSS identifier (gnssId) []', 'Satellite identifier (svId) []']
        time_col = 'GPSSecondsOfWeek [s]'
        label_col = 'label'
        
        # 自动识别特征列 (排除ID、时间、标签)
        all_cols = df_processed.columns.tolist()
        exclude_cols = id_cols + ['GPSWeek [weeks]', time_col, label_col]
        feature_cols = [c for c in all_cols if c not in exclude_cols]
        
        print(f"特征列 ({len(feature_cols)}): {feature_cols}")

        X_list = []
        y_list = []
        
        # 按卫星分组处理
        grouped = df_processed.groupby(id_cols)
        
        for name, group in tqdm(grouped, desc="处理卫星序列"):
            # 确保按时间排序
            group = group.sort_values(by=time_col)
            
            data_feats = group[feature_cols].values
            data_times = group[time_col].values
            data_labels = group[label_col].values
            
            num_records = len(group)
            if num_records < self.seq_len:
                continue

            # 滑动窗口生成
            for i in range(num_records - self.seq_len + 1):
                end_idx = i + self.seq_len
                
                # 检查时间连续性 (防止跨越过大的时间中断)
                window_times = data_times[i: end_idx]
                if np.any(np.diff(window_times) > self.max_gap):
                    continue 
                
                # 获取窗口特征 [Seq_Len, Num_Features]
                window_X = data_feats[i: end_idx, :] 
                
                # 获取标签 (取窗口最后一个点的标签)
                window_y = data_labels[end_idx - 1]

                # 排除无标签数据
                if np.isnan(window_y):
                    continue 
                
                # --- 关键修改：展平 (Flatten) ---
                # 将 (10, 7) 变为 (70,)
                window_X_flat = window_X.flatten()

                X_list.append(window_X_flat)
                y_list.append(window_y)

        if len(X_list) == 0:
            raise ValueError("未生成任何有效窗口！请检查数据时间戳或窗口设置。")

        X_final = np.array(X_list)
        y_final = np.array(y_list) # XGBoost 标签通常是一维数组
        
        print(f"生成完成。")
        print(f"输入矩阵形状 X: {X_final.shape} (样本数, 序列长*特征数)")
        print(f"标签形状 y: {y_final.shape}")
        
        return X_final, y_final, feature_cols

class DataProcessor:
    @staticmethod
    def balance_dataset(X, y):
        """
        下采样平衡正负样本
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
        total = len(y)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
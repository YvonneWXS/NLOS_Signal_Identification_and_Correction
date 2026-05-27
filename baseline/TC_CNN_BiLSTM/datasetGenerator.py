import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from featureEngineering import SmartLocFeatureEngineer  # 导入你刚才保存的模块


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
    (保持不变) 负责将 DataFrame 转换为滑动窗口格式的 Numpy 数组
    """
    def __init__(self, sequence_length=10, max_gap_seconds=1.5):
        self.seq_len = sequence_length
        self.max_gap = max_gap_seconds 

    def make_dataset(self, df_processed):
        # --- 为了节省篇幅，此处逻辑保持你原文件不变 ---
        # --- 请保留原文件 make_dataset 的完整逻辑 ---
        print(f"正在生成滑动窗口数据 (窗口大小={self.seq_len})...")
        id_cols = ['GNSS identifier (gnssId) []', 'Satellite identifier (svId) []']
        time_col = 'GPSSecondsOfWeek [s]'
        label_col = 'label'
        
        all_cols = df_processed.columns.tolist()
        exclude_cols = id_cols + ['GPSWeek [weeks]', time_col, label_col]
        feature_cols = [c for c in all_cols if c not in exclude_cols]

        X_list = []
        y_list = []
        grouped = df_processed.groupby(id_cols)
        dropped_windows = 0

        for name, group in tqdm(grouped, desc="处理卫星序列"):
            group = group.sort_values(by=time_col)
            data_feats = group[feature_cols].values
            data_times = group[time_col].values
            data_labels = group[label_col].values
            num_records = len(group)

            if num_records < self.seq_len: continue

            for i in range(num_records - self.seq_len + 1):
                end_idx = i + self.seq_len
                window_times = data_times[i: end_idx]
                if np.any(np.diff(window_times) > self.max_gap):
                    dropped_windows += 1
                    continue 

                window_X = data_feats[i: end_idx, :] 
                window_y = data_labels[end_idx - 1]

                if np.isnan(window_y): continue 

                X_list.append(window_X)
                y_list.append(window_y)

        if len(X_list) == 0:
            raise ValueError("未生成任何有效窗口！")

        X_final = np.array(X_list)
        y_final = np.array(y_list).reshape(-1, 1)
        
        print(f"生成完成。特征矩阵形状: {X_final.shape}")
        return X_final, y_final


# --- 辅助函数：创建 DataLoader ---
def create_dataloaders(X, y, batch_size=64, test_split=0.2):
    """
    将 numpy 数据划分为 Train/Test 并封装为 PyTorch DataLoader
    """
    dataset_size = len(X)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    # 随机打乱 (Shuffle)
    # 注意：时序数据通常不在序列内部打乱，但在样本间打乱是可以的（也是必须的，为了训练稳定）
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    train_dataset = GNSSSlidingWindowDataset(X_train, y_train)
    val_dataset = GNSSSlidingWindowDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")

    return train_loader, val_loader

class DataProcessor:
    # --- 保持原文件不变 ---
    @staticmethod
    def balance_dataset(X, y):
        # (请保留原代码逻辑)
        y_flat = y.flatten()
        idx_0 = np.where(y_flat == 0)[0]
        idx_1 = np.where(y_flat == 1)[0]
        min_count = min(len(idx_0), len(idx_1))
        
        np.random.seed(42)
        idx_0_sel = np.random.choice(idx_0, min_count, replace=False)
        idx_1_sel = np.random.choice(idx_1, min_count, replace=False)
        selected = np.concatenate([idx_0_sel, idx_1_sel])
        np.random.shuffle(selected)
        return X[selected], y[selected]

    @staticmethod
    def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        # (请保留原代码逻辑)
        total = len(y)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        return (X[:train_end], y[:train_end]), (X[train_end:val_end], y[train_end:val_end]), (X[val_end:], y[val_end:])


# --- 主函数：集成测试 ---
if __name__ == "__main__":
    # 1. 读取原始 CSV
    csv_file = 'RXM-RAWX_with_Angle.csv'
    try:
        df = pd.read_csv(csv_file, sep=';')

        # 2. 特征工程处理
        engineer = SmartLocFeatureEngineer()
        engineer.fit(df)
        df_processed = engineer.transform(df)

        # 3. 生成滑动窗口
        # 假设数据采样率是 1Hz (通常 u-blox 是 1Hz 或 5Hz)
        # 我们可以设置 max_gap 稍微大一点，比如 1.5s
        gen = WindowGenerator(sequence_length=10, max_gap_seconds=1.5)
        X, y = gen.make_dataset(df_processed)

        # 4. 创建 DataLoader
        train_loader, val_loader = create_dataloaders(X, y, batch_size=32)

        # 5. 检查一个 Batch
        sample_X, sample_y = next(iter(train_loader))
        print("\n--- DataLoader 测试 ---")
        print(f"Batch X shape: {sample_X.shape}")  # 应为 [32, 10, 7]
        print(f"Batch y shape: {sample_y.shape}")  # 应为 [32, 1]

    except FileNotFoundError:
        print(f"找不到文件 {csv_file}")
    except Exception as e:
        import traceback

        traceback.print_exc()
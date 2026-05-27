import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from featureEngineering import SmartLocFeatureEngineer  # 导入你刚才保存的模块


class GNSSSlidingWindowDataset(Dataset):
    """
    PyTorch Dataset: 将 GNSS 时间序列数据转换为滑动窗口样本
    """

    def __init__(self, features, labels, sequence_length=10):
        """
        Args:
            features (np.array): 形状 [N, F] 的特征矩阵
            labels (np.array): 形状 [N, ] 的标签数组
            sequence_length (int): 窗口大小 (默认10)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        # 标签通常只需要最后一个时刻的，但也可能需要序列标签。
        # 这里我们取窗口最后一个时刻的标签作为 Target。
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 注意：这里仅仅是简单的索引，实际的滑动窗口逻辑在构建数据列表时已经完成
        # 传入的 features 已经是被切分好的 (Samples, Seq_Len, Features) ?
        # 不，为了节省内存，通常做法是：
        # 1. 预先生成好所有合法的 (Start_Index, End_Index) 对
        # 2. getitem 时切片
        # 但由于我们要处理多颗卫星的中断，最好的方式是预先生成好所有样本列表。

        return self.features[idx], self.labels[idx]


class WindowGenerator:
    """
    负责将 DataFrame 转换为滑动窗口格式的 Numpy 数组
    """

    def __init__(self, sequence_length=10, max_gap_seconds=1.5):
        self.seq_len = sequence_length
        self.max_gap = max_gap_seconds  # 允许的最大时间间隔（秒），超过视为中断

    def make_dataset(self, df_processed):
        """
        输入: 经过 Feature Engineering 处理后的 DataFrame
        输出: (X, y)
            X shape: [Total_Samples, Sequence_Length, Num_Features]
            y shape: [Total_Samples, 1]
        """
        print(f"正在生成滑动窗口数据 (窗口大小={self.seq_len})...")

        # 1. 识别 ID 列和时间列
        # feature_engineering.py 保证了这些列存在
        id_cols = ['GNSS identifier (gnssId) []', 'Satellite identifier (svId) []']
        time_col = 'GPSSecondsOfWeek [s]'
        label_col = 'label'

        # 排除 ID 和 Time 列，只保留特征列
        # 获取所有列
        all_cols = df_processed.columns.tolist()
        # 找出特征列 (排除 ID, Time, Label)
        exclude_cols = id_cols + ['GPSWeek [weeks]', time_col, label_col]
        feature_cols = [c for c in all_cols if c not in exclude_cols]

        print(f"使用特征: {feature_cols}")

        X_list = []
        y_list = []

        # 2. 按卫星分组 (GNSS System + SV ID)
        # 使用 groupby 可能会慢，但逻辑最清晰
        grouped = df_processed.groupby(id_cols)

        # 统计丢弃的窗口数量
        total_windows = 0
        dropped_windows = 0

        for name, group in tqdm(grouped, desc="处理卫星序列"):
            # 必须按时间排序
            group = group.sort_values(by=time_col)

            # 提取 numpy 数组加速处理
            data_feats = group[feature_cols].values
            data_times = group[time_col].values
            data_labels = group[label_col].values

            num_records = len(group)

            if num_records < self.seq_len:
                continue

            # 3. 滑动窗口切片
            # 这是一个向量化的滑动窗口实现吗？
            # 考虑到我们需要检查时间连续性，循环是不可避免的，
            # 但可以用 stride_tricks 优化，为了代码可读性，这里用标准循环。

            for i in range(num_records - self.seq_len + 1):
                # 窗口索引范围: [i, i + seq_len)
                end_idx = i + self.seq_len

                # 时间连续性检查
                # 检查窗口内最后时刻 - 第一时刻 是否大致等于 (seq_len - 1) * 采样间隔
                # 或者更严格：检查相邻点间隔是否都 < max_gap
                window_times = data_times[i: end_idx]
                time_diffs = np.diff(window_times)

                if np.any(time_diffs > self.max_gap):
                    dropped_windows += 1
                    continue  # 时间中断，跳过此窗口

                # 提取特征和标签
                window_X = data_feats[i: end_idx, :]  # Shape: [Seq_Len, Features]

                # 标签取窗口最后一个点的标签 (Many-to-One)
                # 也可以取整个序列 (Many-to-Many)，视你的模型输出层而定
                window_y = data_labels[end_idx - 1]

                if np.isnan(window_y):
                    continue  # 标签无效

                X_list.append(window_X)
                y_list.append(window_y)
                total_windows += 1

        if len(X_list) == 0:
            raise ValueError("未生成任何有效窗口！请检查数据量或时间连续性。")

        # 转换为 Numpy 数组
        X_final = np.array(X_list)  # [N, Seq, F]
        y_final = np.array(y_list).reshape(-1, 1)  # [N, 1]

        print(f"生成完成。")
        print(f"有效样本数: {len(X_final)}")
        print(f"因时间中断丢弃的窗口数: {dropped_windows}")
        print(f"特征矩阵形状: {X_final.shape}")

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
    @staticmethod
    def balance_dataset(X, y):
        """
        对数据集进行 1:1 的 LOS/NLOS 平衡
        """
        print(f"正在平衡数据集... 原始数量: {len(y)}")
        y_flat = y.flatten()
        
        # 找到两类的索引
        idx_0 = np.where(y_flat == 0)[0] # LOS
        idx_1 = np.where(y_flat == 1)[0] # NLOS
        
        count_0 = len(idx_0)
        count_1 = len(idx_1)
        print(f"  原始分布 -> LOS: {count_0}, NLOS: {count_1}")
        
        if count_0 == 0 or count_1 == 0:
            raise ValueError("某一类样本数量为0，无法进行平衡！请检查数据源。")
            
        # 确定最小数量
        min_count = min(count_0, count_1)
        
        # 随机下采样
        np.random.seed(42) # 固定种子保证复现
        idx_0_sel = np.random.choice(idx_0, min_count, replace=False)
        idx_1_sel = np.random.choice(idx_1, min_count, replace=False)
        
        # 合并并打乱
        selected_indices = np.concatenate([idx_0_sel, idx_1_sel])
        np.random.shuffle(selected_indices)
        
        X_bal = X[selected_indices]
        y_bal = y[selected_indices]
        
        print(f"  平衡后 -> 总数: {len(y_bal)} (LOS: {min_count}, NLOS: {min_count})")
        return X_bal, y_bal

    @staticmethod
    def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        随机切分训练、验证、测试集
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
        
        total = len(y)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        # 已经是打乱过的，直接切片
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        print(f"数据集划分完成:")
        print(f"  Train: {len(y_train)}")
        print(f"  Val:   {len(y_val)}")
        print(f"  Test:  {len(y_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


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
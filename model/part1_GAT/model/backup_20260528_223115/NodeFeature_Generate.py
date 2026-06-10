"""
节点特征生成模块
======================================
功能: 从 EpochData 提取 11 维固定节点特征
输入: EpochData 对象 (单历元)
输出: (N, 11) numpy 特征矩阵

硬约束: 特征维度固定为 11，使用显式列索引构建，禁止动态扩展。
        禁止将 NLOS label 作为特征列，防止数据泄漏。

特征表:
  0: elevation / 90.0
  1: azimuth / 360.0
  2: cno / 60.0
  3: prStdev / 5.0
  4: prMes / 3e7
  5: prInnovation / 100.0  (pseudorange_error in km)
  6: cos(elevation rad)
  7: GPS one-hot
  8: Glonass one-hot
  9: Galileo one-hot
  10: BeiDou one-hot
"""

import numpy as np
from typing import List, Tuple

# 硬约束: 固定 11 维
FEATURE_DIM = 11

# GNSS 星座 one-hot 映射
_GNSS_TO_INDEX = {'GPS': 7, 'Glonass': 8, 'Galileo': 9, 'BeiDou': 10}


def extract_node_features(epoch_data) -> np.ndarray:
    """
    从单历元 EpochData 提取 (N, 11) 节点特征矩阵

    Args:
        epoch_data: EpochData 对象，含 observations 列表

    Returns:
        feature_matrix: (N, 11) numpy 数组
    """
    N = len(epoch_data.observations)
    features = np.zeros((N, FEATURE_DIM), dtype=np.float32)

    for i, obs in enumerate(epoch_data.observations):
        # 特征 0: 仰角 (归一化 ÷ 90°)
        features[i, 0] = obs.elevation / 90.0

        # 特征 1: 方位角 (归一化 ÷ 360°)
        features[i, 1] = obs.azimuth / 360.0

        # 特征 2: 载噪比 (归一化 ÷ 60 dBHz)
        features[i, 2] = obs.cno / 60.0

        # 特征 3: 伪距标准差 (归一化 ÷ 5 m)
        features[i, 3] = obs.pr_stdev / 5.0

        # 特征 4: 伪距测量值 (归一化 ÷ 3e7, ~20000 km 量级)
        features[i, 4] = obs.pr_mes / 3e7

        # 特征 5: 伪距创新量 (归一化 ÷ 100 km, 已去均值的伪距误差)
        features[i, 5] = obs.pseudorange_error / 100.0

        # 特征 6: cos(仰角) — 几何精度指标
        features[i, 6] = np.cos(np.radians(obs.elevation))

        # 特征 7-10: GNSS 星座 one-hot (互斥)
        # 注意: 只有匹配的星座设置为 1.0，其余保持 0.0
        col_idx = _GNSS_TO_INDEX.get(obs.gnss_id, -1)
        if 7 <= col_idx <= 10:
            features[i, col_idx] = 1.0
        # 如果 gnss_id 不在四个主要星座中，one-hot 全为 0

    # 硬约束检查
    assert features.shape[1] == FEATURE_DIM, \
        f"FEATURE DIMENSION MISMATCH: expected {FEATURE_DIM}, got {features.shape[1]}"

    return features


def extract_labels(epoch_data) -> np.ndarray:
    """
    从单历元 EpochData 提取 NLOS 标签

    Args:
        epoch_data: EpochData 对象

    Returns:
        labels: (N,) numpy 数组 (0=LOS, 1=NLOS)
    """
    labels = np.array([obs.nlos_label for obs in epoch_data.observations],
                      dtype=np.float32)
    return labels


def extract_pseudorange_errors(epoch_data) -> np.ndarray:
    """
    从单历元 EpochData 提取去均值后的伪距误差 (km)

    Args:
        epoch_data: EpochData 对象

    Returns:
        errors: (N,) numpy 数组
    """
    errors = np.array([obs.pseudorange_error for obs in epoch_data.observations],
                      dtype=np.float32)
    return errors


def build_epoch_tensors(epoch_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建单历元的完整张量组: (features, errors, labels)

    Args:
        epoch_data: EpochData 对象

    Returns:
        (feature_matrix, pseudorange_errors, nlos_labels)
        - feature_matrix: (N, 11)
        - pseudorange_errors: (N,)
        - nlos_labels: (N,)
    """
    features = extract_node_features(epoch_data)
    errors = extract_pseudorange_errors(epoch_data)
    labels = extract_labels(epoch_data)
    return features, errors, labels


# ==================== 测试入口 ====================

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from Data_read import load_and_process_dataset, print_data_statistics
    from config import get_config

    config = get_config()
    dataset = config.DATASETS[0]
    print(f"Loading {dataset}...")

    epochs = load_and_process_dataset(dataset, config)
    if epochs:
        # 测试第一个历元
        ep = epochs[0]
        features = extract_node_features(ep)
        labels = extract_labels(ep)
        errors = extract_pseudorange_errors(ep)

        print(f"\nFirst epoch ({ep.gps_week}:{ep.gps_seconds:.3f}):")
        print(f"  Features shape: {features.shape}")
        assert features.shape[1] == FEATURE_DIM, "FEATURE DIMENSION CHECK FAILED!"
        print(f"  Labels shape:   {labels.shape}")
        print(f"  Errors shape:   {errors.shape}")
        print(f"  Feature ranges:")
        for j in range(FEATURE_DIM):
            col = features[:, j]
            print(f"    dim {j}: [{col.min():.4f}, {col.max():.4f}], mean={col.mean():.4f}")

        # 验证没有 NLOS label 泄漏到特征中
        print(f"\n  NLOS label check: {labels[:5]} (first 5 labels)")
        print(f"  Feature dim 5 (prInnovation) check: {features[:5, 5]}")
        print(f"  One-hot cols check: GPS count={features[:, 7].sum():.0f}, "
              f"Glonass={features[:, 8].sum():.0f}, "
              f"Galileo={features[:, 9].sum():.0f}, "
              f"BeiDou={features[:, 10].sum():.0f}")

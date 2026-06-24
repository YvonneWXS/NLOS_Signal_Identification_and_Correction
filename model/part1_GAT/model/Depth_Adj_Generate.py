"""
图构建模块
======================================
功能: 基于方位角相近度构建邻接矩阵（图结构边）
输入: EpochData 对象（取 azimuth 字段）
输出: edge_index (2, E), edge_attr (E, 1)

边构建规则:
  对每对卫星 (i, j):
    az_diff = |az_i - az_j|
    if az_diff > 180: az_diff = 360 - az_diff   # 360 环绕
    if az_diff < AZIMUTH_THRESHOLD:              # 默认 90
        添加双向边 (i j) 和 (j i)
        边权重: az_diff / threshold              # [0, 1]，越小越相关

Fallback:
  无有效边时，为所有节点添加自环 (i i)，防止 GAT 平均池化导致梯度消失
"""

import numpy as np
from typing import Tuple

# 从 config 导入阈值
from config import Config, get_config


def build_azimuth_graph(epoch_data,
                        azimuth_threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建基于方位角相近度的图结构

    Args:
        epoch_data: EpochData 对象，含 observations 列表
        azimuth_threshold: 方位角差阈值 (度)，默认从 config 读取

    Returns:
        edge_index: (2, E) numpy 数组，int64 类型，列 = (src, dst)
        edge_attr:  (E, 1) numpy 数组，float32 类型，值 ∈ [0, 1]
    """
    if azimuth_threshold is None:
        cfg = get_config()
        azimuth_threshold = cfg.AZIMUTH_THRESHOLD

    N = len(epoch_data.observations)

    # 提取方位角列表
    azimuths = np.array([obs.azimuth for obs in epoch_data.observations],
                        dtype=np.float32)

    edge_src = []
    edge_dst = []
    edge_weights = []

    for i in range(N):
        for j in range(i + 1, N):
            az_diff = abs(float(azimuths[i]) - float(azimuths[j]))
            if az_diff > 180.0:
                az_diff = 360.0 - az_diff

            if az_diff < azimuth_threshold:
                weight = az_diff / azimuth_threshold  # [0, 1]
                # 双向边
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                edge_weights.extend([weight, weight])

    if edge_src:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_attr = np.array(edge_weights, dtype=np.float32).reshape(-1, 1)
    else:
        # Fallback: 无有效边时添加自环
        edge_index = np.array([list(range(N)), list(range(N))], dtype=np.int64)
        edge_attr = np.ones((N, 1), dtype=np.float32)

    return edge_index, edge_attr


# ==================== 测试入口 ====================

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from Data_read import load_and_process_dataset
    from config import get_config

    config = get_config()
    dataset = config.DATASETS[0]
    print(f"Loading {dataset}...")

    epochs = load_and_process_dataset(dataset, config)
    if epochs:
        ep = epochs[0]
        edge_index, edge_attr = build_azimuth_graph(ep)

        print(f"\nFirst epoch: {len(ep.observations)} satellites")
        print(f"  edge_index shape: {edge_index.shape}")
        print(f"  edge_attr shape:  {edge_attr.shape}")
        print(f"  num_edges:        {edge_index.shape[1]}")

        if edge_index.size > 0:
            print(f"  edge_index min/max: {edge_index.min()}/{edge_index.max()}")
            print(f"  edge_attr range:   [{edge_attr.min():.3f}, {edge_attr.max():.3f}]")

        # 测试几个不同卫星数量的 epoch
        print("\nEdge statistics across epochs:")
        for idx in [0, 10, 50, 100, 500]:
            if idx < len(epochs):
                ep = epochs[idx]
                ei, ea = build_azimuth_graph(ep)
                n = len(ep.observations)
                e = ei.shape[1]
                is_self_loop = (ei[0] == ei[1]).all()
                print(f"  epoch {idx}: {n} sats, {e} edges"
                      f"{' (self-loops)' if is_self_loop else ''}")

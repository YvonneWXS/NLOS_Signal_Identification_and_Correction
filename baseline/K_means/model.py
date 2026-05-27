from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import os
import numpy as np
from scipy.stats import mode

class SmartLocKMeans(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters=2, max_iter=300, random_state=42):
        """
        K-means 分类器封装
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters, 
            max_iter=max_iter, 
            random_state=random_state, 
            n_init=10 # 运行10次取最优
        )
        self.cluster_to_label_map = {} # 存储 {Cluster_ID: True_Label}

    def fit(self, X, y=None):
        """
        训练步骤：
        1. 执行无监督聚类
        2. 利用少量标签信息确定每个簇代表的真实类别 (LOS/NLOS)
        """
        print(f"\n=== 开始 K-means 聚类 (k={self.n_clusters}) ===")
        self.model.fit(X)
        
        # --- 簇映射 (Cluster Mapping) ---
        # K-means 自身不知道 0 是 LOS 还是 NLOS
        # 我们统计落入每个簇的样本的真实标签，采用"多数投票"原则定义簇的性质
        if y is not None:
            train_clusters = self.model.predict(X)
            print("正在计算簇与标签的映射关系...")
            for i in range(self.n_clusters):
                # 找到属于簇 i 的所有样本的真实标签
                indices = np.where(train_clusters == i)[0]
                if len(indices) > 0:
                    true_labels = y[indices]
                    # 众数作为该簇的标签
                    majority_label = mode(true_labels, keepdims=True).mode[0]
                    self.cluster_to_label_map[i] = majority_label
                    print(f"  Cluster {i} -> Label {int(majority_label)} (样本数: {len(indices)})")
                else:
                    self.cluster_to_label_map[i] = 0 # 默认 fallback
        else:
            print("警告: 未提供 y 标签，无法建立映射。预测结果将仅为簇 ID。")
            # 默认直接映射
            for i in range(self.n_clusters):
                self.cluster_to_label_map[i] = i
        
        print("=== 训练完成 ===")
        return self

    def predict(self, X):
        # 1. 预测簇 ID
        clusters = self.model.predict(X)
        
        # 2. 将簇 ID 映射回真实标签 (LOS/NLOS)
        predictions = np.array([self.cluster_to_label_map[c] for c in clusters])
        return predictions

    def predict_proba(self, X):
        # K-means 没有直接的概率，但可以用距离中心点的距离的倒数来模拟置信度
        # 这里为了兼容性，简单返回硬分类的 One-hot
        preds = self.predict(X)
        return np.vstack([1-preds, preds]).T

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 保存整个对象 (包含 model 和 mapping)
        joblib.dump(self, path)
        print(f"模型已保存至: {path}")

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件未找到: {path}")
        loaded_obj = joblib.load(path)
        self.model = loaded_obj.model
        self.cluster_to_label_map = loaded_obj.cluster_to_label_map
        print(f"模型已加载: {path}")
        print(f"加载的映射关系: {self.cluster_to_label_map}")

    @property
    def cluster_centers_(self):
        return self.model.cluster_centers_
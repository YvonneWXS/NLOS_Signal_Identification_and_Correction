from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

class SmartLocRandomForest:
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=None, 
                 min_samples_split=2, 
                 min_samples_leaf=1, 
                 max_features='sqrt',
                 random_state=42):
        """
        初始化随机森林模型
        参数参考: 
        - n_estimators: 树的数量 (论文中通常设置在100左右以平衡性能和速度)
        - max_depth: 树的最大深度，None表示不限制直到纯度足够
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,  # 使用所有CPU核心并行训练
            verbose=1
        )

    def fit(self, X_train, y_train):
        print(f"\n=== 开始训练随机森林 (n_estimators={self.model.n_estimators}) ===")
        self.model.fit(X_train, y_train)
        print("=== 训练完成 ===")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # 返回属于正类 (NLOS/Multipath) 的概率
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"模型已保存至: {path}")

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件未找到: {path}")
        self.model = joblib.load(path)
        print(f"模型已加载: {path}")

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
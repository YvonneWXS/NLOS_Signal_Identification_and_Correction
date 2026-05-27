import xgboost as xgb
import joblib
import os

class SmartLocXGBoost:
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=6, 
                 learning_rate=0.1, 
                 subsample=0.8, 
                 colsample_bytree=0.8,
                 use_gpu=False):
        """
        初始化 XGBoost 分类器
        参数参考论文中常见的树模型设置
        """
        # 设置设备
        tree_method = 'hist'
        device = 'cpu'
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                    print("XGBoost 将使用 GPU 加速")
            except:
                pass

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,   # 树的数量
            max_depth=max_depth,         # 树的深度
            learning_rate=learning_rate, # 学习率 (eta)
            subsample=subsample,         # 样本采样率 (防止过拟合)
            colsample_bytree=colsample_bytree, # 特征采样率
            objective='binary:logistic', # 二分类逻辑回归
            eval_metric=['logloss', 'error'],
            tree_method=tree_method,
            device=device,
            random_state=42,
            n_jobs=-1 # 使用所有CPU核心
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练模型
        """
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        print("\n=== 开始 XGBoost 训练 ===")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=10  # 每10轮打印一次日志
        )
        print("=== 训练结束 ===")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1] # 返回正类 (NLOS) 的概率

    def save(self, path):
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 使用 joblib 保存 sklearn 风格的模型
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
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from model import SmartLocKMeans

CONFIG = {
    'output_dir': 'K_means/results_kmeans',
    'model_name': 'smartloc_kmeans.pkl',
    'test_data': 'test_set.npz'
}

def main():
    print("=== 开始测试 (K-means Mode) ===")
    
    # 1. 加载模型
    model_path = os.path.join(CONFIG['output_dir'], CONFIG['model_name'])
    # 初始化一个空对象，通过 load 填充
    model = SmartLocKMeans()
    try:
        model.load(model_path)
    except Exception as e:
        print(f"无法加载模型: {e}")
        return

    # 2. 加载数据
    data_path = os.path.join(CONFIG['output_dir'], CONFIG['test_data'])
    if not os.path.exists(data_path):
        print("测试数据不存在")
        return
    
    data = np.load(data_path)
    X_test = data['X']
    y_test = data['y']
    print(f"测试数据加载完成: {X_test.shape}")

    # 3. 预测
    # 这里的 predict 已经包含了从 Cluster ID 到 Label 的映射
    y_pred = model.predict(X_test)

    # 4. 报告
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. 混淆矩阵
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.xlabel('Predicted (Mapped)')
    plt.ylabel('True')
    plt.title('K-means Classification Results')
    plt.savefig(os.path.join(CONFIG['output_dir'], 'test_final_cm.png'))
    print("混淆矩阵已保存")

if __name__ == "__main__":
    main()
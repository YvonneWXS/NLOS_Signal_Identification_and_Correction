import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from model import SmartLocRandomForest

CONFIG = {
    'output_dir': 'Random_Forest/results_random_forest',
    'model_name': 'smartloc_rf.pkl',
    'test_data': 'test_set.npz'
}

def main():
    print("=== 开始测试 (Random Forest Mode) ===")
    
    # 1. 加载模型
    model_path = os.path.join(CONFIG['output_dir'], CONFIG['model_name'])
    model = SmartLocRandomForest()
    try:
        model.load(model_path)
    except Exception as e:
        print(f"无法加载模型: {e}")
        return

    # 2. 加载数据
    data_path = os.path.join(CONFIG['output_dir'], CONFIG['test_data'])
    if not os.path.exists(data_path):
        print("测试数据不存在，请先运行 train.py")
        return
    
    data = np.load(data_path)
    X_test = data['X']
    y_test = data['y']
    print(f"测试数据加载完成: {X_test.shape}")

    # 3. 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # 4. 报告
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. 绘图 - ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(CONFIG['output_dir'], 'test_roc.png'))
    print("ROC 曲线已保存")

if __name__ == "__main__":
    main()
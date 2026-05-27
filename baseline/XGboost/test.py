import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from model import SmartLocXGBoost

# 配置
CONFIG = {
    'output_dir': 'XGBoost/results_xgboost',
    'model_name': 'smartloc_xgboost.pkl',
    'test_data': 'test_set.npz'
}

def main():
    print("=== 开始测试 (XGBoost Mode) ===")
    
    # 1. 加载模型
    model_path = os.path.join(CONFIG['output_dir'], CONFIG['model_name'])
    model = SmartLocXGBoost() # 初始化 wrapper
    try:
        model.load(model_path) # 加载权重
    except Exception as e:
        print(f"错误: 无法加载模型. {e}")
        return

    # 2. 加载测试数据
    data_path = os.path.join(CONFIG['output_dir'], CONFIG['test_data'])
    if not os.path.exists(data_path):
        print("错误: 测试数据文件未找到。请先运行 train.py")
        return
        
    data = np.load(data_path)
    X_test = data['X']
    y_test = data['y']
    print(f"测试集已加载: {X_test.shape}")

    # 3. 推理
    print("正在进行推理...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) # 获取概率用于 ROC 曲线

    # 4. 打印报告
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # 5. 绘制混淆矩阵
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Test Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    save_cm = os.path.join(CONFIG['output_dir'], 'test_confusion_matrix.png')
    plt.savefig(save_cm)
    print(f"混淆矩阵已保存: {save_cm}")

    # 6. 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    save_roc = os.path.join(CONFIG['output_dir'], 'test_roc_curve.png')
    plt.savefig(save_roc)
    print(f"ROC 曲线已保存: {save_roc}")

if __name__ == "__main__":
    main()
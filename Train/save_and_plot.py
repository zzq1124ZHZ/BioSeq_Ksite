import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve,matthews_corrcoef
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# def save_and_plot(arr_probs, arr_labels):
#     # 保存概率和标签到CSV
#     data = pd.DataFrame({
#         'Probabilities': arr_probs,
#         'Labels': arr_labels
#     })
#     data.to_csv('Stack.csv', index=False)
#
#     # 计算ROC曲线数据
#     fpr, tpr, _ = roc_curve(arr_labels, arr_probs)
#     roc_auc = auc(fpr, tpr)
#
#     # 计算精确率-召回率曲线数据
#     precision, recall, _ = precision_recall_curve(arr_labels, arr_probs)
#     pr_auc = auc(recall, precision)
#
#     # 绘制ROC曲线
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#
#     # 绘制精确率-召回率曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall curve')
#     plt.legend(loc="lower left")
#
#     plt.tight_layout()
#     plt.savefig('output_xgb_cat.png')  # 保存图形到文件
#     plt.show()
#     return data

def save_and_plot(arr_probs, arr_labels):
    # 保存概率和标签到CSV
    data = pd.DataFrame({
        'Probabilities': arr_probs,
        'Labels': arr_labels
    })
    data.to_csv('Stack.csv', index=False)
    # 计算ROC曲线数据
    fpr, tpr, _ = roc_curve(arr_labels, arr_probs)
    roc_auc = auc(fpr, tpr)
    # print(fpr, tpr)
    # 计算精确率-召回率曲线数据
    precision, recall, _ = precision_recall_curve(arr_labels, arr_probs)
    pr_auc = auc(recall, precision)
    print('-----------------------------')
    print(roc_auc)
    print('-----------------------------')
    # 绘制ROC曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')  # AUC小数点后3位
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # 绘制精确率-召回率曲线
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')  # AUC小数点后3位
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('output_xgb_cat.png')  # 保存图形到文件
    plt.show()
    return data

import numpy as np

def precision_recall_curve(y_true, y_scores, thresholds):
    """ 计算不同阈值下的精度和召回率 """
    precisions = []
    recalls = []
    y_true = np.array(y_true)
    # 在每个阈值下计算precision和recall
    for threshold in thresholds:
        # 根据阈值将预测转换为二值（0或1）
        y_pred = (y_scores >= threshold).astype(int)
        # print(y_pred,type(y_pred))
        # print(y_true,type(y_true))
        # print((y_pred == 1) & (y_true == 1))
        # print(threshold)
        # 计算TP、FP、FN
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        # print(TP,FP,FN)
        # 精度和召回率
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # print(precision)
        # print(recall)
        # exit(-1)
        precisions.append(precision)
        recalls.append(recall)
    
    return precisions, recalls

def average_precision(precisions, recalls):
    """ 根据精度和召回率计算AP（通过计算PR曲线下面积） """
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # 将召回率排序，并插值使得曲线变得平滑
    indices = np.argsort(recalls)
    recalls = recalls[indices]
    precisions = precisions[indices]
    print("*"*20)
    print(precisions)
    print(recalls)
    print("*"*20)
    # 计算召回率差值，再乘以相应的精度值，得到PR曲线下的面积
    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    
    return ap

def calculate_mAP(y_true_all, y_scores_all, thresholds):
    """ 计算mAP：对所有类别的AP取平均 """
    APs = []
    for y_true, y_scores in zip(y_true_all, y_scores_all):
        print(y_true,y_scores)
        precisions, recalls = precision_recall_curve(y_true, y_scores, thresholds)
        print(precisions,recalls)
        print("-"*20)
        ap = average_precision(precisions, recalls)
        APs.append(ap)
    
    # 计算所有类别的平均AP
    mAP = np.mean(APs)
    return mAP

# 示例使用
y_true_all = [
    [1, 0, 1, 1, 0, 1],  # 类别1的真实标签
    [0, 1, 1, 0, 0, 1]   # 类别2的真实标签
]
y_scores_all = [
    [0.9, 0.4, 0.8, 0.7, 0.2, 0.6],  # 类别1的预测分数
    [0.3, 0.8, 0.5, 0.2, 0.1, 0.9]   # 类别2的预测分数
]

thresholds = np.linspace(0, 1, 100)
print(thresholds,thresholds.shape)

# 计算 mAP
mAP = calculate_mAP(y_true_all, y_scores_all, thresholds)
print(f"mAP: {mAP}")

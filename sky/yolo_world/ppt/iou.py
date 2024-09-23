import numpy as np

# 计算IoU的函数
def compute_iou(box1, box2):
    """
    计算两个矩形框之间的IoU
    box1和box2的格式：[x_min, y_min, x_max, y_max]
    """
    # 计算交集
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # 计算每个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# 计算AP的函数
def compute_ap(precisions, recalls):
    """ 通过精度和召回率曲线计算AP """
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    print(precisions,recalls)
    # 对召回率进行排序
    indices = np.argsort(recalls)
    print(indices)
    precisions = precisions[indices]
    recalls = recalls[indices]

    # 计算PR曲线下面积
    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return ap

# 计算mAP的函数
def compute_map(true_boxes, pred_boxes, iou_threshold=0.5):
    """
    计算mAP
    true_boxes和pred_boxes的格式：[类别, x_min, y_min, x_max, y_max]
    iou_threshold：IoU阈值，用于判断是否为有效检测
    """
    num_classes = len(set([box[0] for box in true_boxes]))
    print(f"num_classes {num_classes}")
    aps = []

    for c in range(num_classes):
        # 提取类别为c的真实框和预测框
        true_boxes_c = [box[1:] for box in true_boxes if box[0] == c]
        pred_boxes_c = [box[1:] for box in pred_boxes if box[0] == c]

        # 计算每个预测框的IoU，并判断是否为TP或FP
        tp = []
        fp = []
        for pred_box in pred_boxes_c:
            ious = [compute_iou(pred_box, true_box) for true_box in true_boxes_c]
            print(ious)
            max_iou = max(ious) if ious else 0

            if max_iou >= iou_threshold:
                tp.append(1)  # True Positive
            else:
                fp.append(1)  # False Positive

        # 计算精度和召回率
        num_true_boxes = len(true_boxes_c)
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / num_true_boxes if num_true_boxes > 0 else np.zeros(len(tp))
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # 计算类别c的AP
        ap = compute_ap(precisions, recalls)
        aps.append(ap)

    # 计算所有类别的mAP
    mAP = np.mean(aps)
    return mAP

# 示例数据：真实框和预测框（格式：[类别, x_min, y_min, x_max, y_max]）
true_boxes = [
    [0, 50, 50, 150, 150],  # 类别0的真实框
    [1, 30, 30, 120, 120]   # 类别1的真实框
]

pred_boxes = [
    [0, 55, 55, 145, 145],  # 类别0的预测框
    [1, 25, 25, 115, 115],  # 类别1的预测框
    [1, 200, 200, 300, 300] # 类别1的误报框
]

# 计算mAP
mAP_value = compute_map(true_boxes, pred_boxes, iou_threshold=0.5)
print(f"mAP: {mAP_value}")

# 没有使用全连接，使用的是卷积，没有全局平局池化再1*1卷积，而是使用的对每个像素做预测

# 目标检测fine-tune是对目标分类进行fine-tune，一般不直接对目标检测网络fine-tune

import torch
from d2l import torch as d2l

torch.set_printoptions(2)  # 精简输出精度


def multibox_prior(data, sizes, ratios):
    """多框先验框生成器

    Args:
        data (torch.tensor): 特征图,N,C,H,W
        sizes (list): 缩放比
        ratios (list): 宽高比

    Returns:
        torch.tensor: 返回一个形状为 (1, num_anchors, 4) 的张量，其中 num_anchors 是生成的锚框总数，4 代表锚框的四个坐标 (xmin, ymin, xmax, ymax)。
    """    
    
    """生成以每个像素为中心具有不同形状的锚框"""
    ###########获取特征图宽高、每个像素锚框数，设备###########
    in_height, in_width = data.shape[-2:] # 取最后两个维度，即H,W
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # (s1,s2,s3) (r1,r2,r3) 不是一一组合的 s1组合所有r，r1组合所有s，减去一个重复的
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device) # 转为torch.tensor结构
    ratio_tensor = torch.tensor(ratios, device=device)

    ###########计算锚框中心坐标###########
    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长，可以将特征图的坐标转换到[0,1]之间
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h # 左上角的坐标 + 0.5 移到中心，再缩放至0,1之间
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1) # 展平
    # 假设center_h = [0.1667, 0.5, 0.8333]和center_w = [0.1667, 0.5, 0.8333]，则生成的网格坐标为：

    # shift_y = [[0.1667, 0.1667, 0.1667], [0.5, 0.5, 0.5], [0.8333, 0.8333, 0.8333]]
    # shift_x = [[0.1667, 0.5, 0.8333], [0.1667, 0.5, 0.8333], [0.1667, 0.5, 0.8333]]
    # 这些坐标覆盖了整个特征图,展平之后就是所有像素的中心点的坐标，0-1之间的相对坐标
    print("整个特征图的中心点的坐标数目")
    print(shift_x.shape,shift_y.shape)

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    # r1*s_all + s1*r_all
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 处理矩形输入，调整高宽比例适应图像
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    print(w,h)
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2
    print(anchor_manipulations.shape) # 每个像素相对中心点升成的五个坐标

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0) # 复制中心点坐标*5份，使得可以和偏移相加
    output = out_grid + anchor_manipulations # 锚框偏移量+中心点坐标
    return output.unsqueeze(0)

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
            
def test_anchors():
    """测试以anchor框
    """
    img = d2l.plt.imread('./img/catdog.jpg')
    print(img.shape) # plt按照h,w,c存储
    h, w = img.shape[:2] # 获取图片宽高

    print(h, w)
    X = torch.rand(size=(1, 3, h, w))
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]) # size缩放比，ratios宽高比
    # 返回的是Y每个中心点坐标对应的num_anchors的偏移量,即561 * 728 * 5
    Y.shape

    # 访问250,250中心的5个anchor中的第一个的四个偏移

    boxes = Y.reshape(h, w, 5, 4)
    print(boxes[250, 250, 0, :])

    d2l.set_figsize()
    bbox_scale = torch.tensor((w, h, w, h))
    fig = d2l.plt.imshow(img)
    # 以250,250为中心的5个anchor框
    show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
                's=0.75, r=0.5'])
    d2l.plt.savefig("img/test.png")

def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    # 计算每个box各自的面积
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    # 计算交集的左上角和右下角坐标
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

# 尽量保证每个真实框分配一个预测框，但是有可能阈值<0.5，所以也不是一定分配到
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量，初始化为-1
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

#@save
def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def one_example():
    img = d2l.plt.imread('./img/catdog.jpg')
    print(img.shape) # plt按照h,w,c存储
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])

    fig = d2l.plt.imshow(img)
    h, w = img.shape[:2] # 获取图片宽高
    bbox_scale = torch.tensor((w, h, w, h))
    show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
    fig = d2l.plt.imshow(img)
    show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
    d2l.plt.savefig("img/one_example.png")

if __name__ == "__main__":
    # test_anchors()
    one_example()
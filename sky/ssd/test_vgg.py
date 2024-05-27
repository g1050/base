from torch import nn
def vgg16(batch_norm=False) -> nn.ModuleList:
    """ 创建 vgg16 模型

    Parameters
    ----------
    batch_norm: bool
        是否在卷积层后面添加批归一化层
    """
    layers = []
    in_channels = 3
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'C', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        elif v == 'C':
            layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
        else:
            conv = nn.Conv2d(in_channels, v, 3, padding=1)

            # 如果需要批归一化的操作就添加一个批归一化层
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(True)])
            else:
                layers.extend([conv, nn.ReLU(True)])

            in_channels = v

    # 将原始的 fc6、fc7 全连接层替换为卷积层
    layers.extend([
        nn.MaxPool2d(3, 1, 1),
        nn.Conv2d(512, 1024, 3, padding=6, dilation=6),  # conv6 使用空洞卷积增加感受野
        nn.ReLU(True),
        nn.Conv2d(1024, 1024, 1),                        # conv7
        nn.ReLU(True)
    ])

    layers = nn.ModuleList(layers)
    return layers
# import torchvision.models as models

# 加载预训练的 VGG16 模型
# vgg16 = models.vgg16(pretrained=False)

# # 打印模型的各层
# print(vgg16)


from torch import nn
import torchvision.models as models

def vgg16(batch_norm=False) -> nn.ModuleList:
    """ssd中使用的是VGG16的变体

    Args:
        batch_norm (bool, optional): 是否需要归一化. Defaults to False.

    Returns:
        nn.ModuleList: vgg16的层
    """
    layers = [] # 空列表存储网络的各层
    in_channels = 3 # RGB 三通道图像
    cfg = [64,64,'M',128,128,'M',256,256,
           256,'C',512,512,512,'M',512,512,512]
    for v in cfg:
        # 最大池化
        if v == 'M':
            layers.append(nn.MaxPool2d(2,2))
        # ceil_mode的池化
        elif v == 'C':
            layers.append(nn.MaxPool2d(2,2,ceil_mode=True))
        # 输出通道数
        else:
            conv = nn.Conv2d(in_channels,v,3,padding=1)
            # 归一化
            if batch_norm:
                layers.extend([conv,nn.BatchNorm2d(v),nn.ReLU(True)])
            else:
                layers.extend([conv,nn.ReLU(True)])
            in_channels = v
    # 替换原始的全连接层为卷积层
    layers.extend([
        nn.MaxPool2d(3,1,1),
        nn.Conv2d(512,1024,3,padding=6,dilation=6), # conv6使用空洞卷积增加感受野
        nn.ReLU(True),
        nn.Conv2d(1024,1024,1),
        nn.ReLU(True)
    ])

    layers = nn.ModuleList(layers)
    return layers

def vgg16_in_torchvision():
    return models.vgg16(pretrained=False)

if __name__ == "__main__":
    ssd_net = vgg16()
    print(ssd_net)
    print(vgg16_in_torchvision())
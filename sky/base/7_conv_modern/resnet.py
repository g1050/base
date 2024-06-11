# 残差块：在旁边加一个通路，多支持一个简单的小模型
# 残差块加在哪里，都比较随意

# Resnet架构：类似vgg和googleNet，替换成了ResNet块

# ResNet 101 50 34 512(刷分) 18 152

# 重要作用：可以把网络变得更深

# 一条通路正常计算，另一条1*1卷积或者直接连接，最后将二者相加（两条通路上的结果）

# 核心思想:f(x) = x + g(x)，g(x)是之前的逻辑，如果发现x已经可以有很好的效果了，那么g(x)就会拿不到梯度

# 如果训练时候加入了大量的augmentation，那么最后的测试精度是有可能比训练精度要高的

# Resnet如何处理梯度消失，训练超过1000层的网络
# residual将乘法的链式求导法则变为加法
# y'' = f(x) + g(f(x)) ，

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):  
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block: # 除了第一resnet block的所有blcok的第一个residual block块
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

# 每个模块有4个卷积层（不包括恒等映射的
# 卷积层）。 加上第一个
# 卷积层和最后一个全连接层，共有18层。 因此，这种模型通常被称为ResNet-18
# 1(b1) + 4*4(b2-b5) + 1 (Linear) = 18

from export_onnx import export_onnx
export_onnx(net,torch.randn(1, 1, 224, 224),"resnet18.pth","resnet18.onnx")


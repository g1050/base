# 本质上是更深更大的LeNet
# 改进上：dropout、maxpool、Relu激活
# 重要的是计算机视觉研究方法的改变：原来是手动提取特征，现在是自动提取特征
## 比如在房价预测中，就要手动识别出那些关键的因素，对房价影响大的因素

# alexnet中使用了数据增强技术，随机裁剪、调色温、cropresize
# AlexNet的模型变得稍微深一点(加了一层卷积),变得更胖(最后的全连接特征变多)
# 2012年拿下ImageNet比赛,标志着新一轮的神经网络热潮开始

# lenet 2个卷积单元-> AlexNet 5个卷积单元

# 缺点： 结构比较随意


import torch
from torch import nn
from d2l import torch as d2l
class AlexNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.net = nn.Sequential(
            # 这里使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10))
    def forward(self,x):
        return self.net(x)

net = AlexNet()
print(net)
x = torch.randn(1,1,224,224)
y = x 
net(x)
# for layer in net.net:
#     print(layer)
#     y = layer(y)
#     print(y.shape)

# AlexNet(
#   (net): Sequential(
#     (0): Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4), padding=(1, 1))
#     (1): ReLU()
#     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (4): ReLU()
#     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU()
#     (8): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU()
#     (10): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU()
#     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (13): Flatten(start_dim=1, end_dim=-1)
#     (14): Linear(in_features=6400, out_features=4096, bias=True)
#     (15): ReLU()
#     (16): Dropout(p=0.5, inplace=False)
#     (17): Linear(in_features=4096, out_features=4096, bias=True)
#     (18): ReLU()
#     (19): Dropout(p=0.5, inplace=False)
#     (20): Linear(in_features=4096, out_features=10, bias=True)
#   )
# )
# Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4), padding=(1, 1))
# torch.Size([1, 96, 54, 54]) # h'=(224-11+1+2)/4=54
# ReLU()
# torch.Size([1, 96, 54, 54])
# MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# torch.Size([1, 96, 26, 26]) # 最后一个不足 54/2-1=26
# Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
# torch.Size([1, 256, 26, 26])
# ReLU()
# torch.Size([1, 256, 26, 26])
# MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# torch.Size([1, 256, 12, 12]) # 12
# Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# torch.Size([1, 384, 12, 12])
# ReLU()
# torch.Size([1, 384, 12, 12])
# Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# torch.Size([1, 384, 12, 12]) # h'=12-3+1+2 k=3,p=1,s=1时候不改变大小
# ReLU()
# torch.Size([1, 384, 12, 12])
# Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# torch.Size([1, 256, 12, 12])
# ReLU()
# torch.Size([1, 256, 12, 12])
# MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# torch.Size([1, 256, 5, 5]) # h'=12/2 -1 = 5
# Flatten(start_dim=1, end_dim=-1)
# torch.Size([1, 6400])
# Linear(in_features=6400, out_features=4096, bias=True)
# torch.Size([1, 4096])
# ReLU()
# torch.Size([1, 4096])
# Dropout(p=0.5, inplace=False)
# torch.Size([1, 4096])
# Linear(in_features=4096, out_features=4096, bias=True)
# torch.Size([1, 4096])
# ReLU()
# torch.Size([1, 4096])
# Dropout(p=0.5, inplace=False)
# torch.Size([1, 4096])
# Linear(in_features=4096, out_features=10, bias=True)
# torch.Size([1, 10])
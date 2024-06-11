# Network in Network 网络中的网络

# vgg以及AlexNet中大量的计算都花费在了全连接层中
# NiN使用1*1的卷积核来代替全连接层
# NiN使用了全局池化层

# 一个超级宽的单隐藏层MLP的问题在于可能过拟合

# todo: torchscript转C++，然后部署

# 全局池化的作用：不管特征图多大都可以压缩成1*1，把输入变小并且没有可学习的参数


import torch
from torch import nn
from d2l import torch as d2l
import onnx
from onnx import shape_inference

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
class NiNNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.net = nn.Sequential(
            nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            # 标签类别数是10
            nin_block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维的输出转成二维的输出，其形状为(批量大小,10)
            nn.Flatten())
        
    def forward(self,x):
        return self.net(x)
net = NiNNet() 
save_path = "NiN.pth"
onnx_path = "NiN.onnx"
torch.save(net,save_path)
dummy_input = torch.randn(1, 1, 224, 224)
# 导出模型为ONNX格式
torch.onnx.export(net, dummy_input, onnx_path,
                  input_names=['input'], output_names=['output'],
                  opset_version=11)
print("Export successfully.")
model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
onnx.save(model, onnx_path)
print("Shape infer successfully.")
import torch
from torch import nn
from d2l import torch as d2l
import onnx
from onnx import shape_inference


class LeNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))
    def forward(self,x):
        # y = x
        # for layer in self.net:
        #     print(layer)
        #     y = layer(y)
        #     print(y.shape)
        return self.net(x)

net = LeNet()
print(net)
x = torch.randn(1,1,28,28)
net(x)

path = "lenet.onnx"
save_path = "lenet_shape_infer.onnx"
dummy_input = torch.randn(1, 1, 28, 28)
pth_path = "lenet.pth"
# 导出模型为ONNX格式
# torch.onnx.export(net, dummy_input, "lenet.onnx",
#                   input_names=['input'], output_names=['output'],
#                   opset_version=11)
# print("Export successfully.")
# model = onnx.shape_inference.infer_shapes(onnx.load(path))
# onnx.save(model, save_path)
# print("Shape infer successfully.")
torch.save(net,pth_path)
# LeNet （卷积+激活+池化）*2 (线性层+激活)*3
# LeNet(
#   (net): Sequential(
#     (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)) 
#     (1): Sigmoid()
#     (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
#     (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#     (4): Sigmoid()
#     (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
#     (6): Flatten(start_dim=1, end_dim=-1)
#     (7): Linear(in_features=400, out_features=120, bias=True)
#     (8): Sigmoid()
#     (9): Linear(in_features=120, out_features=84, bias=True)
#     (10): Sigmoid()
#     (11): Linear(in_features=84, out_features=10, bias=True)
#   )
# )

# 输入shape (1,1,28,28) 28*28的灰度图
# Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
# torch.Size([1, 6, 28, 28]) 卷积后h'=28-5+1+4=28
# Sigmoid()
# torch.Size([1, 6, 28, 28])
# AvgPool2d(kernel_size=2, stride=2, padding=0)
# torch.Size([1, 6, 14, 14]) 平均池化后特征减半14*14
# Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
# torch.Size([1, 16, 10, 10]) 卷积后h'=14-5+1=10,padding=0不补边
# Sigmoid()
# torch.Size([1, 16, 10, 10])
# AvgPool2d(kernel_size=2, stride=2, padding=0) 
# torch.Size([1, 16, 5, 5]) # 继续减半
# Flatten(start_dim=1, end_dim=-1)
# torch.Size([1, 400]) # 展平 len = 16*5*5
# Linear(in_features=400, out_features=120, bias=True)
# torch.Size([1, 120]) # 全连接
# Sigmoid()
# torch.Size([1, 120])
# Linear(in_features=120, out_features=84, bias=True)
# torch.Size([1, 84]) # 全连接
# Sigmoid()
# torch.Size([1, 84])
# Linear(in_features=84, out_features=10, bias=True)
# torch.Size([1, 10]) # 全连接
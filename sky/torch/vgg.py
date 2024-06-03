import torch
import os
import torchvision
from mydataset import train_loader,test_loader
from torch import nn
os.environ['TORCH_HOME'] = os.path.realpath('../download')
# train_data = torchvision.datasets.ImageNet("../download",split="train",download=True,trans)
vgg16_false = torchvision.models.vgg16(pretrained=False) # 只有结构,使用的是未经过训练的初始化参数
vgg16_true = torchvision.models.vgg16(pretrained=True) # 包含参数
print(vgg16_true)
# Linear(in_features=4096, out_features=1000, bias=True) 分类类别是1000

# 用vgg16来对cifar10分类 新增层
# vgg16_true.add_module("add_linear",nn.Linear(1000,10))
vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_true)
path = "data/vgg16_false.pth"
torch.save(vgg16_false,path)
model_load = torch.load(path)
print(model_load)

# 修改
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)
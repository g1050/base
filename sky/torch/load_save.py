import torch
import os
import torchvision
from mysequential import Cifar10

# from mydataset import train_loader,test_loader
from torch import nn
os.environ['TORCH_HOME'] = os.path.realpath('../download')

# train_data = torchvision.datasets.ImageNet("../download",split="train",download=True,trans)
vgg16_false = torchvision.models.vgg16(pretrained=False) # 只有结构,使用的是未经过训练的初始化参数

# 1 保存结构和参数
print("-"*20)
path = "data/vgg16_false.pth"
# torch.save(vgg16_false,path)
model_load = torch.load(path)
print(model_load)

print("-"*20)
# 2 只保存参数，以字典格式,**官方推荐
path = "data/vgg16_false_dict.pth"
# torch.save(vgg16_false.state_dict(),path)
model_load_dict = torch.load(path)
print(model_load_dict) # 只有参数字典
print("-"*20)
model_load_dict = torchvision.models.vgg16()
model_load_dict.load_state_dict(torch.load(path))
print(model_load_dict) # 结构+参数
print("-"*20)



# 陷阱:
# path = "data/cifar10.pth"
# cifar = Cifar10()
# print(cifar)
# torch.save(cifar,path)
# model = torch.load(path)
# print(model)
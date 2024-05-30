 # dialation 中间空一个洞故名空洞卷积
# floor 向下取整 cell 向上取整

import torch
from torch import nn
input = torch.tensor([
    [1,2,0,3,1],
    [0,1,2,3,1],
    [1,2,1,0,0],
    [5,2,0,1,1],
    [2,1,0,1,1]
],dtype=torch.float32)
input = input.reshape(1,1,5,5)
print(input.shape)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,ceil_mode=False) # cell_mode true表示不足kernel_size(池化核大小)的也取
    def forward(self,x):
        return self.maxpool1(x)
    
model = Model()
output = model(input)
print(output)
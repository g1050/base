# compose transforms的集合
# sequential layers的集合
import torch
from torch import nn

class Cifar10(nn.Module):
    def __init__(self):
        super(Cifar10,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32,32,5,padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,5,padding=2)
        self.maxpool3 = nn.MaxPool2d(2) # 宽高减半
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Linear(1024,64)
        self.linear2 = nn.Linear(64,10)

        self.model1 = nn.Sequential(
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.maxpool2,
            self.conv3,
            self.maxpool3,
            self.flatten1,
            self.linear1,
            self.linear2
        )
    def forward(self,x):
        return self.model1(x)

input = torch.ones((64,3,32,32))
model = Cifar10()
output = model(input)
print(output.shape)

# tensorboard 可视化网络结构 
# writer.add_graph(model,input)
# L1 loss 差值求和/n
import torch
from torch import nn
from mysequential import Cifar10
from mydataset import train_loader,test_loader

def many_loss():
    inputs = torch.tensor([1,2,3],dtype=torch.float32).reshape(1,1,1,3)
    targets = torch.tensor([1,2,5],dtype=torch.float32).reshape(1,1,1,3)
    loss = nn.L1Loss(reduction='sum') # L1 loss
    result = loss(inputs,targets)
    print(result)

    loss_mse = nn.MSELoss()
    result = loss_mse(inputs,targets) # 均方误差,L2 loss
    print(result)

    # 交叉熵
    # coco 80个类别
    x = torch.randn((80))
    print(x,x.shape)
    y = torch.tensor([1])
    x = x.reshape((1,80))
    loss_cross = nn.CrossEntropyLoss()
    result = loss_cross(x,y)
    print(result)

def test():
    model = Cifar10()
    loss = nn.CrossEntropyLoss()
    for data in train_loader:
        imgs,targets = data
        print(imgs.shape)
        print(f"targets info {targets} {targets.shape} {targets.dtype}" )
        output = model(imgs)
        print(f"output info {output} {output.shape} {output.dtype}" )
        l = loss(output,targets)
        print(l)
        l.backward()
        
        break
if __name__ == "__main__":
    test()
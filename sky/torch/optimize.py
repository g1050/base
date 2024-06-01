from mysequential import Cifar10
from mydataset import train_loader,test_loader
from torch import nn
import torch
device = "cuda:0"
model = Cifar10().to(device)
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(),lr=0.01)
for data in train_loader:
    imgs,targets = data
    imgs = imgs.to(device)
    targets = targets.to(device)
    print(imgs.shape)
    print(f"targets info {targets} {targets.shape} {targets.dtype}" )
    output = model(imgs)
    print(f"output info {output} {output.shape} {output.dtype}" )
    l = loss(output,targets)
    print(l)
    optim.zero_grad()
    l.backward()
    optim.step()
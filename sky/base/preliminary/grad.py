import torch

# 自动微分
x = torch.arange(4,dtype=torch.float32)
x.requires_grad_(True)
y = x*x
y.sum().backward()
print(x.grad)

x.grad.zero_()
y = torch.dot(x,x)
y.backward()
print(x.grad)
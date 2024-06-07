# multilayer perception 多层感知机：全连接+非线性激活 多层重叠 隐藏层128->64->32->16->8这样逐渐缩小
# 感知机只能产生线性分割面没办法解决XOR函数
# 单层感知机等价于批量为1的梯度下降
# 激活函数：非线性，sigmoid(0,1)、tanh(-1,1)、ReLU
# mlp vs svm
import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms

def mlp_zero():
    pass

def mlp_torch():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.model1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
        def forward(self,x):
            return self.model1(x)
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.randn(28,28) # 标准正态分布,flatten期望(n,c,h,w)
    input = torch.randn(1,1,28,28)
    print(input.shape)
    model = MLP()
    model.apply(init_weights)
    print(model)
    model = model.to(device)
    input = input.to(device)
    output = model(input)
    print(output.shape)

    batch_size, lr, num_epochs = 256,0.1,10
    loss = nn.CrossEntropyLoss(reduction='none')
    ## 
    # 'none'：不进行任何归约（即不对批次中的损失求和或求平均），返回每个样本的损失。
    # 'mean'：对批次中的损失求平均，返回一个标量。
    # 'sum'：对批次中的损失求和，返回一个标量。
    loss = loss.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)

    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="../../download",train=True,transform=trans,download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../../download",train=False,transform=trans,download=True)
    
    train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=4)
    test_iter = data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=4)

    # timer = d2l.Timer()
    # for x,y in train_iter:
    #     # continue
    #     print(f"x {x.shape} y {y[0]}") # x的shape N,C,H,W
    # print(f"length of train dataset {len(mnist_train)}")
    # print(f"length of train {len(train_iter)},time {timer.stop()}sec")

    # 开始训练

    def train_epoch(net, train_iter, loss, optimizer, device):
        net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y).mean()
            l.backward()
            optimizer.step()
    
    def evaluate_accuracy(net, data_iter, device):
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in data_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                # print("-"*20)
                # print(X.shape,y.shape)
                # print(y_hat.shape)
                # print("-"*20)
                predicted = torch.argmax(y_hat, dim=1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total

    for epoch in range(num_epochs):
        train_epoch(model, train_iter, loss, optimizer, device)
        train_acc = evaluate_accuracy(model, train_iter, device)
        test_acc = evaluate_accuracy(model, test_iter, device)
        print(f'Epoch {epoch + 1}, Train acc {train_acc:.4f}, Test acc {test_acc:.4f}')

if __name__ == "__main__":
    # mlp_zero()
    mlp_torch()
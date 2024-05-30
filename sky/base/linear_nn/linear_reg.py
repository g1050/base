# 由于选取所有样本计算梯度，计算量太大，所以选用小批量梯度下降
# 超参数：可以调整，但是不在训练过程中更新的参数，如batch_size和学习率
import torch
from d2l import torch as d2l
import math
import numpy as np
import random
from torch.utils import data
from torch import nn

# 加速计算
def main():
    n = 100000
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.zeros(n)
    timer = d2l.Timer()

    for i in range(n):
        c[i] = a[i] + b[i]
    print(timer.stop())

    timer.start()
    c = a+b
    print(timer.stop())
    print(c.device)

    device = 'cuda:0'
    a = a.to(device)
    b = b.to(device)
    c = c.to(device)
    timer.start()
    c = a + b
    print(timer.stop())
    print(c.device)

# 在高斯噪声的假设下，最小化均方误差等价于对线性模型的极大似然估计
# 全连接：每个输入都与每个输出相连接,每个输入都对输出有影响

def gauss():
    # 高斯函数
    def normal(x,mu,sigma):
        p = 1/math.sqrt(2*math.pi*sigma**2)
        return p * np.exp((-0.5/sigma**2)*(x-mu)**2)
    x = np.arange(-7,7,0.01)
    params = [(0,1),(0,2),(3,1)]
    d2l.plot(x,[normal(x,mu,sigma) for mu,sigma in params],xlabel='x',ylabel='y',figsize=(4.5,2.5),
             legend=[f"{mu,sigma}" for mu,sigma in params])
    d2l.plt.savefig('data/img.png')

def implement():
    def create_data(w,b,nums_example):
        x = torch.normal(0,1,(nums_example,len(w)))
        print(f"x.shape {x.shape} w.shape {w.shape}")
        y = torch.matmul(x,w) + b
        y += torch.normal(0,0.01,y.shape) # 加入噪声
        return x,y.reshape(-1,1)
    true_w = torch.tensor([2,-3.4])
    true_b = 4.2
    features,labels = create_data(true_w,true_b,1000)

    def read_data(batch_size,features,labels):
        nums_example = len(features)
        indicies = list(range(nums_example))
        random.shuffle(indicies)
        for i in range(0,nums_example,batch_size):
            index_tensor = torch.tensor(indicies[i:min(i+batch_size,nums_example)]) # 防止越界
            yield features[index_tensor],labels[index_tensor]

    batch_size = 10
    for x,y in read_data(batch_size,features,labels):
        print(f"x {x} \n y {y}")
        break
    w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    def net(x,w,b):
        return torch.matmul(x,w) + b
    def loss(y_hat,y):
        return (y_hat-y.reshape(y_hat.shape))**2/2
    def sgd(params,batch_size,lr):
        with torch.no_grad():
            for param in params:
                param -= lr*param.grad / batch_size
                param.grad.zero_()

    lr = 0.03
    num_epochs = 3
    for epoch in range(0,num_epochs):
        for x,y in read_data(batch_size,features,labels):
            f = loss(net(x,w,b),y)
            f.sum().backward()
            sgd([w,b],batch_size,lr)
        with torch.no_grad():
            train_l = loss(net(features,w,b),labels)
            print(f"epoch {epoch}, loss {train_l.mean()}")
    print(f"w 误差 {true_w-w }\nb 误差 {true_b -b}\n")
    print(f"true w {true_w} true b {true_b}\n")
    print(f"w {w} b {b}")

    d2l.set_figsize()
    d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
    d2l.plt.savefig("data/img.png")

def simple_implement():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w,true_b,1000)
    def load_array(data_arrays,batch_size,is_train=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset,batch_size,shuffle=is_train)
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    print(next(iter(data_iter)))
    print(next(iter(data_iter))[0].shape,next(iter(data_iter))[1].shape)
    
    net = nn.Sequential(nn.Linear(2,1))
    net[0].weight.data.normal_(0,0.01)
    net[0].weight.data.fill_(0)
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.03)
    num_epochs = 3
    for epoch in range(num_epochs):
        for feature,label in data_iter:
            y_hat = net(feature)
            l = loss(y_hat,label)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        # 计算损失
        l = loss(net(features),labels)
        print(f"expoch {epoch+1} loss {l}")
    print(f"true w {true_w} true b {true_b} ")
    print(f"w {net[0].weight.data} b {net[0].bias.data}")
if __name__ == "__main__":
    # main()
    # gauss()
    # implement()
    simple_implement()
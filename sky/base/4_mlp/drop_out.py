# 正则：避免权重太大，导致过拟合

# 丢弃法：在层之间加噪音
# 扰动： 以概率p变为0,概率1-p变为 xi/(1-p)，其期望值是不变的
# 用法：在全连接的输出上进行drop_out，尝作用于mlp的隐藏层输出上
# drop out属于正则，只在训练的时候使用正则，所以用于部署的模型没有见过Dropout层
# dropout的p也属于超参数

# todo： 正则的含义是什么、正则项

# dropout -> fc，BN -> conv


# 扰动： 以概率p变为0,概率1-p变为 xi/(1-p)，其期望值是不变的
# dropout主要是对全连接层训练使用
from torch import nn
from d2l import torch as d2l
import torch
from torchvision import transforms
import torchvision
def dropout_concise():
    dropout1, dropout2 = 0.2, 0.5
    net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层, 参数为概率
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="../../download",train=True,transform=trans,download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../../download",train=False,transform=trans,download=True)
    
    # train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=4)
    # test_iter = data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=4)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

if __name__ == "__main__":
    dropout_concise()
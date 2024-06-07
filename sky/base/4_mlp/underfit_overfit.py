# 过拟合和欠拟合
# 过拟合：模型容量大，数据简单，泛化能力差，泛化误差大
# 欠拟合：模型容量小，数据复杂
# overfit和underfit通过train dataset和val dataset来观察

# 训练误差：训练时候的误差
# 泛化误差：新的数据集上的误差

# 训练集：训练模型参数
# 验证数据集：评估模型好坏，不要和训练数据集混在一起，不参与训练模型，用来调整超参的好坏等等，不能代表模型泛化能力，只用来训练超参数
# 测试数据集：只用一次的数据集
# 超参数：学习率，epoch num，模型深度，宽度

# K-折交叉验证
# 数据集不够的时候采用，但是计算成本高，深度学习一般不使用

# 使用多项式拟合来变现，过拟合的表现就是train loss下降,test loss确不下降/上升

import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def overfit():
    max_degree = 20  # 多项式的最大阶数
    n_train, n_test = 100, 100  # 训练和测试数据集大小
    true_w = np.zeros(max_degree)  # 分配大量的空间
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) # 存放多项式的系数

    # 生成特征数据
    features = np.random.normal(size=(n_train + n_test, 1)) # 特征只有一维
    np.random.shuffle(features)
    # 升成多项式数据
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) # 计算各个幂次[0,20)幂次
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)! 对每列分别计算阶乘
    
    # labels的维度:(n_train+n_test,)
    labels = np.dot(poly_features, true_w) # x乘系数
    labels += np.random.normal(scale=0.1, size=labels.shape) # 加入噪声

    # NumPy ndarray转换为tensor
    true_w, features, poly_features, labels = [torch.tensor(x, dtype=
        torch.float32) for x in [true_w, features, poly_features, labels]]

    print(features.shape,poly_features.shape,labels.shape)
    print(true_w)
    # features[:2], poly_features[:2, :], labels[:2]
    def evaluate_loss(net, data_iter, loss):  #@save
        """评估给定数据集上模型的损失"""
        metric = d2l.Accumulator(2)  # 损失的总和,样本数量
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            metric.add(l.sum(), l.numel())
        return metric[0] / metric[1]

    def train(train_features, test_features, train_labels, test_labels,tag_name:str,
          num_epochs=400):
        loss = nn.MSELoss(reduction='none')
        input_shape = train_features.shape[-1]
        # 不设置偏置，因为我们已经在多项式中实现了它
        net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
        batch_size = min(10, train_labels.shape[0])
        train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                    batch_size)
        test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                                batch_size, is_train=False)
        trainer = torch.optim.SGD(net.parameters(), lr=0.01)
        animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                                xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                                legend=['train', 'test'])
        for epoch in range(num_epochs):
            d2l.train_epoch_ch3(net, train_iter, loss, trainer)
            if epoch == 0 or (epoch + 1) % 20 == 0:
                # animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                #                         evaluate_loss(net, test_iter, loss)))
                train_loss = evaluate_loss(net, train_iter, loss)
                test_loss = evaluate_loss(net, test_iter, loss)
                writer.add_scalars(tag_name,{"test_loss":test_loss,"train_loss":train_loss},epoch)
                print(train_loss,test_loss)
        print('weight:', net[0].weight.data.numpy())

    # 正常
    train(poly_features[:n_train, :], poly_features[n_train:, :],
        labels[:n_train], labels[n_train:],"normal")

    # 欠拟合
    # 从多项式特征中选择前2个维度，即1和x
    train(poly_features[:n_train, :2], poly_features[n_train:, :2],
        labels[:n_train], labels[n_train:],"underfit")
    
    # 过拟合
    # 从多项式特征中选取所有维度
    # 由于模型不复杂，所以过拟合的的预测结果也差不多
    train(poly_features[:n_train, :], poly_features[n_train:, :],
        labels[:n_train], labels[n_train:],"overfit",num_epochs=1500)
if __name__ == "__main__":
    overfit()

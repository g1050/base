# 控制模型容量
# 1. 模型参数少 2. 限制每个参数可选范围，限制参数范围模型空间就有限，相对来说平滑，就可以一定程度避免过拟合
# 硬性限制：w的L2范数小于sita，每个w小于根号sita
# 柔性限制：损失函数后面+罚函数

# 1-eta*lamda，后者乘积小于1，深度学习中称为权重衰退
# lamda就成为了控制模型复杂度的超参数

# weigh_decay 防止过拟合 权重衰退

# l2范数是开根号的，罚用的是l2的平方
import torch
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def wd_zero():
    # loss后加了一个罚函数,数学上推导出来的
    def l2_penalty(w):
        return torch.sum(w.pow(2)) / 2
    l = loss(net(X), y) + lambd * l2_penalty(w) # 其中的lamdb即为权重衰退的超参数

def wd_concise():
    def init_params():
        w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        return [w, b]
    
    def train(lambd):
        w, b = init_params()
        def l2_penalty(w):
            return torch.sum(w.pow(2)) / 2

        # 匿名函数
        net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
        num_epochs, lr = 100, 0.003
        # animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
        #                         xlim=[5, num_epochs], legend=['train', 'test'])
        for epoch in range(num_epochs):
            for X, y in train_iter:
                # 增加了L2范数惩罚项，
                # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
                l = loss(net(X), y) + lambd * l2_penalty(w)
                l.sum().backward()
                d2l.sgd([w, b], lr, batch_size)
            if (epoch + 1) % 5 == 0:
                # animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                #                         d2l.evaluate_loss(net, test_iter, loss)))
                train_loss = d2l.evaluate_loss(net, train_iter, loss)
                test_loss = d2l.evaluate_loss(net, test_iter, loss)
                writer.add_scalars(f"weight_decay_{lambd}",{"test_loss":test_loss,"train_loss":train_loss},epoch)
                print(train_loss,test_loss)
        print('w的L2范数是：', torch.norm(w).item())
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = d2l.synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size)
    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)
    # 忽略正则化，不使用正则化，模型容量大，泛化能力差，在训练集上表现好，在测试集上loss不下降
    train(lambd=0)
    # 使用正则化
    train(lambd=3)

if __name__ == "__main__":
    wd_concise()
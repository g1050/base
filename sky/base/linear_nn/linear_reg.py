# 由于选取所有样本计算梯度，计算量太大，所以选用小批量梯度下降
# 超参数：可以调整，但是不在训练过程中更新的参数，如batch_size和学习率
import torch
from d2l import torch as d2l
import math
import numpy as np

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
    pass
if __name__ == "__main__":
    # main()
    # gauss()
    implement()
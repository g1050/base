# 数值稳定性
# 梯度爆炸：1.5^100=非常大的数，可能超出float，100表示网络很深
## 梯度爆炸后，lr太小训练太慢，lr太大爆炸，需要在训练过程中不断调整学习率
## fp16训练快，但是很可能超出范围，梯度爆炸

# 梯度消失：0.8^100=非常小的数
## sigmoid，输入大点的时候，梯度很小，当层数深的时候，梯度消失
## 梯度消失后，训练没有进展，仅仅顶部训练效果好，底部尤为严重


# 训练稳定：让梯度值在合理范围内
# 1. 乘法变加法
# 2. 归一化、裁剪
# 3. 合理权重初始化和激活函数
# 3 展开
# 希望每层的输出和梯度都是稳定的，即均值方差一定
# 初始化：基于上述假设，可以求得初始化时的公式，即xavier方法
# 激活：基于上述假设，激活函数要近似于y=x，所以tanh和relu效果不错，sigmoid可以*4-1（根据泰勒展开）近似y=x
import torch
from d2l import torch as d2l
def test_code():
    # pytorch提供的初始化方法Xavier初始化和kaiming初始化

    # 梯度消失
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)
    y.backward(torch.ones_like(x))

    # d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
    #         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))

    M = torch.normal(0, 1, size=(4,4))
    print('一个矩阵 \n',M)
    for i in range(100):
        M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

    print('乘以100个矩阵后\n', M) # 梯度爆炸
if __name__ == "__main__":
    test_code()
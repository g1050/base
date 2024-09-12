import torch
import torch.nn as nn

# 对于2D卷积输出的BN层
bn = nn.BatchNorm2d(num_features=64)

# 假设输入是 [batch_size, channels, height, width]
input = torch.ones(16, 64, 32, 32)
output = bn(input)
print(output)
print(output.shape)
print(bn.weight) # gamma
print(bn.bias) # beta

# bn层先进行归一化，减均值除方差，分母加一个很小的数upsilon
# 然后进行缩放和偏移 y = gamma * x_hat + beta

# 关闭缩放和平移
# bn_no_affine = nn.BatchNorm2d(num_features=64, affine=False)

####
# conv + bn融合，训练时候不能融合，推理时候可以融合
# 训练时候是按照batch调整参数的，bn就是为了防止一个批次内分布的偏移提出的，除此外还有正则化的作用，防止梯度爆炸

import torch
import torch.nn as nn

# 定义一个包含卷积和BN的模型
class ConvBNModel(nn.Module):
    def __init__(self):
        super(ConvBNModel, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        return self.bn(self.conv(x))

model = ConvBNModel()

# 使用 PyTorch 的融合功能
fused_model = torch.quantization.fuse_modules(model, ['conv', 'bn'])

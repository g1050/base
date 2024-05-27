import torch
import torch.nn as nn
import torch.nn.functional as F
def BatchNorm2d(num_features=3,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True):
    """2维度批归一化

    Args:
        num_features (int, optional): 特征的通道数. Defaults to 3.
        eps (_type_, optional): epsilon,为了计算稳定性. Defaults to 1e-5.
        momentum (float, optional): 运行时均值和方差的动量. Defaults to 0.1.
        affine (bool, optional): 可学习的放射参数. Defaults to True.
        track_running_stats (bool, optional): 跟踪运行时. Defaults to True.
    """
    pass
conv_layer = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
batch_norm = nn.BatchNorm2d(num_features=16)
input = torch.randn(1,3,32,32)
output1 = conv_layer(input)
output2 = batch_norm(output1)
print(output1.shape)
print(output2.shape)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # 定义一个二维卷积层
# conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# # 定义一个二维批归一化层
# batch_norm = nn.BatchNorm2d(num_features=16)

# # 创建一个示例输入张量，形状为 (batch_size, in_channels, height, width)
# input_tensor = torch.randn(1, 3, 32, 32)  # batch_size = 1, in_channels = 3, height = 32, width = 32

# # 应用卷积层到输入张量
# output_tensor = conv_layer(input_tensor)

# # 应用批归一化层到卷积层的输出
# normalized_output = batch_norm(output_tensor)

# print(normalized_output.shape)

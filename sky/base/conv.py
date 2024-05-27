# import torch
# import torch.nn as nn

# # 定义一个二维卷积层
# conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, \
#                        stride=1, padding=0, dilation=1, \
#                         groups=1, bias=True, padding_mode='zeros')

import torch
import torch.nn as nn

def Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=0,dilation=1,\
                    groups=1,bias=True,padding_mode='zeros'):
    """卷积操作

    Args:
        in_channels (int, optional): 输入通道数. Defaults to 3.
        out_channels (int, optional): 输出通道数. Defaults to 3.
        kernel_size (int, optional): 卷积核大小. Defaults to 3.
        stride (int, optional): 步长. Defaults to 1.
        padding (int, optional): 补充0的层数. Defaults to 0.
        dilation (int, optional): 膨胀,卷积核之间的距离. Defaults to 1.
        bias (bool, optional): 学习偏置. Defaults to True.
        padding_mode (str, optional): 填充方式,0填充,reflect填充,边缘复制填充,循环填充. Defaults to 'zeros'.
    """
    pass

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
with torch.no_grad():
    conv_layer.weight.fill_(1.0)
print(conv_layer.weight)
input_tensor = torch.reshape(torch.range(1,16),(1,1,4,4))
output_tensor = conv_layer(input_tensor)
print(output_tensor.shape)
print(output_tensor)
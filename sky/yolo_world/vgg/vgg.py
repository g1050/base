import torch
from torch import nn

# vgg块：若干conv+relu，最后通过池化
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
    
# Sequential output shape:         torch.Size([1, 64, 112, 112])
# Sequential output shape:         torch.Size([1, 128, 56, 56])
# Sequential output shape:         torch.Size([1, 256, 28, 28])
# Sequential output shape:         torch.Size([1, 512, 14, 14])
# Sequential output shape:         torch.Size([1, 512, 7, 7])
# Flatten output shape:    torch.Size([1, 25088])
# Linear output shape:     torch.Size([1, 4096])
# ReLU output shape:       torch.Size([1, 4096])
# Dropout output shape:    torch.Size([1, 4096])
# Linear output shape:     torch.Size([1, 4096])
# ReLU output shape:       torch.Size([1, 4096])
# Dropout output shape:    torch.Size([1, 4096])
# Linear output shape:     torch.Size([1, 10])
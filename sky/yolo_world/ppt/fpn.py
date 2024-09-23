import torch
import torch.nn as nn
import torchvision.models as models

class FPN(nn.Module):
    def __init__(self, backbone, out_channels=256):
        super(FPN, self).__init__()
        
        # 从 ResNet 提取不同阶段的特征层 (C2, C3, C4, C5)
        self.backbone = backbone
        self.out_channels = out_channels
        
        # 1x1 卷积层将 ResNet 的输出特征层调整到相同的通道数
        self.conv1x1_C5 = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.conv1x1_C4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.conv1x1_C3 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.conv1x1_C2 = nn.Conv2d(256, out_channels, kernel_size=1)
        
        # 3x3 卷积层精细化输出特征图
        self.conv3x3_P5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3x3_P4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3x3_P3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3x3_P2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # ResNet 提取特征 (C2, C3, C4, C5)
        C2 = self.backbone.layer1(x)  # Output: [batch, 256, H/4, W/4]
        C3 = self.backbone.layer2(C2) # Output: [batch, 512, H/8, W/8]
        C4 = self.backbone.layer3(C3) # Output: [batch, 1024, H/16, W/16]
        C5 = self.backbone.layer4(C4) # Output: [batch, 2048, H/32, W/32]

        # 1x1 卷积调整通道数
        P5 = self.conv1x1_C5(C5)  # P5 is the top-most layer
        P4 = self.conv1x1_C4(C4) + nn.functional.interpolate(P5, scale_factor=2, mode='nearest')
        P3 = self.conv1x1_C3(C3) + nn.functional.interpolate(P4, scale_factor=2, mode='nearest')
        P2 = self.conv1x1_C2(C2) + nn.functional.interpolate(P3, scale_factor=2, mode='nearest')
        
        # 3x3 卷积精细化
        P5 = self.conv3x3_P5(P5)
        P4 = self.conv3x3_P4(P4)
        P3 = self.conv3x3_P3(P3)
        P2 = self.conv3x3_P2(P2)
        
        return P2, P3, P4, P5

# 测试 FPN 模型
if __name__ == "__main__":
    # 使用预训练的 ResNet50 作为骨干网络
    resnet = models.resnet50(pretrained=True)
    
    # 去掉最后的分类层 (全连接层)
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    
    # 创建 FPN 实例
    fpn = FPN(resnet)
    
    # 创建随机输入
    x = torch.randn(1, 3, 224, 224)  # [batch_size, channels, height, width]
    
    # 运行 FPN
    P2, P3, P4, P5 = fpn(x)
    
    # 输出特征图的形状
    print(f"P2 shape: {P2.shape}")  # Expected shape: [1, 256, 56, 56]
    print(f"P3 shape: {P3.shape}")  # Expected shape: [1, 256, 28, 28]
    print(f"P4 shape: {P4.shape}")  # Expected shape: [1, 256, 14, 14]
    print(f"P5 shape: {P5.shape}")  # Expected shape: [1, 256, 7, 7]

# 代码说明：
# backbone: 使用 ResNet50 提取特征。我们从 layer1 到 layer4 提取特征，即 C2 到 C5。
# 1x1 卷积：对每个层（C2, C3, C4, C5）使用 1x1 卷积调整到相同的通道数（256）。
# 上采样和融合：将高层次的特征图上采样（如 P5 上采样到与 P4 一样大小），并与对应的低层特征融合。
# 3x3 卷积：对每个金字塔层（P2, P3, P4, P5）进一步用 3x3 卷积进行精细化处理。
# 这样，通过 FPN，我们得到了多尺度的特征图，适合处理不同大小的目标。
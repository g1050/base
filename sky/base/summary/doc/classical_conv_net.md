# LeNet
1. 网络定义
```
class LeNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))
    def forward(self,x):
        # y = x
        # for layer in self.net:
        #     print(layer)
        #     y = layer(y)
        #     print(y.shape)
        return self.net(x)
```
LeNet （卷积+激活+池化）\*2 (线性层+激活)\*3
2. shape infer
```
LeNet(
  (net): Sequential(
    (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)) 
    (1): Sigmoid()
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): Sigmoid()
    (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Linear(in_features=400, out_features=120, bias=True)
    (8): Sigmoid()
    (9): Linear(in_features=120, out_features=84, bias=True)
    (10): Sigmoid()
    (11): Linear(in_features=84, out_features=10, bias=True)
  )
)
# 输入shape (1,1,28,28) 28*28的灰度图
# Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
# torch.Size([1, 6, 28, 28]) 卷积后h'=28-5+1+4=28
# Sigmoid()
# torch.Size([1, 6, 28, 28])
# AvgPool2d(kernel_size=2, stride=2, padding=0)
# torch.Size([1, 6, 14, 14]) 平均池化后特征减半14*14
# Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
# torch.Size([1, 16, 10, 10]) 卷积后h'=14-5+1=10,padding=0不补边
# Sigmoid()
# torch.Size([1, 16, 10, 10])
# AvgPool2d(kernel_size=2, stride=2, padding=0) 
# torch.Size([1, 16, 5, 5]) # 继续减半
# Flatten(start_dim=1, end_dim=-1)
# torch.Size([1, 400]) # 展平 len = 16*5*5
# Linear(in_features=400, out_features=120, bias=True)
# torch.Size([1, 120]) # 全连接
# Sigmoid()
# torch.Size([1, 120])
# Linear(in_features=120, out_features=84, bias=True)
# torch.Size([1, 84]) # 全连接
# Sigmoid()
# torch.Size([1, 84])
# Linear(in_features=84, out_features=10, bias=True)
# torch.Size([1, 10]) # 全连接
```
3. 可视化
lenet架构示意图
![](../images/lenet.svg)
lenet的onnx结构  
![](../images/lenet_shape_infer.onnx.png)

# AlexNet
> AlexNet相较于Lenet模型稍微加大（线性层）、加深（卷积层）一点，
由2个卷积单元扩展到5个卷积单元，最后的全连接层"变胖"，使用了更多的特征做全连接，AlexNet还使用了数据增强技术
AlexNet标志着新一轮的神经网络的热潮的开始，提取特征从传统的手工提取变为现在的机器提取。
 1. 网络定义
 ```
 AlexNet(
  (net): Sequential(
    (0): Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU()
    (8): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU()
    (10): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU()
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (13): Flatten(start_dim=1, end_dim=-1)
    (14): Linear(in_features=6400, out_features=4096, bias=True)
    (15): ReLU()
    (16): Dropout(p=0.5, inplace=False)
    (17): Linear(in_features=4096, out_features=4096, bias=True)
    (18): ReLU()
    (19): Dropout(p=0.5, inplace=False)
    (20): Linear(in_features=4096, out_features=10, bias=True)
  )
)
```
2. shape infer
输入示例采用224  
```  
Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4), padding=(1, 1))
torch.Size([1, 96, 54, 54]) # h'=(224-11+1+2)/4=54
ReLU()
torch.Size([1, 96, 54, 54])
MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
torch.Size([1, 96, 26, 26]) # 最后一个不足 54/2-1=26
Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
torch.Size([1, 256, 26, 26])
ReLU()
torch.Size([1, 256, 26, 26])
MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
torch.Size([1, 256, 12, 12]) # 12
Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
torch.Size([1, 384, 12, 12])
ReLU()
torch.Size([1, 384, 12, 12])
Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
torch.Size([1, 384, 12, 12]) # h'=12-3+1+2 k=3,p=1,s=1时候不改变大小
ReLU()
torch.Size([1, 384, 12, 12])
Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
torch.Size([1, 256, 12, 12])
ReLU()
torch.Size([1, 256, 12, 12])
MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
torch.Size([1, 256, 5, 5]) # h'=12/2 -1 = 5
Flatten(start_dim=1, end_dim=-1)
torch.Size([1, 6400])
Linear(in_features=6400, out_features=4096, bias=True)
torch.Size([1, 4096])
ReLU()
torch.Size([1, 4096])
Dropout(p=0.5, inplace=False)
torch.Size([1, 4096])
Linear(in_features=4096, out_features=4096, bias=True)
torch.Size([1, 4096])
ReLU()
torch.Size([1, 4096])
Dropout(p=0.5, inplace=False)
torch.Size([1, 4096])
Linear(in_features=4096, out_features=10, bias=True)
torch.Size([1, 10])
```
3. 可视化  
lenet和AlexNet架构对比  
![](../images/alexnet.svg)

# VGG
1. 网络定义
vgg11是由5个卷积块+3个全连接层组成，故名vgg11  
从LeNet->AlexNet（加大加深）->VGG（加大加深、模块化）并没有太多新的结构提出，只是增大了模型规模，增大了参数量  
然后加入了一些正则化的方法来限制模型复杂度避免过拟合，增加一些数据增强的方法提高精度
```
Sequential(
  (0): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (4): Sequential(
    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (5): Flatten(start_dim=1, end_dim=-1)
  (6): Linear(in_features=25088, out_features=4096, bias=True)
  (7): ReLU()
  (8): Dropout(p=0.5, inplace=False)
  (9): Linear(in_features=4096, out_features=4096, bias=True)
  (10): ReLU()
  (11): Dropout(p=0.5, inplace=False)
  (12): Linear(in_features=4096, out_features=10, bias=True)
)
```
2. 可视化  
![](../images/vgg.svg)

# NiN
> NiN也使用了block的概念，但是引入了全局平均池化，就是对通道进行池化，不管多大的通道都可以压缩成1*1，并且没有可学习的参数
NiN是在AlexNet后提出，因为大量的参数都在全连接层，所以NiN引入全局池化，而取消了fc，大大减少了参数量，但是可能带来训练变慢的问题
```
Sequential output shape:     torch.Size([1, 96, 54, 54])
MaxPool2d output shape:      torch.Size([1, 96, 26, 26])
Sequential output shape:     torch.Size([1, 256, 26, 26])
MaxPool2d output shape:      torch.Size([1, 256, 12, 12])
Sequential output shape:     torch.Size([1, 384, 12, 12])
MaxPool2d output shape:      torch.Size([1, 384, 5, 5])
Dropout output shape:        torch.Size([1, 384, 5, 5])
Sequential output shape:     torch.Size([1, 10, 5, 5])
AdaptiveAvgPool2d output shape:      torch.Size([1, 10, 1, 1])
Flatten output shape:        torch.Size([1, 10])
```
例如上述中的最后Flatten之前对5*5进行全局池化，直接变成了1\*1的用于分类的类别信息
![](../images/NiN.png)
# GoogLeNet
> 名字中的L大写致敬LeNet，Inception直译“开端”，也是电影名字盗梦空间，类似transformer的变形金刚
卷积核大小的选取对结果以及训练性能影响较大，Inception结构中使用类似并联的结构，并联了多种尺寸的卷积核
Inception可以计算更少，复杂度低，提取的信息更丰富，但就设计上来说没有太多道理，实验证明其是work的就好
可以参考的设计思想，从数据到输出，宽高减半通道数加倍，不断的来提取特征
GoogLeNet使用了9个Inception，是第一个达到上百层的模型
![](../images/googleNet.png)
# ResNet
ResNet中引入了残差块，可以大大加深模型的深度，类似导数的加法法则y = f(x) + g(f(x))，其中f(x)表示上层直接连接，g(f(x))表示上层处理后连接，求导时候  
利用加法法则，变乘法为加法，避免梯度相乘过大    
![](../images/resnet.png)
上述图中的红色是一个resnet block其中包括两个残差块，第一个残差块加入了1\*1卷积块，第二个直接连接  
每个模块有4个卷积层（不包括恒等映射的卷积层）。 加上第一个卷积层和最后一个全连接层，共有18层。  
因此，这种模型通常被称为ResNet-18，1(b1) + 4*4(b2-b5) + 1 (Linear) = 18
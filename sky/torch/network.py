from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
def main():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        # 前向传播
        def forward(self,input):
            output = input + 1
            return output
        
    net = Model()
    x = torch.arange(10)
    print(x)
    print(net(x)) # call函数中的钩子函数调用了forward

    # convolution

    input = torch.tensor([[1,2,0,3,1],
                        [0,1,2,3,1],
                        [1,2,1,0,0],
                        [5,2,3,1,1],
                        [2,1,0,1,1]])
    kernel = torch.tensor([[1,2,1],
                        [0,1,0],
                        [2,1,0]])
    input = input.reshape((1,1,5,5))
    kernel = kernel.reshape((1,1,3,3))
    print(input.shape,kernel.shape)
    output = F.conv2d(input=input,weight=kernel,stride=1)
    print(output)
    output = F.conv2d(input=input,weight=kernel,stride=1,padding=1)
    print(output)
    def myConv2d(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride,
            padding,
            dilation,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
        ) -> None:
        """2维卷积

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (_size_2_t): _description_
            stride (_size_2_t, optional): 步长,可以是单个数,也可以是元组分别控制横向和纵向步长. Defaults to 1.
            padding (Union[str, _size_2_t], optional): 填充宽度,可以是int/tuple. Defaults to 0.
            dilation (_size_2_t, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to True.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            dtype (_type_, optional): _description_. Defaults to None.
        """
        pass
def test():
    dataset = torchvision.datasets.CIFAR10("../download",train=False,transform=transforms.ToTensor(),download=True)
    dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=True)
    class Model(nn.Module):
        def __init__(self):
            super(Model,self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        def forward(self,input):
            return self.conv1(input)
    model = Model()
    print(model)
    for data in dataloader:
        imgs,targets = data
        output = model(imgs)
        print(imgs.shape,output.shape)
        break
if __name__ == "__main__":
    # main()
    test()
import torch
from torch import nn

def non_linear_activations():
    input = torch.arange(4).reshape(-1,1,2,2)
    print(input)
    class Model(nn.Module):
        def __init__(self):
            super(Model,self).__init__()
            # inplace = True 原地操作
            self.relu1 = nn.ReLU(inplace=False)
        def forward(self,input):
            return self.relu1(input)
        
    model = Model()
    print(model(input))
# normalize
# transformer
# linear y=kx+b
# dropout 防止过拟合
def linear():
    input = torch.arange(32*32,dtype=torch.float32).reshape(1,1,32,32)
    input = input.flatten()
    print(input.shape)
    output = nn.Linear(32*32,10)(input)
    print(output.shape)

if __name__ == "__main__":
    # non_linear_activations()
    linear()

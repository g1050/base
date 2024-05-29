import torch

def main():
    # print(dir(torch))
    shape = (3,4)
    ones = torch.ones(shape)
    print(ones)
    twe = torch.arange(12)
    print(twe.reshape(shape))
    print(f"total element num in twe is  {twe.numel()}")
    # 正态分布
    randn = torch.randn(1024)
    #样本太少导致生成的均值不为0,有误差
    print(randn.sum()/randn.numel(),randn.mean()) 
    print(randn.var(),randn.std())
    print(((randn-randn.mean())**2).sum()/randn.numel())
    print(randn)
    #从列表构造
    from_list = torch.tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
    print(from_list)

def test1():
    shape = (3,4)
    x = torch.arange(12,dtype=torch.float32).reshape(shape=shape)
    y = torch.arange(12,dtype=torch.float32).reshape(shape=shape)
    print(torch.cat((x,y),dim=0).shape)  # shape(6,4)
    y = y.reshape(12,1)
    x = x.reshape(1,12)
    print(y.shape)
    print((x+y).shape) # todo:广播机制 (12,12)

    x = x.reshape(shape)
    # 索引和切片
    print(x[-1])
    print(x[-1,-1])
    print(x[:,1:2])

    # 转numpy
    print(type(x.numpy()))

if __name__ == "__main__":
    # main()
    test1()


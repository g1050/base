# todo: n,c,h,w
# 沿着轴的最外侧可以访问一个最小批量

# 张量按照元素相乘为Hadmard乘积

# 范数,L2，L1范数,P范数 xi的P次方求和开p次根号

import torch
def main():
    #n,c,h,w （1,3，3,3）
    shape = (1,3,3,3)
    x = torch.arange(1*3*3*3).reshape(shape)
    print(x)
    print(x[0,0,:,:]) #第一通道
    print(x[0,0,0,:]) #第一通道的第一行
    print(x[0,1,:,0]) #第二通道的第二列

    print(x.sum(axis=0))
    y = torch.arange(2*3*3*3).reshape((2,3,3,3))
    print(y)
    print(y.sum(axis=0)) # 沿batch维度求和，第一个batch和第二个batch相加
    print(y[0,:,:,:])
    print(y.sum(axis=1),y.sum(axis=1).shape) #(2,3,3,3)
    print(y[:,0,:,:]) # 沿channel维度，每个batch的3个channel相加，就只剩下一个channel，达到降维
    print()
    print(y.sum(axis=2),y.sum(axis=2).shape) # 沿H维度，每个batch、channel中的的所有列(H)相加
    print(y[:,:,0,:])

    # 范数
    z = torch.tensor([3.0,4.0])
    # L2范数
    print(z.norm())
    print(torch.sqrt((z**2).sum()))
    # L1范数
    print(z.abs().sum())

if __name__ == "__main__":
    main()
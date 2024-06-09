# nn.functional 中包含Relu这样没有参数的计算

from torch  import nn
def layer_block(): 
    class MySequential(nn.Module):
        def __init__(self, *args):
            super().__init__()
            for idx, module in enumerate(args):
                # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
                # 变量_modules中。_module的类型是OrderedDict,有序字典
                self._modules[str(idx)] = module

        def forward(self, X):
            # OrderedDict保证了按照成员添加的顺序遍历它们
            for block in self._modules.values():
                X = block(X)
            return X
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    # net(X)
    print(net)

    class NestMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU())
            self.linear = nn.Linear(32, 16)

        def forward(self, X):
            return self.linear(self.net(X))

    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20))
    print(chimera)

if __name__ == "__main__":
    layer_block()
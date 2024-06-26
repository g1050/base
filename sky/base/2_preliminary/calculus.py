# import numpy as np
# from d2l import torch as d2l
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# def f(x):
#     return 3 * x ** 2 - 4 * x


# def numerical_lim(f, x, h):
#     return (f(x + h) - f(x)) / h


# h = 0.1
# for i in range(5):
#     print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
#     h *= 0.1

# x = np.arange(0, 3, 0.1)
# d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# d2l.plt.show()

# x = np.arange(0.5, 3, 0.2)
# d2l.plot(x, [x ** 3 - 1 / x, 4 * x - 4], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# d2l.plt.show()

import numpy as np
from d2l import torch as d2l
import os

def main():
    def f(x):
        return 3*x**2-4*x # grad: 6*x-4
    # 逼近求导数
    def numerical_lim(f,x,h):
        return (f(x+h)-f(x))/h

    h = 0.1
    for _ in range(5):
        print(f"h = {h:.5f},numerical limit = {numerical_lim(f,1,h)}")
        h *= 0.1 # h逐渐逼近
def test():
    x = np.arange(0,3,0.1) # (start,end,step) 左开右闭
    print(x)
    d2l.plot(x, [x ** 3 - 1 / x, 4 * x - 4], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
    d2l.plt.savefig('data/img.png')
if __name__ == "__main__":
    # main()
    test()
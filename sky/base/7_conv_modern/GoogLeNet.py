# Inception 盗梦空间
# 含并行连结的网络

# Inception块，不改变输入和输出的宽高，最后把所有通道连接在一起
# Inception可以计算更少，复杂度低，提取的信息更丰富

# Inception V1、V2、V3 +BN、+残差
# Inception V3使用广泛，其中使用了soft label

# GoogleNet使用了9个Inception，第一个达到上白层的网络

# 思想可以参考，但是实际上的各种设计应该是Google跑出来的效果最好的结构，没太多道理
# 模型从上往下，宽高减半减半，通道数变变变多

# d2l dive into deep learning


# 分布式计算不能让瓶颈出现在通信上，不管是机器之间通信还是cpu和gpu之间的通信

# 性能的权衡：增大bs可以提高系统性能，但是增大bs训练的的收敛就会变差

# 计算/通讯 计算通信比越大越好
# 跨机器一般都是数据并行，GPT3内部单机模型并行，多机还是使用数据并行

# 训练集有n个类别,bs最好不要超过10*n
# atten是一个layer，是一个机制
# transformer是一种架构，区别去seq2seq中间插入attention，transformer全部使用self-attn

# 多头注意力机制
# 多个注意力，然后concat起来
# 类似多通道，每个头提取不同的信息
# 为了避免多头的for loop拼成一个矩阵做一次运算再拆开

# todo：BN和LN区别

# transformer block
# encode block：self_attention + add/norm + ffn + add/norm
# decode block: 

#  机器学习发展到现在，就是大牛换个主干网络。小牛做主干网络的下游任务。小小牛就把前一个主干网络的下游任务的trick用在这个主干网络的下游任务。
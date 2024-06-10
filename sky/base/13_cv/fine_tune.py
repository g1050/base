# 网络架构，特征提取 + 线性分类
# 核心思想：利用预训练模型，对提取的特征对模型进行微调，加上自己的线性分类层
# 可以固定住一些层的参数不进行更新避免overfitting的可能性

# 为什么需要归一化？因为使用的在ImageNet上的预训练模型使用了这些参数

# 训练代码：最后一层的学习率和pretrain中的参数不同，采用不同的学习率lr*10
# 微调：特征模块微调，最后分类器重新学习

# fine tunning = transfer learning

# lr_period,lr_decay = 4,0.9 每隔4个epoch后lr*0.9

# weight decay 是正则化，控制模型复杂度
# lr decay 控制在最优解附近时候，学习率减小，不会大幅度跳跃
# 除了上述方法，还有cosine控制学习率，余弦退火衰减
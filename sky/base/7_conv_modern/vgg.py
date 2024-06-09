# vgg: 相同计算量下，多个3*3效果优于少量5*5
# vgg块 多个块串联vgg-16 vgg-19，相当于封装了3个卷积+maxpool
# vgg： 封装的AlexNet

# 经典的设计：宽高减半，通道数翻倍
# vgg计算代价更大，但是效果比AlexNet好
# 单机多卡并行
# 常见的切分方案：数据并行、
# 模型并行，模型大到一个GPU放不下的时候
# 当一个模型能用单卡计算的时候，通过数据并行用单机多卡计算
# 当一个模型很大的时候(transformer)，只能模型并行了

# 小批量多GPU是直接将多个GPU的梯度叠加在一起的

# 分布式中通信的操作
# all_reduce 把所有梯度加在一个gpu上，再赋值回去
# scatter 分发

# batch_size小的时候lr不能太大
def multi_gpus_zero():
    pass

def multi_gpus_concise():
    pass

if __name__ == "__main__":
    multi_gpus_zero()
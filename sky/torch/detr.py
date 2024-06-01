import torch
import os
# TORCH_HOME 来指定模型权重文件的下载位置
# 默认情况下，PyTorch 会将下载的模型权重保存在 ~/.cache/torch/hub 目录下
# DETR DE tection TR ansformer
os.environ['TORCH_HOME'] = os.path.realpath('../download')
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
print(model)
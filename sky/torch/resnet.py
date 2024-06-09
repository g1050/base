import torchvision
import torch
model = torchvision.models.resnet50(pretrained=False)
print(model)
torch.save(model,"data/resnet50.pth")
# # 创建输入张量
dummy_input = torch.randn(1, 3, 224, 224)
# 导出模型为ONNX格式
torch.onnx.export(model, dummy_input, "resnet50.onnx",
                  input_names=['input'], output_names=['output'],
                  opset_version=11)
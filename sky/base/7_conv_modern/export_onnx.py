
def export_onnx(net,dummy_input,pth_path,onnx_path):
    import onnx
    from onnx import shape_inference
    import torch
    torch.save(net,pth_path)
    # 导出模型为ONNX格式
    torch.onnx.export(net, dummy_input, onnx_path,
                    input_names=['input'], output_names=['output'],
                    opset_version=11)
    print("Export successfully.")
    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    onnx.save(model, onnx_path)
    print("Shape infer successfully.")
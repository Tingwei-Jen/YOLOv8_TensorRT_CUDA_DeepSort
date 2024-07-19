import argparse
import torch
import torchvision
import onnx
from model import ResNet18Reid

def convert(args):
    model = ResNet18Reid(num_classes=1501, reid=True)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # dummy input
    dummy_input = torch.randn(1, 3, 128, 64)

    # # 將 dummy input 傳入模型
    # output = model(dummy_input)
    # # check model output shapes
    # print("model output shape: ", output.shape) 

    # transform model to onnx
    torch.onnx.export(model, dummy_input, args.output_onnx_path, 
                      export_params=True, 
                      opset_version=17, 
                      do_constant_folding=True,
                      input_names=["input"], 
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"},  # 動態批次大小
                                    "output": {0: "batch_size"}} # 動態批次大小
    )

def check_dynamic_shapes(model_path):
    # 加載 ONNX 模型
    model = onnx.load(model_path)

    # 檢查模型的輸入和輸出
    for input in model.graph.input:
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f"Input '{input.name}' shape: {shape}")

    for output in model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"Output '{output.name}' shape: {shape}")

    # 判斷是否有動態形狀
    has_dynamic_shapes = any(dim.dim_value == 0 for input in model.graph.input for dim in input.type.tensor_type.shape.dim)
    if has_dynamic_shapes:
        print("The model has dynamic shapes.")
    else:
        print("The model has static shapes.")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Resnet18Reid to onnx")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='check_point path, default: None')
    parser.add_argument('--output_onnx_path', type=str, default='model.onnx', help='output onnx model path, default is model.onnx')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    convert(args)
    check_dynamic_shapes(args.output_onnx_path)
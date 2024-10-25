import torch.onnx
import torchvision.models as models

batch_size = 16
device = torch.device("cuda")

def convert2onnx(resnet):
    resnet.eval()
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(
        resnet,
        dummy_input,
        "resnet18.onnx",
        verbose=True,
        opset_version=13,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        do_constant_folding=True, 
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
convert2onnx(resnet)
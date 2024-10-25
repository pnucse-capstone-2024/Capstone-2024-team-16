import onnxruntime
import onnx
import numpy as np
import torchvision.models as models
import torch
import os
from time import time

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

device = 'cpu'

model_onnx_path = os.path.join(".", "resnet18.onnx")
so = onnxruntime.SessionOptions()
exproviders = ["CPUExecutionProvider"]

ort_session = onnxruntime.InferenceSession(model_onnx_path, so, providers=exproviders)

# ONNX 런타임에서 계산된 결과값
x = torch.randn(1, 3, 224, 224, requires_grad=True, dtype=torch.float32, device=device) #

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
pre_time = time()
ort_outs = ort_session.run(None, ort_inputs)
print("ONNX Runtime processing time : ", time()-pre_time)

resnet = (models.resnet18(weights=models.ResNet18_Weights.DEFAULT)).to(device=device)

pre_time = time()
torch_out = resnet(x)
print("Pytorch processing time : ", time()-pre_time)

# 기존 output과 onnx output 값 차이 비교
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-02, atol=1e-02)
import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the model
torch_model = MyModel()
torch_model.eval()  # Set the model to inference mode

# Prepare the input tensor
torch_input = torch.randn(1, 1, 32, 32)

# Specify the path where you want to save the ONNX model
onnx_path = "my_model.onnx"

# Export the model
torch.onnx.export(torch_model,                # model being run
                  torch_input,                # model input (or a tuple for multiple inputs)
                  onnx_path,                  # where to save the model (can be a file or file-like object)
                  export_params=True,         # store the trained parameter weights inside the model file
                  opset_version=11,           # the ONNX version to export the model to
                  do_constant_folding=True,   # whether to execute constant folding for optimization
                  input_names=['input'],      # the model's input names
                  output_names=['output'],    # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})

print("Model was successfully exported to ONNX format.")

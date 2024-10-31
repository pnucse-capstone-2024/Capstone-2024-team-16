import onnx
onnx_model = onnx.load("my_model.onnx")
onnx.checker.check_model(onnx_model)
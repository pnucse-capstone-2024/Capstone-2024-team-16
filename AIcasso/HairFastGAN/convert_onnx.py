import torch
import numpy as np
from models.stylegan2.model import Generator  # Adjust import path as necessary

# Parameters for the Generator
size = 1024  # The output resolution of the generator
style_dim = 512  # The dimensionality of the style vector
n_mlp = 8  # Number of layers in the mapping network
channel_multiplier = 2  # Channel multiplier for the generator

# Initialize the Generator
generator = Generator(size, style_dim, n_mlp, channel_multiplier=channel_multiplier)

# Load the model weights
# Ensure the path to your .pt file is correct and the key for the state dict is appropriate
model_path = 'pretrained_models/StyleGAN/ffhq.pt'  # Update this path to where your model weights are stored
weights = torch.load(model_path, map_location='cpu')['g_ema']  # 'g_ema' is the key for the generator model weights
generator.load_state_dict(weights)
generator.eval()

# Create a dummy latent vector as input for the ONNX export
# This vector simulates the input the generator expects
dummy_latent = torch.randn(1, style_dim)  # Batch size of 1, style_dim dimensions

# Export the generator to ONNX
output_onnx_path = 'generator.onnx'  # Path where the ONNX model will be saved
torch.onnx.export(generator, dummy_latent, output_onnx_path,
                  export_params=True, opset_version=12,
                  input_names=['latent'], output_names=['output'],
                  dynamic_axes={'latent': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print(f"ONNX model has been saved to {output_onnx_path}")

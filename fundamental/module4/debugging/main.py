import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import SqueezeNet
from pathlib import Path

import helper_utils

from simple_cnn import SimpleCNNDebug, SimpleCNN, SimpleCNN2Seq, SimpleCNN2SeqDebug

path_dataset = Path.cwd() / "data/FashionMNIST_data"

dataset = helper_utils.get_dataset(path_dataset)

transform = transforms.ToTensor()
dataset.transform = transform

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

img_batch, label_batch = next(iter(dataloader))
print("Batch shape:", img_batch.shape)  # Should be [batch_size, 1, 28, 28]

print("-"*45)

simple_cnn = SimpleCNN()

try:
    output = simple_cnn(img_batch)  
except Exception as e:
    print(f"\033[91mError during forward pass: {e}\033[0m")

print("-"*45)

simple_cnn_debug = SimpleCNNDebug()

try:
    output = simple_cnn_debug(img_batch)  
except Exception as e:
    print(f"\033[91mError during forward pass: {e}\033[0m")

print("-"*45)

simple_cnn_seq = SimpleCNN2Seq()

try:
    output = simple_cnn_seq(img_batch)  
except Exception as e:
    print(f"\033[91mError during forward pass: {e}\033[0m")

print("-"*45)
simple_cnn_seq_debug = SimpleCNN2SeqDebug()

try:
    output = simple_cnn_seq_debug(img_batch)  
except Exception as e:
    print(f"\033[91mError during forward pass: {e}\033[0m")

print("-"*45)
simple_cnn_seq_debug = SimpleCNN2SeqDebug()

for idx, (img_batch, _) in enumerate(dataloader):
    if idx < 5:
        print(f"=== Batch {idx} ===")
        output_debug = simple_cnn_seq_debug(img_batch)

print("-"*45)

# Load SqueezeNet model
complex_model = SqueezeNet()

print(complex_model)

print("-"*45)

# Iterate through the main blocks
for name, block in complex_model.named_children():
    print(f"Block {name} has a total of {len(list(block.children()))} layers:")
    
    # List all children layers in the block
    for idx, layer in enumerate(block.children()):
        # Check if the layer is terminal (no children) or not
        if len(list(layer.children())) == 0:
            print(f"\t {idx} - Layer {layer}")
        # If the layer has children, it's a sub-block, then print only the number of children and its name
        else:
            layer_name = layer._get_name()  # More user-friendly name
            print(f"\t {idx} - Sub-block {layer_name} with {len(list(layer.children()))} layers")          


print("-"*45)

first_fire_module = complex_model.features[3]

for idx, module in enumerate(first_fire_module.modules()):
    # Avoid printing the top-level module itself
    if idx > 0 :
        print(module)


print("-"*45)
type_layer = nn.Conv2d

selected_layers = [layer for layer in complex_model.modules() if isinstance(layer, type_layer)]

print(f"Number of {type_layer.__name__} layers: {len(selected_layers)}")

print("-"*45)
# total number of parameters in the model
total_params = sum(p.numel() for p in complex_model.parameters())
print(f"Total number of parameters in the model: {total_params}")

print("-"*45)
counting_params = {}
# For each terminal layer print its number of parameters
for layer in complex_model.named_modules():
    n_children = len(list(layer[1].children()))
    if n_children == 0:  # Terminal layer
        layer_name = layer[0]
        n_parameters = sum(p.numel() for p in layer[1].parameters())
        counting_params[layer_name] = n_parameters
        print(f"Layer {layer_name} has {n_parameters} parameters")

# Plotting the distribution of parameters per layer
helper_utils.plot_counting(counting_params)
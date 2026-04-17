"""
Lesson 1 - Module 4: Debugging Neural Networks (debugging/main.py)
==================================================================
WHAT YOU'LL LEARN:
  * Debugging shape mismatches in CNN forward passes
  * Using debug versions of models that print intermediate tensor shapes
  * Inspecting layer parameters (weight and bias shapes)
  * Exploring complex model architectures (SqueezeNet)
  * Counting parameters per layer to understand model complexity

KEY CONCEPT:
  The #1 bug in deep learning is shape mismatches. When you see
  "RuntimeError: mat1 and mat2 shapes cannot be multiplied", this module
  shows you how to debug it by printing shapes at every layer.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import SqueezeNet
from pathlib import Path

import helper_utils

from simple_cnn import SimpleCNNDebug, SimpleCNN, SimpleCNN2Seq, SimpleCNN2SeqDebug


# ==================== STEP 1: LOAD DATA ====================
# FashionMNIST: 28x28 grayscale images of clothing items (10 classes)

path_dataset = Path.cwd() / "data/FashionMNIST_data"
dataset = helper_utils.get_dataset(path_dataset)

transform = transforms.ToTensor()
dataset.transform = transform

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Get one batch to test with
img_batch, label_batch = next(iter(dataloader))
print("Batch shape:", img_batch.shape)  # [64, 1, 28, 28] = [batch, channels, height, width]


# ==================== STEP 2: DEBUG SHAPE MISMATCHES ====================
# KEY CONCEPT: When building a model, run a dummy batch through it early.
# If shapes are wrong, the error message tells you exactly where it failed.

# Test 1: SimpleCNN -- this works correctly
simple_cnn = SimpleCNN()
try:
    output = simple_cnn(img_batch)
except Exception as e:
    print(f"Error during forward pass: {e}")

# Test 2: SimpleCNNDebug -- prints shapes at every layer
# This is a debugging technique: subclass your model and add print statements
# to the forward() method to see exactly where shapes go wrong.
simple_cnn_debug = SimpleCNNDebug()
try:
    output = simple_cnn_debug(img_batch)
    # OBSERVE the output: you'll see shapes like:
    #   Input:         [64, 1, 28, 28]
    #   After Conv+ReLU: [64, 32, 28, 28]  (channels: 1->32, size preserved by padding)
    #   After Pool:      [64, 32, 14, 14]  (spatial halved by MaxPool)
    #   After Flatten:   [64, 6272]         (32 * 14 * 14 = 6272)
    #   After FC1:       [64, 128]
    #   After FC2:       [64, 10]            (10 classes)
except Exception as e:
    print(f"Error during forward pass: {e}")


# ==================== STEP 3: SEQUENTIAL VERSION WITH DEBUGGING ====================
# nn.Sequential wraps layers into a clean container.
# The debug version adds activation statistics (mean, std, min, max) at each stage.

simple_cnn_seq = SimpleCNN2Seq()
try:
    output = simple_cnn_seq(img_batch)
except Exception as e:
    print(f"Error during forward pass: {e}")

# Debug version with statistics
simple_cnn_seq_debug = SimpleCNN2SeqDebug()
try:
    output = simple_cnn_seq_debug(img_batch)
    # OBSERVE: The debug version prints statistics about activations.
    # This is useful for detecting:
    #   - Vanishing activations (mean near 0, std near 0)
    #   - Exploding activations (very large values)
    #   - Dead ReLU (all values are 0)
except Exception as e:
    print(f"Error during forward pass: {e}")

# Run through multiple batches to verify consistency
for idx, (img_batch, _) in enumerate(dataloader):
    if idx < 5:
        print(f"=== Batch {idx} ===")
        output_debug = simple_cnn_seq_debug(img_batch)


# ==================== STEP 4: EXPLORE A COMPLEX MODEL (SqueezeNet) ====================
# KEY CONCEPT: Real-world models are complex nested structures.
# Understanding how to navigate them is essential for:
#   - Transfer learning (freezing/replacing specific layers)
#   - Debugging
#   - Understanding what the model is doing

complex_model = SqueezeNet()
print(complex_model)

# Navigate the model's structure using .named_children()
for name, block in complex_model.named_children():
    print(f"Block '{name}' has {len(list(block.children()))} layers:")

    for idx, layer in enumerate(block.children()):
        if len(list(layer.children())) == 0:
            # Terminal layer (Conv2d, ReLU, etc.) -- has no sub-layers
            print(f"  {idx} - Layer {layer}")
        else:
            # Sub-block (contains more layers inside)
            layer_name = layer._get_name()
            print(f"  {idx} - Sub-block {layer_name} with {len(list(layer.children()))} layers")

# Dive into a specific sub-module (e.g., the first Fire module)
first_fire_module = complex_model.features[3]
print("\n--- First Fire Module internals ---")
for idx, module in enumerate(first_fire_module.modules()):
    if idx > 0:  # Skip the top-level module itself
        print(module)


# ==================== STEP 5: COUNT PARAMETERS ====================
# KEY CONCEPT: The number of parameters tells you:
#   - How much memory the model needs
#   - How much training data you need (more params = need more data)
#   - How fast inference will be

# Filter for specific layer types
type_layer = nn.Conv2d
selected_layers = [layer for layer in complex_model.modules() if isinstance(layer, type_layer)]
print(f"\nNumber of {type_layer.__name__} layers: {len(selected_layers)}")

# Total parameters
total_params = sum(p.numel() for p in complex_model.parameters())
print(f"Total number of parameters in the model: {total_params:,}")

# Per-layer parameter count
counting_params = {}
for layer in complex_model.named_modules():
    n_children = len(list(layer[1].children()))
    if n_children == 0:  # Terminal layer only (skip containers)
        layer_name = layer[0]
        n_parameters = sum(p.numel() for p in layer[1].parameters())
        counting_params[layer_name] = n_parameters
        print(f"Layer {layer_name} has {n_parameters:,} parameters")

# Visualize the parameter distribution across layers
helper_utils.plot_counting(counting_params)

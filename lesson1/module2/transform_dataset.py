import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import helper_utils
from pathlib import Path


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU")

data_path = Path.cwd() / "./module2/data"

# Compute MNIST mean and std dynamically (over the entire training set)
# This mirrors your delivery example: calculate them yourself instead of hardcoding

temp_transform = transforms.Compose([
    transforms.ToTensor()  # Only this — scales pixels to [0.0, 1.0]
])

temp_dataset = torchvision.datasets.MNIST(
    root=data_path,
    train=True,
    download=True,
    transform=temp_transform
)

# CHANGE: num_workers=0 to avoid multiprocessing error on macOS
loader = DataLoader(temp_dataset, batch_size=128, shuffle=False, num_workers=0)

channel_sum = 0.0
channel_sum_squared = 0.0
total_pixels = 0

for images, _ in loader:
    batch_pixels = images.size(0) * 28 * 28
    total_pixels += batch_pixels
    
    channel_sum += images.sum(dim=[0, 2, 3])
    channel_sum_squared += (images ** 2).sum(dim=[0, 2, 3])

computed_mean = channel_sum / total_pixels
computed_std = torch.sqrt(channel_sum_squared / total_pixels - computed_mean ** 2)

print(f"Computed Mean: {computed_mean.item():.6f}")  # ~0.1307
print(f"Computed Std:  {computed_std.item():.6f}")   # ~0.3081

# Use these in your final transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((computed_mean.item(),), (computed_std.item(),))
])
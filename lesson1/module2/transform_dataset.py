"""
Lesson 1 - Module 2: Transform & Normalize Datasets (transform_dataset.py)
===========================================================================
 WHAT YOU'LL LEARN:
  • How to COMPUTE mean and std for a dataset (instead of using hardcoded values)
  • Why normalization matters for training stability
  • How to use a temporary DataLoader to compute dataset statistics

 KEY CONCEPT:
  In digit_detective.py, we used hardcoded MNIST mean (0.1307) and std (0.3081).
  But for YOUR OWN datasets, you need to compute these values yourself.
  This script shows you exactly how to do that.

 THE PROCESS:
  1. Load images with ToTensor() only (scales to [0, 1])
  2. Iterate through ALL images, accumulating sum and sum-of-squares
  3. Compute: mean = sum / N,  std = sqrt(sum_squared/N - mean²)
"""

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


# ==================== DEVICE SELECTION ====================
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


# ==================== STEP 1: LOAD DATASET WITH MINIMAL TRANSFORM ====================
# Use ONLY ToTensor() — no normalization yet! We need raw [0, 1] pixel values
# to compute the dataset's actual mean and standard deviation.
temp_transform = transforms.Compose([
    transforms.ToTensor()  # PIL Image → Tensor, pixel values scaled to [0.0, 1.0]
])

temp_dataset = torchvision.datasets.MNIST(
    root=data_path,
    train=True,
    download=True,
    transform=temp_transform
)

#  NOTE: num_workers=0 to avoid multiprocessing issues on macOS
loader = DataLoader(temp_dataset, batch_size=128, shuffle=False, num_workers=0)


# ==================== STEP 2: COMPUTE MEAN AND STD ====================
#  CONCEPT: We need to compute statistics over ALL pixels in the dataset.
#   For a single-channel (grayscale) image dataset:
#     mean = (sum of all pixels) / (total number of pixels)
#     std  = sqrt( (sum of pixel²) / N - mean² )
#
#   This uses Welford's online algorithm idea: accumulate running totals.

channel_sum = 0.0           # Sum of all pixel values
channel_sum_squared = 0.0   # Sum of all pixel values squared
total_pixels = 0            # Total pixel count

for images, _ in loader:
    # Each batch has shape: [batch_size, channels, height, width]
    # images.size(0) = batch_size, 28×28 = pixels per image
    batch_pixels = images.size(0) * 28 * 28
    total_pixels += batch_pixels

    # Sum across all images, spatial dimensions — keep channel dimension
    # dim=[0, 2, 3] means: sum over batch, height, and width
    channel_sum += images.sum(dim=[0, 2, 3])

    # Sum of squares (needed for variance calculation)
    channel_sum_squared += (images ** 2).sum(dim=[0, 2, 3])

# Calculate mean: total sum / total count
computed_mean = channel_sum / total_pixels

# Calculate std: sqrt(E[X²] - E[X]²) — this is the standard deviation formula
computed_std = torch.sqrt(channel_sum_squared / total_pixels - computed_mean ** 2)

print(f"Computed Mean: {computed_mean.item():.6f}")  # Should be ~0.1307
print(f"Computed Std:  {computed_std.item():.6f}")   # Should be ~0.3081


# ==================== STEP 3: USE COMPUTED VALUES IN TRANSFORM ====================
# Now that we know the actual mean and std, create the proper transform pipeline.
transform = transforms.Compose([
    transforms.ToTensor(),                                       # Scale to [0, 1]
    transforms.Normalize((computed_mean.item(),), (computed_std.item(),))  # Standardize
])

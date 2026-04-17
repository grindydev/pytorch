"""
Lesson 1 - Module 4: CNN Building Blocks (cnn_block.py)
========================================================
WHAT YOU'LL LEARN:
  * Building a reusable CNN block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
  * Stacking CNN blocks into a full CNN model (SimpleCNN)
  * How spatial dimensions shrink through convolution + pooling layers
  * Using Dropout to prevent overfitting
  * Computing flattened feature size for the fully-connected classifier

KEY CONCEPT:
  A CNN (Convolutional Neural Network) is THE architecture for image tasks.
  Unlike Dense networks (which flatten everything first), CNNs preserve spatial
  structure by sliding small filters across the image to detect patterns:
  edges -> textures -> object parts -> whole objects.

  Conv2d:  Learns local patterns (edges, corners, textures)
  BatchNorm: Stabilizes training by normalizing layer outputs
  ReLU:     Adds non-linearity (sets negatives to 0)
  MaxPool:  Shrinks spatial size, keeps the strongest features
  Dropout:  Randomly zeros neurons during training (prevents overfitting)
"""

import torch
import torch.nn as nn


# ==================== CNNBlock: A Reusable Building Block ====================
# KEY CONCEPT: Instead of writing each layer separately, we package
# Conv -> BatchNorm -> ReLU -> MaxPool into ONE reusable block.
# This makes the model cleaner and easier to modify.

class CNNBlock(nn.Module):
    """
    A single convolutional block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d

    WHAT EACH LAYER DOES:
      Conv2d:      Slides learnable filters across the image to detect features.
                   Output channels = number of different filters/patterns.
      BatchNorm2d: Normalizes outputs across the batch. This stabilizes training
                   and allows higher learning rates. (Think: "standardize each feature map")
      ReLU:        max(0, x) -- introduces non-linearity. Without this, stacking
                   layers is just a single linear transformation.
      MaxPool2d:   Takes the maximum value in each 2x2 window. Reduces spatial
                   dimensions by half (e.g., 32x32 -> 16x16), keeping the
                   strongest feature activations.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        """
        Args:
            in_channels:  Number of input channels (3 for RGB, or output of previous block)
            out_channels: Number of output channels (number of filters to learn)
            kernel_size:  Size of the convolution filter (3 means 3x3)
            padding:      Pixels added around the input border (padding=1 with kernel=3
                          preserves spatial dimensions BEFORE pooling)
        """
        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            # Conv2d: The core operation. Each filter slides across the image.
            # With kernel=3, padding=1, stride=1: spatial size stays the same.
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding
            ),
            # BatchNorm2d: Normalizes each feature map to have ~mean=0, ~std=1.
            # Helps prevent "internal covariate shift" -- layers don't have to
            # constantly adapt to changing input distributions.
            nn.BatchNorm2d(num_features=out_channels),
            # ReLU: Non-linear activation. Fast, simple, works well.
            nn.ReLU(),
            # MaxPool2d: Downsamples by taking the max in each 2x2 window.
            # Halves both height and width. Reduces computation and helps
            # the model focus on the most prominent features.
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.block(x)
        return x


# ==================== SimpleCNN: The Full Model ====================
# ARCHITECTURE:
#   Input [3, 32, 32]  (CIFAR-100: 3 color channels, 32x32 pixels)
#     -> CNNBlock1: 3 -> 32 channels,   32x32 -> 16x16 (MaxPool halves)
#     -> CNNBlock2: 32 -> 64 channels,  16x16 -> 8x8
#     -> CNNBlock3: 64 -> 128 channels,  8x8  -> 4x4
#     -> Flatten: 128 * 4 * 4 = 2048 features
#     -> Linear(2048, 512) -> ReLU -> Dropout(0.6)
#     -> Linear(512, num_classes)
#
# KEY INSIGHT: Each block doubles channels but halves spatial size.
#   Total features before classifier: channels * height * width = 128 * 4 * 4

class SimpleCNN(nn.Module):
    """
    A 3-block CNN for image classification.

    WHY CNNs BEAT DENSE NETWORKS FOR IMAGES:
      - Dense networks flatten the image, losing spatial relationships
      - CNNs preserve spatial structure and detect local patterns
      - Parameter sharing: one filter is applied across ALL positions
        (far fewer parameters than a dense layer over all pixels)
    """

    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Three convolutional blocks with increasing channel depth
        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)    # RGB -> 32 features
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)   # 32 -> 64 features
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)  # 64 -> 128 features

        # Fully-connected classifier
        # Input size: 128 channels * 4 height * 4 width = 2048
        # (Input is 32x32, halved 3 times by MaxPool: 32->16->8->4)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),     # [batch, 128, 4, 4] -> [batch, 2048]
            nn.Linear(4 * 4 * 128, 512), # 2048 -> 512 hidden neurons
            nn.ReLU(),                    # Non-linearity
            nn.Dropout(p=0.6),            # Randomly zeros 60% of neurons during training
            nn.Linear(512, num_classes)   # 512 -> num_classes (one score per class)
        )

    def forward(self, x):
        # Pass through each convolutional block
        x = self.conv_block1(x)  # [batch, 3, 32, 32]  -> [batch, 32, 16, 16]
        x = self.conv_block2(x)  # [batch, 32, 16, 16] -> [batch, 64, 8, 8]
        x = self.conv_block3(x)  # [batch, 64, 8, 8]   -> [batch, 128, 4, 4]

        # Flatten and classify
        x = self.classifier(x)   # [batch, 128, 4, 4]  -> [batch, num_classes]
        return x

# Learning Guide: Lesson 1, Module 4 -- CNN Basics and Image Classification

## Module Overview

Dense networks flatten images to 1D vectors, losing spatial information.
CNNs (Convolutional Neural Networks) process images as 2D grids, detecting
patterns like edges, textures, and objects. This module teaches you how to
build, debug, and train CNNs for real image classification tasks.

## Recommended Reading Order

1. **cnn/cnn_block.py** -- The building block: Conv2d + BatchNorm + ReLU + MaxPool
2. **cnn/main.py** -- Stacking blocks into a full CNN, training on CIFAR-100
3. **debugging/simple_cnn.py** -- Debug versions that print shapes and stats
4. **debugging/main.py** -- Debugging shape mismatches, exploring model architecture
5. **nature_classification/main.py** -- End-to-end CNN pipeline from scratch

## Concept Map

```
CNN Building Block (cnn_block.py)
   |
   +--> Conv2d: learnable filters detect local patterns
   +--> BatchNorm2d: stabilize training
   +--> ReLU: non-linearity
   +--> MaxPool2d: downsample, keep strongest features
   |
   v
SimpleCNN: stack 3 blocks + classifier
   |
   +--> Block 1: 3 -> 32 channels,  32x32 -> 16x16
   +--> Block 2: 32 -> 64 channels, 16x16 -> 8x8
   +--> Block 3: 64 -> 128 channels, 8x8 -> 4x4
   +--> Flatten -> Linear(2048, 512) -> Dropout -> Linear(512, classes)
   |
   v
Training: transforms, augmentation, best-model checkpointing
   |
   v
Debugging: print shapes, activation statistics, parameter counts
```

## File Summaries

### cnn_block.py
Defines CNNBlock (reusable Conv+BN+ReLU+Pool) and SimpleCNN (3 blocks + classifier).
Focus on: how spatial dimensions shrink (32->16->8->4) while channels grow
(3->32->64->128), and why the flattened size is 128*4*4=2048.

### cnn/main.py
Full CNN training pipeline on CIFAR-100 (15-class subset).
Introduces: separate train/val transforms, weight_decay (L2 regularization),
best-model checkpointing (save model with highest validation accuracy).
Focus on: the training_loop function and why we save the best model state.

### simple_cnn.py (debugging/)
Four versions of the same CNN:
- SimpleCNN: clean version
- SimpleCNNDebug: prints shapes at every layer
- SimpleCNN2Seq: uses nn.Sequential
- SimpleCNN2SeqDebug: adds activation statistics (mean, std, min, max)
Focus on: how debug versions subclass and override forward() to add print statements.

### debugging/main.py
Shows how to debug shape mismatches (the #1 bug in deep learning) and how to
explore complex model architectures (SqueezeNet). Introduces parameter counting.
Focus on: running a dummy batch through the model early to catch shape errors.

### nature_classification/main.py
End-to-end CNN built from scratch (no CNNBlock helper). Shows the prototyping
pattern: start with a small subset (9 classes), then scale to full dataset (15 classes).
Focus on: the complete pipeline from data loading to prediction visualization.

## Common Questions

**Q: Why use CNNs instead of dense networks for images?**
A: Dense networks flatten the image, losing spatial relationships (which pixels
are next to which). CNNs slide small filters across the image, preserving spatial
structure. They also share parameters (one filter applied everywhere), needing
far fewer weights than a dense layer over all pixels.

**Q: What is BatchNorm and why is it needed?**
A: BatchNorm normalizes each feature map to have mean~0 and std~1 across the
batch. This stabilizes training because later layers don't have to constantly
adapt to changing input distributions (called "internal covariate shift").
It lets you use higher learning rates and train faster.

**Q: What is Dropout?**
A: During training, Dropout randomly zeros out a fraction of neurons (e.g., 50%).
This prevents the model from relying too heavily on any single neuron and
reduces overfitting. During evaluation, Dropout is disabled.

**Q: What is weight_decay?**
A: Also called L2 regularization. It adds a penalty for large weights to the
loss function, encouraging the model to use smaller, more distributed weights.
This reduces overfitting. In Adam, this is called AdamW.

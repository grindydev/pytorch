# Learning Guide: Lesson 2, Module 2 -- Transfer Learning and Data Preprocessing

## Module Overview

Training from scratch requires huge datasets and long training times. Transfer
learning lets you start with a model already trained on millions of images and
adapt it to YOUR task with minimal additional training. This is how most real-
world ML projects work.

## Recommended Reading Order

1. **datasets/main.py** -- Quick tour of built-in torchvision datasets
2. **pre_processing/main.py** -- Image preprocessing and augmentation techniques
3. **transfer_learning/main.py** -- The three transfer learning strategies

## Concept Map

```
Pre-trained Model (trained on ImageNet, 14M images)
   |
   +--> Already knows: edges, textures, shapes, object parts
   |
   v
Three Strategies:
   |
   +--> Feature Extraction: freeze all, train new head only
   |    Best for: small dataset, similar task
   |
   +--> Fine-Tuning: unfreeze top layers, low LR
   |    Best for: medium dataset, somewhat different task
   |
   +--> Full Retraining: unfreeze everything, moderate LR
   |    Best for: large dataset, very different task
   |
   v
Key Operations:
   +--> requires_grad = False (freeze)
   +--> Replace final layer for new num_classes
   +--> filter(lambda p: p.requires_grad, model.parameters()) for optimizer
```

## File Summaries

### datasets/main.py
Quick tour of loading built-in datasets (CIFAR-10) and applying transforms.
Short and simple -- just shows the basic pattern.
Focus on: how datasets return (PIL_image, label) tuples.

### pre_processing/main.py
Comprehensive tour of image preprocessing: resizing, cropping, flipping,
rotation, color jitter, normalization. Shows Oxford-IIIT Pet dataset.
Focus on: the difference between PIL transforms and tensor transforms.

### transfer_learning/main.py
The core file. Demonstrates all three transfer learning strategies using
MobileNetV3 and ResNet18 on EMNIST data.
Focus on: the freezing pattern (requires_grad=False), replacing the final
layer, and why fine-tuning uses a much lower learning rate.

## Common Questions

**Q: Why does transfer learning work?**
A: Early layers in image models learn universal features (edges, corners,
textures) useful for ANY image task. Only the later layers need task-specific
adjustments. By reusing the early layers, you need far less data and time.

**Q: Why use a lower LR for fine-tuning?**
A: The pre-trained weights are already good. You want small adjustments, not
large changes that destroy the learned features. A learning rate 10-100x
smaller than the initial training is typical.

**Q: How do I know which strategy to use?**
A: If you have less than 1000 images per class, use Feature Extraction.
If you have 1000-10000, try Fine-Tuning. If you have 10000+, Full Retraining
may work best. Always start with Feature Extraction as a baseline.

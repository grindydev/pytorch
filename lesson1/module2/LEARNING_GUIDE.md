# Learning Guide: Lesson 1, Module 2 -- Datasets, Transforms, and Image Classification

## Module Overview

Now that you know tensors and basic training, this module teaches you how to
work with IMAGE data and build your first image classifier. You learn the
standard dataset pipeline: load data, apply transforms, create batches, and
train a model to recognize handwritten digits and letters.

## Recommended Reading Order

1. **digit_detective.py** -- Full pipeline: MNIST dataset, transforms, DNN, training loop
2. **transform_dataset.py** -- How to compute dataset statistics from scratch
3. **letter_detective.py** -- Harder problem: 26-class EMNIST with a deeper network

## Concept Map

```
Raw Data (images on disk)
   |
   v
torchvision.datasets (built-in dataset loaders)
   |
   v
Transforms (ToTensor, Normalize, augmentation)
   |
   v
DataLoader (batching + shuffling)
   |
   v
Model (Dense Neural Network for images)
   |
   +--> Flatten image to 1D vector
   +--> Linear layers + ReLU
   +--> CrossEntropyLoss for multi-class
   +--> Adam optimizer
   |
   v
Training Loop (same 5-step pattern from Module 1)
```

## File Summaries

### digit_detective.py
Your first image classifier. Uses MNIST handwritten digits (0-9).
Introduces: torchvision datasets, transforms pipeline, DataLoader for batching,
CrossEntropyLoss for multi-class problems, and the Adam optimizer.
Focus on: the transform pipeline order (ToTensor before Normalize) and
understanding output shapes at each layer.

### transform_dataset.py
Shows how to compute mean and std for a dataset instead of using hardcoded values.
This is a practical skill you need when working with your own datasets.
Focus on: the two-pass algorithm (mean first, then std using the mean).

### letter_detective.py
A harder version of digit_detective: 26 letter classes instead of 10 digits.
Introduces: deeper networks for harder problems, label adjustment (1-indexed to
0-indexed), and using the model as OCR to decode a hidden message.
Focus on: the complete training pipeline (create model, train epoch, evaluate,
full loop) as reusable functions.

## Common Questions

**Q: Why do we need transforms?**
A: Images come in different sizes and pixel ranges (0-255). Transforms make
them uniform: same size, same value range, same format (tensor). Without this,
the model cannot learn effectively.

**Q: Why ToTensor before Normalize?**
A: ToTensor converts PIL image to tensor and scales pixels from [0,255] to [0,1].
Normalize then standardizes to mean~0, std~1. The order matters because
Normalize expects tensor input with values in a reasonable range.

**Q: What is the difference between SGD and Adam?**
A: SGD uses a fixed learning rate for all parameters. Adam adapts the learning
rate per parameter based on gradient history. Adam generally converges faster
with less tuning, making it the default choice for most projects.

**Q: Why do EMNIST labels need -1 adjustment?**
A: EMNIST Letters labels are 1-indexed (A=1, B=2, ..., Z=26), but PyTorch's
CrossEntropyLoss expects 0-indexed labels (0, 1, ..., 25). Subtracting 1 fixes
this mismatch. This kind of index mismatch is a common real-world gotcha.

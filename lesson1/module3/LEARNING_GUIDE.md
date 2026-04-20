# Learning Guide: Lesson 1, Module 3 -- Custom Datasets and Data Pipelines

## Module Overview

Real-world data does not come neatly packaged like MNIST. This module teaches
you how to build custom Dataset classes that load images from folders, CSV files,
and .mat files. You also learn data augmentation, train/val/test splitting, and
robust error handling for corrupted data.

## Recommended Reading Order

1. **data_management/flower_dataset.py** -- Custom Dataset class anatomy
2. **data_management/main.py** -- Transforms, splitting, augmentation, robust loading
3. **data_pileline/plants_dataset.py** -- Building a Dataset from CSV + image folder
4. **data_pileline/main.py** -- Computing stats, dynamic transforms, DataLoader creation

## Concept Map

```
Custom Dataset (your own data)
   |
   +--> __init__: setup paths, load labels
   +--> __len__: return total sample count
   +--> __getitem__: load one (image, label) pair
   |
   v
Transforms
   |
   +--> Training: augmentation (flip, rotate, jitter) + normalize
   +--> Validation/Test: normalize only (no randomness)
   |
   v
Split: train / val / test
   |
   v
DataLoader: batch + shuffle
   |
   v
Robust loading: handle corrupted files, wrong formats, missing data
```

## File Summaries

### flower_dataset.py
Defines three custom Dataset classes:
- FlowerDataset: loads images from a folder, labels from a .mat file
- RobustFlowerDataset: skips corrupted images, auto-converts grayscale to RGB
- MonitoredDataset: tracks access counts and load times for debugging
Focus on: the __len__ and __getitem__ methods -- these are ALL PyTorch requires.

### data_management/main.py
Full data pipeline: load dataset, apply transforms, split into train/val/test,
create DataLoaders, apply augmentation to training only, handle corrupted data.
Focus on: why augmentation only applies to training data, and the
Denormalize class for reversing normalization to visualize images.

### plants_dataset.py
A Dataset built from a CSV file (filenames + labels) and an image folder.
More realistic than FlowerDataset -- this is how most Kaggle datasets work.
Focus on: how read_df, load_labels, and retrieve_image work together.

### data_pileline/main.py
Shows how to compute dataset mean/std from scratch, create separate transform
pipelines for training vs validation, and use SubsetWithTransform to apply
different transforms to different data splits.
Focus on: the two-pass mean/std algorithm and the SubsetWithTransform pattern.

## Common Questions

**Q: Why do I need a custom Dataset class?**
A: Built-in datasets (MNIST, CIFAR) handle everything for you. But your own
data will be in custom formats: folders of images, CSV files, databases.
A custom Dataset class makes YOUR data work with PyTorch's DataLoader.

**Q: What is augmentation and why only for training?**
A: Augmentation creates random variations of training images (flip, rotate, crop)
so the model sees more diversity and generalizes better. During validation/testing,
you need consistent, deterministic data to measure true performance.

**Q: Why split into three sets (train/val/test)?**
A: Training set: model learns from this. Validation set: you tune hyperparameters
using this. Test set: used ONCE at the end to measure final performance on
completely unseen data. If you only use train/test, you risk overfitting your
hyperparameters to the test set.

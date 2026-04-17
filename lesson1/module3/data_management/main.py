"""
Lesson 1 - Module 3: Data Management — Custom Datasets, Transforms & Robustness
================================================================================
 WHAT YOU'LL LEARN:
  • Building a custom Dataset class from scratch (FlowerDataset)
  • Transform pipelines: Resize → CenterCrop → ToTensor → Normalize
  • Denormalization: reversing normalization to visualize processed images
  • Train/val/test splitting with torch.utils.data.random_split
  • Data augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter
  • Robust data loading: handling corrupted images gracefully
  • Monitoring the data pipeline: tracking access counts and load times

 KEY CONCEPT:
  In the real world, data is messy. Images come in different sizes, some are
  corrupted, some are grayscale instead of RGB. This module teaches you how to
  build a data pipeline that handles all of these issues.

 DATASET:
  Oxford 102 Flower Dataset — 102 categories of flowers, varying image sizes.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from pathlib import Path
import data_subset

import helper_utils
from flower_dataset import FlowerDataset, Denormalize, RobustFlowerDataset, MonitoredDataset


# ==================== STEP 1: LOAD AND EXPLORE THE DATASET ====================
# Download the flower dataset if not already present
path_dataset = Path.cwd() / 'data/flower_data'
helper_utils.download_dataset(path_dataset)

# Show the folder structure so we know what we're working with
helper_utils.print_data_folder_structure(path_dataset, max_depth=1)

# Initialize the custom dataset
#  CONCEPT: FlowerDataset is a custom class that inherits from torch.utils.data.Dataset.
#   It must implement __len__() and __getitem__() — the two methods PyTorch requires.
dataset = FlowerDataset(path_dataset)

print(f'Number of samples in the dataset: {len(dataset)}\n')

# Retrieve one sample to see what we get
sel_idx = 10
img, label = dataset[sel_idx]  # Returns (PIL_Image, int_label)

img_size_info = f"Image size: {img.size}"  # PIL uses .size (width, height)
print(f'{img_size_info}, Label: {label}\n')

# Visualize the raw image
helper_utils.plot_img(img, label=label, info=img_size_info)

# Show all unique labels and their descriptions
dataset_labels = dataset.labels
unique_labels = set(dataset_labels)
for label in unique_labels:
    print(f'Label: {label}, Description: {dataset.get_label_description(label)}')

# Display a grid of random samples for visual inspection
helper_utils.visual_exploration(dataset, num_rows=2, num_cols=4)


# ==================== STEP 2: DEFINE TRANSFORM PIPELINE ====================
#  CONCEPT: Images in the wild have DIFFERENT sizes. Neural networks expect
#   FIXED-size inputs. The transform pipeline solves this:
#   1. Resize(256, 256) — scale all images to the same size
#   2. CenterCrop(224) — crop the center 224×224 region (standard for many models)
#   3. ToTensor() — PIL → Tensor, [0,255] → [0,1]
#   4. Normalize(mean, std) — standardize using ImageNet statistics
#
#  NOTE: (0.485, 0.456, 0.406) and (0.229, 0.224, 0.225) are the ImageNet mean/std.
#   Using these is common practice when working with pre-trained models (Lesson 2).

mean = [0.485, 0.456, 0.406]  # ImageNet mean for R, G, B channels
std = [0.229, 0.224, 0.225]   # ImageNet std for R, G, B channels

transform = transforms.Compose([
    # --- Image transforms (operate on PIL Images) ---
    transforms.Resize((256, 256)),     # Resize to 256×256
    transforms.CenterCrop(224),        # Crop center 224×224 (standard input size)
    # --- Bridge: PIL → Tensor ---
    transforms.ToTensor(),             # Convert to tensor, scale to [0, 1]
    # --- Tensor transforms (operate on Tensors) ---
    transforms.Normalize(mean=mean, std=std),  # Standardize channels
])

# Create a new dataset with the transform applied
dataset_transformed = FlowerDataset(path_dataset, transform=transform)

# Retrieve the same sample — now it's a tensor with shape [3, 224, 224]
img_transformed, label = dataset_transformed[sel_idx]
helper_utils.quick_debug(img_transformed)  # Check shape, range, etc.
helper_utils.plot_img(img_transformed, label=label)  # Will look weird (normalized colors)


# ==================== STEP 3: DENORMALIZE FOR VISUALIZATION ====================
#  CONCEPT: After normalization, pixel values are no longer in [0, 1].
#   To display the image correctly, we need to REVERSE the normalization:
#     original_pixel = normalized_pixel * std + mean
#   The Denormalize class does this math for us.

denormalize = Denormalize(mean=mean, std=std, transforms=transforms)
img_tensor = denormalize(img_transformed)

img_shape_info = f"Image Shape: {img_tensor.size()}"
helper_utils.plot_img(img_tensor, label=label, info=img_shape_info)


# ==================== STEP 4: SPLIT INTO TRAIN / VAL / TEST ====================
#  CONCEPT: We need THREE splits:
#   - Training set: The model LEARNS from this data (adjusts weights)
#   - Validation set: Used during training to tune hyperparameters & detect overfitting
#   - Test set: Used ONCE at the end to measure final performance (unseen data)
#
#   Typical split: 70% train, 15% val, 15% test

def split_dataset(dataset, val_fraction=0.15, test_fraction=0.15):
    """Split dataset into train, validation, and test sets."""
    total_size = len(dataset)
    val_size = int(total_size * val_fraction)
    test_size = int(total_size * test_fraction)
    train_size = total_size - val_size - test_size

    # random_split randomly partitions the dataset into non-overlapping subsets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    return train_dataset, val_dataset, test_dataset


train_dataset, val_dataset, test_dataset = split_dataset(dataset_transformed)

print(f"Length of training dataset:   {len(train_dataset)}")
print(f"Length of validation dataset: {len(val_dataset)}")
print(f"Length of test dataset:       {len(test_dataset)}")


# ==================== STEP 5: CREATE DATA LOADERS ====================
batch_size = 32

#  TRAINING: shuffle=True (randomize order each epoch — prevents memorization)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#  VALIDATION & TEST: shuffle=False (order doesn't matter for evaluation)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# ==================== STEP 6: SIMULATE TRAINING EPOCHS ====================
# This shows the STRUCTURE of training (without actual model training yet).
# Each epoch has a training pass + validation pass.

n_epochs = 2

for epoch in range(n_epochs):
    print(f"=== Processing epoch {epoch} ===")

    # --- Training pass ---
    print(f"Pass number {epoch} through the training set")
    print('Training...')
    train_samples = len(train_dataset)
    train_bar = helper_utils.get_dataloader_bar(train_dataloader, color='blue')

    for batch, (images, labels) in enumerate(train_dataloader):
        helper_utils.update_dataloader_bar(train_bar, batch, batch_size, train_samples)
        # In a real training loop, you'd do:
        # optimizer.zero_grad()
        # outputs = model(images)
        # loss = loss_function(outputs, labels)
        # loss.backward()
        # optimizer.step()

    # --- Validation pass ---
    print(f"\nPass number {epoch} through the validation set")
    print('Validation...')
    val_bar = helper_utils.get_dataloader_bar(val_dataloader, color='orange')
    val_samples = len(val_dataset)

    for batch, (images, labels) in enumerate(val_dataloader):
        helper_utils.update_dataloader_bar(val_bar, batch, batch_size, val_samples)
        # In real training: evaluate model accuracy here


# --- Final test pass ---
print("\nFinal pass through the test set for evaluation")
test_bar = helper_utils.get_dataloader_bar(test_dataloader, color='green')
test_samples = len(test_dataset)

for batch, (images, labels) in enumerate(test_dataloader):
    helper_utils.update_dataloader_bar(test_bar, batch, batch_size, test_samples)


# ==================== STEP 7: DATA AUGMENTATION ====================
#  CONCEPT: Augmentation artificially increases dataset diversity by applying
#   RANDOM transformations to training images. This helps the model generalize
#   better (reduce overfitting) by seeing varied versions of each image.
#
#  IMPORTANT: Only apply augmentation to TRAINING data!
#   Validation and test data should only get standard preprocessing.

def get_augmentation_transform(mean, std):
    """
    Creates a transform pipeline with data augmentation for training.

    The pipeline has two parts:
    1. Random augmentations (different each time)
    2. Deterministic preprocessing (same each time)
    """
    # --- Random augmentations ---
    augmentations_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to mirror flip
        transforms.RandomRotation(degrees=10),    # Rotate ±10 degrees
        transforms.ColorJitter(brightness=0.2),   # Randomly change brightness ±20%
    ]

    # --- Deterministic preprocessing (same every time) ---
    main_transforms = [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    # Combine: augmentations first, then standard preprocessing
    transform = transforms.Compose(augmentations_transforms + main_transforms)
    return transform


augmentation_transform = get_augmentation_transform(mean=mean, std=std)
dataset_augmented = FlowerDataset(path_dataset, transform=augmentation_transform)

# Show 8 different augmented versions of the same image
#  OBSERVE: Each version is slightly different due to random augmentation
helper_utils.visualize_augmentations(dataset_augmented, denormalize, idx=sel_idx, num_versions=8)

# Apply augmentation to training only; basic preprocessing to val/test
train_dataset = data_subset.SubsetWithTransform(train_dataset, transform=augmentation_transform)
val_dataset = data_subset.SubsetWithTransform(val_dataset, transform=transform)
test_dataset = data_subset.SubsetWithTransform(test_dataset, transform=transform)


# ==================== STEP 8: ROBUST DATA LOADING ====================
#  CONCEPT: Real-world data is MESSY. Images can be:
#   - Corrupted (can't open the file)
#   - Too small (smaller than 32×32 pixels)
#   - Wrong color mode (grayscale instead of RGB)
#
#   RobustFlowerDataset handles all these cases gracefully:
#   - Skips corrupted files and returns the next valid image
#   - Auto-converts grayscale → RGB
#   - Logs all errors for debugging

corrupted_dataset_path = Path.cwd() / 'module3/corrupted_flower_data'
robust_dataset = RobustFlowerDataset(corrupted_dataset_path)

# Test with a known tiny image (idx=2) — should skip it and return the next valid one
idx = 2
img, label = robust_dataset[idx]
helper_utils.plot_img(img)

# Verify the next image is the same (confirming the skip logic works)
next_img, next_label = robust_dataset[idx + 1]
helper_utils.plot_img(next_img)

# Test with a known grayscale image (idx=4) — should auto-convert to RGB
idx = 4
original_img_path = os.path.join(robust_dataset.img_dir, f"image_{idx + 1:05d}.jpg")
original_img = Image.open(original_img_path)
print(f"Mode of the original image file: {original_img.mode}")  # 'L' = grayscale

img, label = robust_dataset[idx]
helper_utils.plot_img(img)
print(f"Mode of the corrected image: {img.mode}")  # 'RGB' after correction

# Test with a known corrupted/unreadable image (idx=6) — should skip
idx = 6
robust_img = robust_dataset[idx][0]
helper_utils.plot_img(robust_img)

next_img, next_label = robust_dataset[idx + 1]
helper_utils.plot_img(next_img)

# Show a summary of all errors encountered
robust_dataset.get_error_summary()


# ==================== STEP 9: PIPELINE MONITORING ====================
#  CONCEPT: MonitoredDataset wraps the robust dataset and tracks:
#   - How many times each image was accessed
#   - How long each load takes (detect slow I/O)
#   - Which images were never loaded (potential issues)

monitored_corrupt_dataset = MonitoredDataset(corrupted_dataset_path)

# Iterate through the entire dataset to trigger monitoring
for idx in range(len(monitored_corrupt_dataset)):
    img, label = monitored_corrupt_dataset[idx]

# Print performance statistics
monitored_corrupt_dataset.print_stats()

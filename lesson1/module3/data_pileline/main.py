"""
Lesson 1 - Module 3: Data Pipeline -- Custom Dataset from CSV (data_pileline)
==============================================================================
WHAT YOU'LL LEARN:
  * Building a custom Dataset from CSV + image folder (PlantsDataset)
  * Computing dataset mean/std dynamically with a two-pass algorithm
  * Creating separate transform pipelines: basic vs. augmented
  * Splitting data into train/val/test and wrapping each with its own transform
  * Using SubsetWithTransform to apply different transforms per split

KEY CONCEPT:
  Real-world datasets often come as a CSV file (with filenames + labels) +
  a folder of images. This module shows you how to turn that into a PyTorch
  Dataset that works with DataLoader, just like built-in datasets.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from tqdm.auto import tqdm
import helper_utils
import unittests
from pathlib import Path

from plants_dataset import PlantsDataset


# ==================== STEP 1: EXPLORE THE RAW DATA ====================
# Understand the data structure before writing any code.

path_dataset = Path.cwd() / 'module3/data_pileline/plants_dataset'
helper_utils.print_data_folder_structure(path_dataset, max_depth=1)

# The CSV file maps image filenames to category labels
df_labels = pd.read_csv(f'{path_dataset}/df_labels.csv')
print(df_labels.head())

# The classname file maps integer labels to human-readable names
with open(f'{path_dataset}/classname.txt', 'r') as f:
    class_names = f.read().splitlines()
print(class_names)


# ==================== STEP 2: INITIALIZE THE CUSTOM DATASET ====================
# PlantsDataset reads the CSV, loads images from disk, and returns (image, label) pairs.
# NOTE: transform=None means we get raw PIL images for now.

plants_dataset = PlantsDataset(root_dir=path_dataset, transform=None)
print(f'Length of the dataset: {len(plants_dataset)}')

# Retrieve one sample to verify
sel_idx = 10
img, label = plants_dataset[sel_idx]
print(f'Description: {plants_dataset.get_label_description(label)}')
print(f'Image shape: {img.size}\n')  # PIL Image uses .size = (width, height)

unittests.exercise_1(PlantsDataset, root_dir=path_dataset)


# ==================== STEP 3: COMPUTE DATASET MEAN AND STD ====================
# KEY CONCEPT: We need per-channel mean and std for normalization, but they
# are not provided -- we must compute them ourselves.
#
# ALGORITHM (two-pass):
#   Pass 1: Compute mean = sum_of_all_pixels / total_pixel_count
#   Pass 2: Compute std  = sqrt( sum_of(pixel - mean)^2 / total_pixel_count )
#
# WHY TWO PASSES? The standard deviation formula requires the mean, which
# we only know after Pass 1.

def get_mean_std(dataset: Dataset):
    """Computes per-channel mean and standard deviation over the entire dataset."""
    # First, resize all images to a fixed size so pixel counts are consistent
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()  # PIL -> Tensor, pixels in [0, 1]
    ])

    # --- Pass 1: Compute Mean ---
    total_pixels = 0
    sum_pixels = torch.zeros(3)  # One sum per RGB channel

    mean_loader = tqdm(dataset, desc="Pass 1/2: Computing Mean")

    for img, _ in mean_loader:
        img_tensor = preprocess(img)
        # Reshape to [3, num_pixels] so we can sum per channel
        pixels = img_tensor.view(3, -1)
        sum_pixels += pixels.sum(dim=1)       # Sum of all pixel values per channel
        total_pixels += pixels.size(1)         # Count total pixels

    mean = sum_pixels / total_pixels

    # --- Pass 2: Compute Standard Deviation ---
    sum_squared_diff = torch.zeros(3)

    std_loader = tqdm(dataset, desc="Pass 2/2: Computing Std")

    for img, _ in std_loader:
        img_tensor = preprocess(img)
        pixels = img_tensor.view(3, -1)
        diff = pixels - mean.unsqueeze(1)       # Subtract mean from each pixel
        sum_squared_diff += (diff ** 2).sum(dim=1)  # Sum of squared differences

    std = torch.sqrt(sum_squared_diff / total_pixels)

    return mean, std


mean, std = get_mean_std(plants_dataset)
print(f"\nMean: {mean}")
print(f" Std: {std}")


# ==================== STEP 4: CREATE TRANSFORM PIPELINES ====================
# KEY CONCEPT: Training data gets augmentation (random variations), while
# validation/test data gets only basic preprocessing.

def get_transformations(mean, std):
    """
    Returns two transform pipelines:
      1. main_transform -- deterministic preprocessing (for val/test)
      2. transform_with_augmentation -- adds random flips/rotations (for train)
    """
    # Basic preprocessing: resize, convert to tensor, normalize
    main_tfs = [
        transforms.Resize((128, 128)),       # Resize to fixed size
        transforms.ToTensor(),                # PIL -> Tensor, [0,1]
        transforms.Normalize(mean, std),      # Standardize channels
    ]

    # Random augmentations to increase training diversity
    augmentation_tfs = [
        transforms.RandomVerticalFlip(),      # 50% chance to flip vertically
        transforms.RandomRotation(degrees=15) # Rotate +/- 15 degrees
    ]

    main_transform = transforms.Compose(main_tfs)
    transform_with_augmentation = transforms.Compose(augmentation_tfs + main_tfs)

    return main_transform, transform_with_augmentation


main_transform, transform_with_augmentation = get_transformations(mean, std)

print(main_transform)
print(transform_with_augmentation)

unittests.exercise_2(get_transformations)

# Verify the transform on a sample image
img_transformed = main_transform(img)
print(f"Transformed Image shape: {img_transformed.shape}\n")

# Get a denormalization helper (reverses Normalize for visualization)
denormalize = helper_utils.Denormalize(mean, std)

# Test the augmented transform on a sample image
img_augmented = transform_with_augmentation(img)


# ==================== STEP 5: HELPER CLASS -- SubsetWithTransform ====================
# KEY CONCEPT: After splitting a dataset with random_split, we get Subset objects.
# Subsets inherit the parent dataset's transform. But we need DIFFERENT transforms
# for train vs. val/test. This wrapper solves that.

class SubsetWithTransform(Dataset):
    """Wraps a Subset and applies a specific transform to it."""

    def __init__(self, subset: Subset, transform=None):
        self.subset = subset       # The data subset
        self.transform = transform # The transform to apply to this subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]  # Gets raw image from the parent dataset
        if self.transform:
            image = self.transform(image) # Apply THIS subset's transform
        return image, label


# ==================== STEP 6: SPLIT AND CREATE DATALOADERS ====================

def get_dataloaders(dataset, batch_size, val_fraction, test_fraction,
                    main_transform, augmentation_transform):
    """
    Splits dataset into train/val/test, applies appropriate transforms,
    and returns DataLoaders for each.

    IMPORTANT: Augmentation goes ONLY on training data!
    """
    total_size = len(dataset)
    val_size = int(total_size * val_fraction)
    test_size = int(total_size * test_fraction)
    train_size = total_size - val_size - test_size

    # Random split into three non-overlapping subsets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Wrap each subset with its own transform
    train_dataset = SubsetWithTransform(subset=train_dataset, transform=augmentation_transform)
    val_dataset = SubsetWithTransform(subset=val_dataset, transform=main_transform)
    test_dataset = SubsetWithTransform(subset=test_dataset, transform=main_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)   # Shuffle for training
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)      # No shuffle for eval
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = get_dataloaders(
    dataset=plants_dataset,
    batch_size=32,
    val_fraction=0.15,
    test_fraction=0.2,
    main_transform=main_transform,
    augmentation_transform=transform_with_augmentation,
)

print('=== Train Loader ===')
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of samples in train_dataset: {len(train_loader.dataset)}")
print(f"Transforms applied to train_dataset: {train_loader.dataset.transform}")

print('\n=== Test Loader ===')
print(f"Number of batches in test_loader: {len(test_loader)}")
print(f"Number of samples in test_dataset: {len(test_loader.dataset)}")
print(f"Transforms applied to test_dataset: {test_loader.dataset.transform}")

unittests.exercise_3(get_dataloaders, plants_dataset)

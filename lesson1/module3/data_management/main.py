## data_management.py

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

# Define the path to the root directory of the dataset.
path_dataset = Path.cwd() / 'data/flower_data'

# Call the function to download and prepare the dataset.
helper_utils.download_dataset(path_dataset)

# Display the folder structure of the dataset directory up to a depth of one.
helper_utils.print_data_folder_structure(path_dataset, max_depth=1)

# Initialize the dataset object, providing the path to the data.
dataset = FlowerDataset(path_dataset)

# Print the total number of samples in the dataset.
print(f'Number of samples in the dataset: {len(dataset)}\n')

# Define an index for a sample to retrieve.
sel_idx = 10

# Retrieve the image and label for the selected index.
img, label = dataset[sel_idx]

# Create a string detailing the image's dimensions.
img_size_info = f"Image size: {img.size}"

# Print the image size information along with its corresponding label.
print(f'{img_size_info}, Label: {label}\n')

helper_utils.plot_img(img, label=label, info=img_size_info)

# Get all labels from the dataset object.
dataset_labels = dataset.labels

# Create a set of unique labels to remove duplicates.
unique_labels = set(dataset_labels)

# Iterate through each unique label.
for label in unique_labels:
    # Print the numerical label and its corresponding text description.
    print(f'Label: {label}, Description: {dataset.get_label_description(label)}')

print("-"*45)

# Display a 2x4 grid of random samples from the dataset for visual inspection.
helper_utils.visual_exploration(dataset, num_rows=2, num_cols=4)

# Define the mean values for normalization.
mean = [0.485, 0.456, 0.406]
# Define the standard deviation values for normalization.
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    # images transforms
    transforms.Resize((256, 256)),  # Resize images to 256x256 pixels
    transforms.CenterCrop(224),  # Center crop to 224x224 pixels
    # bridge to tensor
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # tensor transforms
    transforms.Normalize(mean=mean, std=std),
])

# Create a new dataset instance with the specified image transformations.
dataset_transformed = FlowerDataset(path_dataset, transform=transform)

# Retrieve the transformed image and its label using the same index.
img_transformed, label = dataset_transformed[sel_idx]

# quick check
helper_utils.quick_debug(img_transformed)

# Plot the transformed image
helper_utils.plot_img(img_transformed, label=label)

# Create an instance of the Denormalize class with the original mean and std.
denormalize = Denormalize(mean=mean, std=std, transforms=transforms)
# Apply the denormalization transform to the image tensor.
img_tensor = denormalize(img_transformed)

# Create an information string with the tensor's shape.
img_shape_info = f"Image Shape: {img_tensor.size()}"
# # Plot the denormalized image to visualize the result.
helper_utils.plot_img(img_tensor, label=label, info=img_shape_info)

def split_dataset(dataset, val_fraction=0.15, test_fraction=0.15):
    """
    Split the dataset into training, validation, and test sets.
    
    By default, this function splits the data into 70% for training,
    15% for validation, and 15% for testing.
    """

    # Calculate the sizes of each split.
    total_size = len(dataset)
    val_size = int(total_size * val_fraction)
    test_size = int(total_size * test_fraction)
    train_size = total_size - val_size - test_size

    # Use random_split to create the datasets.
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    return train_dataset, val_dataset, test_dataset


train_dataset, val_dataset, test_dataset = split_dataset(dataset_transformed)

print(f"Length of training dataset:   {len(train_dataset)}")
print(f"Length of validation dataset: {len(val_dataset)}")
print(f"Length of test dataset:       {len(test_dataset)}")

print("-"*45)

# Set the batch size for the data loaders.
batch_size = 32

# Create the DataLoader for the training set, with shuffling enabled.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Create the DataLoader for the validation set, with shuffling disabled.
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Create the DataLoader for the test set, with shuffling disabled.
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the total number of training epochs.
n_epochs = 2

# Start the main training loop for each epoch.
for epoch in range(n_epochs):
    # Print a header to indicate the start of a new epoch.
    print(f"=== Processing epoch {epoch} ===")

    # Announce the start of the training phase.
    print(f"Pass number {epoch} through the training set")
    print('Training...')
    # Get the total number of samples in the training set.
    train_samples = len(train_dataset)
    # Create a progress bar for the training data loader.
    train_bar = helper_utils.get_dataloader_bar(train_dataloader, color='blue')
    
    # Iterate over the training data loader to get batches of images and labels.
    for batch, (images, labels) in enumerate(train_dataloader):
        # Update the training progress bar for the current batch.
        helper_utils.update_dataloader_bar(train_bar, batch, batch_size, train_samples)

    # Announce the start of the validation phase.
    print(f"\nPass number {epoch} through the validation set")
    print('Validation...')
    # Create a progress bar for the validation data loader.
    val_bar = helper_utils.get_dataloader_bar(val_dataloader, color='orange')
    # Get the total number of samples in the validation set.
    val_samples = len(val_dataset)
    
    # Iterate over the validation data loader to get batches of images and labels.
    for batch, (images, labels) in enumerate(val_dataloader):
        # Update the validation progress bar for the current batch.
        helper_utils.update_dataloader_bar(val_bar, batch, batch_size, val_samples)


# Announce the final evaluation on the test set.
print("\nFinal pass through the test set for evaluation")
# Create a progress bar for the test data loader.
test_bar = helper_utils.get_dataloader_bar(test_dataloader, color='green')
# Get the total number of samples in the test set.
test_samples = len(test_dataset)

# Iterate over the test data loader to get batches of images and labels.
for batch, (images, labels) in enumerate(test_dataloader):
    # Update the test progress bar for the current batch.
    helper_utils.update_dataloader_bar(test_bar, batch, batch_size, test_samples)

print("-"*45)

def get_augmentation_transform(mean, std):
    """
    Creates and returns a composition of image transformations for data augmentation
    and preprocessing.

    Args:
        mean (list or tuple): A sequence of mean values for each channel.
        std (list or tuple): A sequence of standard deviation values for each channel.

    Returns:
        torchvision.transforms.Compose: A composed pipeline of transformations.
    """
    # Define a list of data augmentation transformations to be applied randomly.
    augmentations_transforms = [
        # Randomly flip the image horizontally with a 50% probability.
        transforms.RandomHorizontalFlip(p=0.5),
        # Randomly rotate the image within a range of +/- 10 degrees.
        transforms.RandomRotation(degrees=10),
        # Randomly adjust the brightness of the image.
        transforms.ColorJitter(brightness=0.2),
    ]
    
    # Define the main list of standard, non-random transformations.
    main_transforms = [
        # Resize the input image to 256x256 pixels.
        transforms.Resize((256, 256)),
        # Crop the center 224x224 pixels of the image.
        transforms.CenterCrop(224),
        # Convert the PIL Image to a PyTorch tensor.
        transforms.ToTensor(),
        # Normalize the tensor with the provided mean and standard deviation.
        transforms.Normalize(mean=mean, std=std),
    ]

    # Combine the augmentation and main transformations into a single pipeline.
    transform = transforms.Compose(augmentations_transforms + main_transforms)
    # Return the final composed transform object.
    return transform


# Create the augmentation and preprocessing pipeline, providing the normalization stats.
augmentation_transform = get_augmentation_transform(mean=mean, std=std)

# Initialize a new dataset instance that will use the augmentation pipeline.
dataset_augmented = FlowerDataset(path_dataset, transform=augmentation_transform)

# Display 8 augmented versions of the selected image to see the transformations.
helper_utils.visualize_augmentations(dataset_augmented, denormalize, idx=sel_idx, num_versions=8)

# Apply the augmentation pipeline to the training subset.
train_dataset = data_subset.SubsetWithTransform(train_dataset, transform=augmentation_transform)
# Apply the basic preprocessing transform to the validation subset.
val_dataset = data_subset.SubsetWithTransform(val_dataset, transform=transform)
# Apply the basic preprocessing transform to the test subset.
test_dataset = data_subset.SubsetWithTransform(test_dataset, transform=transform)

print(train_dataset.transform)
print(val_dataset.transform)
print(test_dataset.transform)

print("-"*45)

# Define the path to the directory containing the corrupted dataset.
corrupted_dataset_path =  Path.cwd() / 'module3/corrupted_flower_data'

# Initialize the robust dataset handler with the path to the corrupted data.
robust_dataset = RobustFlowerDataset(corrupted_dataset_path)

# Set the index to a known corrupted image. Image 2 is tiny
idx = 2

# Attempt to retrieve the image; the robust dataset will skip the bad one and return the next.
img, label = robust_dataset[idx]

# Plot the retrieved image, which should be the one following the corrupted one.
helper_utils.plot_img(img)

# Explicitly retrieve the next image in the sequence to verify.
next_img, next_label = robust_dataset[idx + 1]

# Plot the next image; it should be identical to the one above.
helper_utils.plot_img(next_img)

# Set the index to a known grayscale image.
# Image 4 is corrupted (grayscale)
idx = 4

# Reconstruct the path to the original image file.
original_img_path = os.path.join(robust_dataset.img_dir, f"image_{idx + 1:05d}.jpg")
# Open the original image directly to check its mode before correction.
original_img = Image.open(original_img_path)
# Print the mode of the original, uncorrected image.
print(f"Mode of the original image file: {original_img.mode}")  # Prints 'L' for 8-bit grayscale. A standard color image would be 'RGB'.

# Retrieve the image; the robust loader should automatically convert it to RGB.
img, label = robust_dataset[idx]

# Plot the image to visually confirm it's now in color.
helper_utils.plot_img(img)

# Print the image's mode to confirm it has been corrected to 'RGB'.
print(f"Mode of the corrected image: {img.mode}")

# Set the index to a known corrupted or unreadable image.
idx = 6

# Attempt to retrieve the image; the robust loader should skip the corrupted file and return the next one.
robust_img = robust_dataset[idx][0]

# Plot the retrieved image, which should be the sample from the next index (7).
helper_utils.plot_img(robust_img)

# Explicitly retrieve the next image in the sequence to verify the fallback logic.
# Check next image to ensure it's correct
next_img, next_label = robust_dataset[idx + 1]

# Plot the next image; it should be identical to the one above, confirming the skip.
helper_utils.plot_img(next_img)

# Display the summary of any corrupted or problematic images found during loading.
robust_dataset.get_error_summary()

print("-"*45)

# Initialize the monitored dataset with the path to the potentially corrupted data.
monitored_corrupt_dataset = MonitoredDataset(corrupted_dataset_path)

# Loop through every index in the dataset to simulate a full pass.
# Iterate through the dataset to trigger monitoring
for idx in range(len(monitored_corrupt_dataset)):
    # Access the sample at the current index to trigger the monitoring and error-handling logic.
    img, label = monitored_corrupt_dataset[idx]

# Print the statistics
monitored_corrupt_dataset.print_stats()
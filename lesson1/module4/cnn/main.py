"""
Lesson 1 - Module 4: CNN Training on CIFAR-100 (cnn/main.py)
==============================================================
WHAT YOU'LL LEARN:
  * Defining separate transform pipelines for training (with augmentation) vs. validation
  * Building and training a CNN on CIFAR-100 (15-class subset)
  * The train_epoch / validate_epoch / training_loop pattern
  * Best model checkpointing: saving the model with highest validation accuracy
  * Using weight_decay (L2 regularization) to prevent overfitting

KEY CONCEPT:
  This is the first time we train a CNN (Convolutional Neural Network).
  Unlike the Dense networks in Module 2 that flattened images to 1D vectors,
  CNNs process 2D images directly, preserving spatial structure.

DATASET: CIFAR-100 (15-class subset: flowers, mammals, insects)
  - 32x32 color images, 3 channels (RGB)
  - A harder dataset than MNIST: natural images, more classes, lower resolution
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import helper_utils
import unittests
from pathlib import Path

from cnn_block import CNNBlock, SimpleCNN


# ==================== STEP 1: DEVICE SELECTION ====================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

data_path = Path.cwd() / 'data/cifar_100'

# Pre-calculated statistics for CIFAR-100 (3 channels: R, G, B)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


# ==================== STEP 2: DEFINE TRANSFORM PIPELINES ====================
# KEY CONCEPT: Training gets augmentation (random flips/rotations for variety),
# validation gets only basic preprocessing (ToTensor + Normalize).
# This is a standard best practice.

def define_transformations(mean, std):
    """
    Creates separate transform pipelines for training and validation.

    Training:   Augmentation + ToTensor + Normalize
    Validation: ToTensor + Normalize (NO augmentation -- data must be consistent)
    """
    # Training transforms -- includes random augmentation for variety
    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to mirror horizontally
        transforms.RandomVerticalFlip(p=0.5),    # 50% chance to flip vertically
        transforms.RandomRotation(degrees=15),   # Rotate +/- 15 degrees
        transforms.ToTensor(),                    # PIL -> Tensor, [0,255] -> [0,1]
        transforms.Normalize(mean=mean, std=std), # Standardize channels
    ])

    # Validation transforms -- deterministic, no augmentation
    val_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transformations, val_transformations


train_transform, val_transform = define_transformations(cifar100_mean, cifar100_std)
print("Training Transformations:")
print(train_transform)
print("\nValidation Transformations:")
print(val_transform)

unittests.exercise_1(define_transformations)


# ==================== STEP 3: LOAD DATASET AND CREATE DATALOADERS ====================
# We use a 15-class subset of CIFAR-100: 5 flowers + 5 mammals + 5 insects

all_target_classes = [
    # Flowers
    'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
    # Mammals
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    # Insects
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'
]

train_dataset, val_dataset = helper_utils.load_cifar100_subset(
    all_target_classes, train_transform, val_transform, root=data_path
)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# ==================== STEP 4: VERIFY MODEL ARCHITECTURE ====================
# Quick sanity check: make sure the model produces the right output shape.

# Test CNNBlock
verify_cnn_block = CNNBlock(in_channels=3, out_channels=16)
dummy_input = torch.randn(1, 3, 32, 32)   # 1 image, 3 channels, 32x32
output = verify_cnn_block(dummy_input)
print(f"CNNBlock: Input shape {dummy_input.shape} -> Output shape {output.shape}")
# Output: [1, 16, 16, 16] -- channels doubled, spatial size halved

unittests.exercise_2(CNNBlock)

# Test SimpleCNN
verify_simple_cnn = SimpleCNN(num_classes=15)
dummy_input = torch.randn(64, 3, 32, 32)   # Batch of 64 images
output = verify_simple_cnn(dummy_input)
print(f"SimpleCNN: Input shape {dummy_input.shape} -> Output shape {output.shape}")
# Output: [64, 15] -- one score per class for each image

unittests.exercise_3(SimpleCNN, CNNBlock)

# Instantiate the actual model
num_classes = len(train_dataset.classes)
model = SimpleCNN(num_classes)


# ==================== STEP 5: DEFINE LOSS AND OPTIMIZER ====================
# CrossEntropyLoss: standard for multi-class classification
# Adam with weight_decay: weight_decay adds L2 regularization (penalizes large weights)
#   This helps prevent overfitting by encouraging smaller, more distributed weights.

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)


# ==================== STEP 6: TRAINING FUNCTION ====================
def train_epoch(model, train_loader, loss_function, optimizer, device):
    """
    Trains the model for one epoch.

    Returns: average training loss for the epoch.
    """
    model.train()  # Enable training mode (activates dropout, updates batchnorm)
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Standard 5-step training loop
        optimizer.zero_grad()          # Clear old gradients
        outputs = model(images)        # Forward pass
        loss = loss_function(outputs, labels)  # Compute loss
        loss.backward()                # Backpropagation
        optimizer.step()               # Update weights

        # Accumulate loss (multiply by batch size because loss is averaged per batch)
        running_loss += loss.item() * images.size(0)

    # Average loss over ALL samples (not just batches)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


helper_utils.verify_training_process(SimpleCNN, train_loader, loss_function, train_epoch, device)
unittests.exercise_4(train_epoch)


# ==================== STEP 7: VALIDATION FUNCTION ====================
def validate_epoch(model, val_loader, loss_function, device):
    """
    Evaluates the model on the validation set.

    Returns: (average_val_loss, accuracy_percentage)
    """
    model.eval()  # Evaluation mode (disables dropout, freezes batchnorm)
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient tracking -- saves memory and computation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)                          # Forward pass only
            val_loss = loss_function(outputs, labels)        # Compute loss
            running_val_loss += val_loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)  # Get the class with highest score

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_accuracy = 100.0 * correct / total

    return epoch_val_loss, epoch_accuracy


helper_utils.verify_validation_process(SimpleCNN, val_loader, loss_function, validate_epoch, device)
unittests.exercise_5(validate_epoch)


# ==================== STEP 8: FULL TRAINING LOOP WITH BEST MODEL SAVING ====================
# KEY CONCEPT: We track the best validation accuracy and save that model's weights.
#   The model at the LAST epoch isn't always the best -- it might have started
#   overfitting. We restore the best checkpoint at the end.

def training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    """
    Full training loop with best-model checkpointing.

    After each epoch:
      1. Train on training data
      2. Evaluate on validation data
      3. If this is the best accuracy so far, save the model weights

    At the end, restore the best model weights.
    """
    model.to(device)

    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0

    train_losses, val_losses, val_accuracies = [], [], []

    print("--- Training Started ---")

    for epoch in range(num_epochs):
        # Train
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        train_losses.append(epoch_loss)

        # Validate
        epoch_val_loss, epoch_accuracy = validate_epoch(model, val_loader, loss_function, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Accuracy: {epoch_accuracy:.2f}%")

        # Save best model checkpoint
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            # copy.deepcopy ensures we save a SNAPSHOT, not a reference
            best_model_state = copy.deepcopy(model.state_dict())

    print("--- Finished Training ---")

    # Restore the best model weights
    if best_model_state:
        print(f"\n--- Best model: {best_val_accuracy:.2f}% val accuracy at epoch {best_epoch} ---")
        model.load_state_dict(best_model_state)

    metrics = [train_losses, val_losses, val_accuracies]
    return model, metrics


# ==================== STEP 9: RUN TRAINING ====================
trained_model, training_metrics = training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    num_epochs=50,
    device=device
)

# Plot training curves (loss and accuracy over epochs)
helper_utils.plot_training_metrics(training_metrics)


# ==================== STEP 10: PREVIEW OF LESSON 2 ====================
# Run a more advanced training strategy (will be taught in the next course)
from c2_preview.c2_preview import course_2_preview

trained_model = course_2_preview(
    train_dataset,
    val_dataset,
    loss_function,
    device,
    num_epochs=5
)

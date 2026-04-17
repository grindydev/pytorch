"""
Lesson 1 - Module 4: Nature Classification with CNN (nature_classification/main.py)
===================================================================================
WHAT YOU'LL LEARN:
  * Building a CNN from scratch with explicit layer definitions (not using CNNBlock)
  * Complete end-to-end pipeline: data -> model -> training -> evaluation -> prediction
  * Prototyping: start with a small subset (9 classes), then scale to full dataset (15 classes)
  * Tracing data flow through the model to understand shape transformations
  * Visualizing model predictions on validation images

KEY CONCEPT:
  This is the capstone of Lesson 1 -- putting everything together:
  transforms, DataLoader, CNN architecture, training loop, and evaluation.

DATASET: CIFAR-100 (9-class subset -> 15-class full)
  32x32 color images of flowers, mammals, and insects
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import helper_utils
from pathlib import Path


# ==================== STEP 1: SETUP ====================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# CIFAR-100 normalization constants
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

# Training: augmentation for variety
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),     # Mirror horizontally
    transforms.RandomRotation(15),         # Rotate +/- 15 degrees
    transforms.ToTensor(),                 # PIL -> Tensor
    transforms.Normalize(cifar100_mean, cifar100_std)  # Standardize
])

# Validation: deterministic preprocessing only
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])


# ==================== STEP 2: PROTOTYPE WITH SMALL SUBSET (9 CLASSES) ====================
# KEY CONCEPT: When building a new model, START SMALL. Use a subset of classes
# to iterate faster. Once the model works, scale up to the full dataset.

subset_target_classes = [
    # Flowers
    'orchid', 'poppy', 'sunflower',
    # Mammals
    'fox', 'raccoon', 'skunk',
    # Insects
    'butterfly', 'caterpillar', 'cockroach'
]

data_path = Path.cwd() / 'data/cifar_100'

train_dataset_proto, val_dataset_proto = helper_utils.load_cifar100_subset(
    subset_target_classes, train_transform, val_transform, root=data_path
)

batch_size = 64

train_loader_proto = DataLoader(train_dataset_proto, batch_size=batch_size, shuffle=True)
val_loader_proto = DataLoader(val_dataset_proto, batch_size=batch_size, shuffle=False)


# ==================== STEP 3: DEFINE THE CNN ARCHITECTURE ====================
# KEY CONCEPT: This CNN has 3 conv blocks (each: Conv -> ReLU -> MaxPool)
# followed by fully-connected layers for classification.
#
# SHAPE TRANSFORMATIONS (input 32x32):
#   Block 1: [3, 32, 32]  -> Conv(3->32)  -> [32, 32, 32] -> Pool -> [32, 16, 16]
#   Block 2: [32, 16, 16] -> Conv(32->64) -> [64, 16, 16] -> Pool -> [64, 8, 8]
#   Block 3: [64, 8, 8]   -> Conv(64->128)-> [128, 8, 8]  -> Pool -> [128, 4, 4]
#   Flatten: [128, 4, 4]  -> [2048]
#   FC1:    [2048] -> [512] -> ReLU -> Dropout
#   FC2:    [512]  -> [num_classes]

class SimpleCNN(nn.Module):
    """A CNN with 3 convolutional blocks and 2 fully-connected layers."""

    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Classifier ---
        self.flatten = nn.Flatten()
        # After 3 pooling layers: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 2048 -> 512
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)            # 50% dropout to prevent overfitting
        self.fc2 = nn.Linear(512, num_classes)    # 512 -> num_classes

    def forward(self, x):
        # Conv Block 1
        x = self.pool1(self.relu1(self.conv1(x)))
        # Conv Block 2
        x = self.pool2(self.relu2(self.conv2(x)))
        # Conv Block 3
        x = self.pool3(self.relu3(self.conv3(x)))
        # Classifier
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x


# Create prototype model (9 classes)
num_classes = len(train_dataset_proto.classes)
prototype_model = SimpleCNN(num_classes)
print(prototype_model)

# Visualize how data flows through the model (shape at each layer)
print("\n--- Tracing Data Flow ---")
helper_utils.print_data_flow(prototype_model)


# ==================== STEP 4: TRAIN THE PROTOTYPE ====================
loss_function = nn.CrossEntropyLoss()
optimizer_prototype = optim.Adam(prototype_model.parameters(), lr=0.001)


def training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    """
    Full training loop with train + validation each epoch.

    Returns: (trained_model, [train_losses, val_losses, val_accuracies])
    """
    model.to(device)
    train_losses, val_losses, val_accuracies = [], [], []

    print("--- Training Started ---")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()              # Clear gradients
            outputs = model(images)            # Forward pass
            loss = loss_function(outputs, labels)  # Compute loss
            loss.backward()                    # Backpropagation
            optimizer.step()                   # Update weights

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                val_loss = loss_function(outputs, labels)
                running_val_loss += val_loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        epoch_accuracy = 100.0 * correct / total
        val_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Accuracy: {epoch_accuracy:.2f}%")

    print("--- Finished Training ---")
    metrics = [train_losses, val_losses, val_accuracies]
    return model, metrics


# Train prototype on 9 classes
trained_proto_model, training_metrics_proto = training_loop(
    model=prototype_model,
    train_loader=train_loader_proto,
    val_loader=val_loader_proto,
    loss_function=loss_function,
    optimizer=optimizer_prototype,
    num_epochs=15,
    device=device
)

helper_utils.plot_training_metrics(training_metrics_proto)


# ==================== STEP 5: SCALE UP TO FULL DATASET (15 CLASSES) ====================
# KEY CONCEPT: Now that the prototype works, train on the full dataset.
# Same architecture, just more classes.

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create a NEW model instance for 15 classes
num_classes = len(train_dataset.classes)
model = SimpleCNN(num_classes)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train on full 15-class dataset
trained_model, training_metrics = training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    num_epochs=25,
    device=device
)

helper_utils.plot_training_metrics(training_metrics)

# Visualize predictions on validation images
helper_utils.visualise_predictions(
    model=trained_model,
    data_loader=val_loader,
    device=device,
    grid=(5, 5)
)

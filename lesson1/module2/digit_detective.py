"""
Lesson 1 - Module 2: Digit Detective — MNIST Classification (digit_detective.py)
=================================================================================
 WHAT YOU'LL LEARN:
  • Loading built-in datasets with torchvision (MNIST handwritten digits)
  • Image transforms: ToTensor + Normalize
  • Building a simple DNN (Dense Neural Network) for image classification
  • The training loop: train_epoch() and evaluate() functions
  • Using DataLoader for batching and shuffling
  • CrossEntropyLoss for multi-class classification
  • Adam optimizer — an adaptive learning rate optimizer

 KEY CONCEPT:
  MNIST is the "Hello World" of deep learning — 28×28 grayscale images of
  handwritten digits (0–9). The goal: predict which digit an image shows.
  This is a 10-class classification problem.

 MODEL ARCHITECTURE:
  Input (784) → Dense(128) → ReLU → Dense(10)
  - 784 = 28 × 28 pixels, flattened to a 1D vector
  - 128 hidden neurons learn patterns (edges, curves, loops)
  - 10 output neurons = one per digit class (0–9)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import helper_utils
from pathlib import Path


# ==================== STEP 1: DEVICE SELECTION ====================
#  CONCEPT: Deep learning is MUCH faster on GPU. PyTorch supports:
#   - CUDA: NVIDIA GPUs (most common for training)
#   - MPS: Apple Silicon GPUs (M1/M2/M3 Macs)
#   - CPU: Fallback, works but slow
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU")


# ==================== STEP 2: EXPLORE RAW DATA (WITHOUT TRANSFORMS) ====================
# First, let's see what the data looks like BEFORE any processing.

data_path = Path.cwd() / "data/MNIST_data"

# Load MNIST training set WITHOUT any transforms
train_dataset_without_transform = torchvision.datasets.MNIST(
    root=data_path,      # Where to store/load data on disk
    train=True,          # True = training split (60k images), False = test split (10k images)
    download=True        # Download from the internet if not found locally
)

# Each sample is a (PIL_Image, label) tuple
image_pil, label = train_dataset_without_transform[0]  # Get the first sample

print(f"Image type:        {type(image_pil)}")    # <class 'PIL.Image.Image'>
print(f"Image Dimensions:  {image_pil.size}")      # (28, 28) — width × height
print(f"Label Type:        {type(label)}")          # <class 'int'>
print(f"Label value:       {label}")                 # The digit (0–9)


# ==================== STEP 3: DEFINE TRANSFORMS ====================
#  CONCEPT: Transforms are a pipeline of operations applied to each image:
#   1. ToTensor() — Converts PIL Image → PyTorch Tensor, scales pixels from [0,255] to [0.0, 1.0]
#   2. Normalize(mean, std) — Standardizes: (pixel - mean) / std
#      This helps the model converge faster.
#
#  NOTE: (0.1307,) and (0.3081,) are the pre-computed mean and std of MNIST.
#   These values were calculated over the entire training set.

transform = transforms.Compose([
    transforms.ToTensor(),                          # PIL → Tensor, [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,))      # Standardize to ~mean=0, ~std=1
])


# ==================== STEP 4: LOAD DATASETS WITH TRANSFORMS ====================
# Now load the datasets WITH the transform pipeline applied to each image.

train_dataset = torchvision.datasets.MNIST(
    root=data_path,
    train=True,
    download=True,
    transform=transform  # ← The transform is applied here
)

# Check what we get after transformation
image_tensor, label = train_dataset[0]
print(f"Image Type (after transform):  {type(image_tensor)}")   # torch.Tensor
print(f"Image Shape (after transform): {image_tensor.shape}")   # [1, 28, 28] — (channels, height, width)
print(f"Label value:                   {label}")

#  NOTE: The shape is [1, 28, 28] because:
#   - 1 = number of color channels (grayscale = 1, RGB = 3)
#   - 28 = height in pixels
#   - 28 = width in pixels

# Visualize the tensor image
helper_utils.display_image(image_tensor, label, "MNIST Digit (Tensor)", show_values=True)

# Load the TEST dataset (used to evaluate model performance on unseen data)
test_dataset = torchvision.datasets.MNIST(
    root=data_path,
    train=False,         # ← This gives us the test split (10k images)
    download=True,
    transform=transform
)


# ==================== STEP 5: CREATE DATA LOADERS ====================
#  CONCEPT: DataLoader handles batching, shuffling, and parallel loading.
#   Instead of feeding one image at a time, we feed a BATCH of images.
#   This is much more efficient for GPU computation.
#
#   - batch_size=64: Process 64 images at once
#   - shuffle=True for training: Randomize order each epoch (prevents the model
#     from memorizing the order of examples)
#   - shuffle=False for testing: Order doesn't matter for evaluation

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# ==================== STEP 6: DEFINE THE NEURAL NETWORK ====================
#  CONCEPT: A simple feed-forward (dense) neural network for classification.
#   - Flatten: Converts 2D image [1, 28, 28] → 1D vector [784]
#   - Linear(784, 128): First hidden layer — learns 128 features from 784 pixels
#   - ReLU: Non-linear activation (sets negative values to 0)
#   - Linear(128, 10): Output layer — 10 scores, one per digit class

class SimpleMNISTDNN(nn.Module):
    """A simple deep neural network for MNIST digit classification."""

    def __init__(self):
        super(SimpleMNISTDNN, self).__init__()
        self.flatten = nn.Flatten()  # Reshape [batch, 1, 28, 28] → [batch, 784]
        self.layers = nn.Sequential(
            nn.Linear(784, 128),  # Input layer → Hidden layer
            nn.ReLU(),            # Non-linear activation
            nn.Linear(128, 10)    # Hidden layer → Output layer (10 classes)
        )

    def forward(self, x):
        x = self.flatten(x)  # Flatten the 2D image to 1D
        x = self.layers(x)   # Pass through the layers
        return x


# ==================== STEP 7: DEFINE LOSS FUNCTION & OPTIMIZER ====================
#  CONCEPT: CrossEntropyLoss — the standard loss for multi-class classification.
#   It combines LogSoftmax + NLLLoss in one step.
#   The model outputs raw "logits" (scores), and this loss:
#   1. Converts scores to probabilities via softmax
#   2. Measures how wrong the predicted probabilities are vs. the true label
#
#  CONCEPT: Adam optimizer — an adaptive learning rate optimizer.
#   Unlike SGD which uses a fixed learning rate, Adam automatically adjusts
#   the learning rate for each parameter based on gradient history.
#   Generally converges faster than plain SGD.

model = SimpleMNISTDNN()
loss_function = nn.CrossEntropyLoss()              # Multi-class classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam with learning rate 0.001


# ==================== STEP 8: TRAINING FUNCTION (ONE EPOCH) ====================
#  CONCEPT: One "epoch" = one complete pass through the entire training dataset.
#   Since we use batches, one epoch = multiple batches = multiple optimizer steps.

def train_epoch(model, loss_function, optimizer, train_loader, device):
    """
    Trains the model for one epoch.
    Returns: (trained_model, average_loss_for_epoch)
    """
    model = model.to(device)   # Move model to GPU/CPU
    model.train()               #  Set to training mode (enables dropout, batchnorm updates, etc.)

    epoch_loss = 0.0
    running_loss = 0.0
    num_correct_predictions = 0
    total_predictions = 0
    total_batches = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to the same device as the model
        inputs, targets = inputs.to(device), targets.to(device)

        # --- THE CORE TRAINING LOOP (same 5 steps as leaner.py) ---
        optimizer.zero_grad()        # Step 1: Clear old gradients
        outputs = model(inputs)      # Step 2: Forward pass — get predictions
        loss = loss_function(outputs, targets)  # Step 3: Calculate loss
        loss.backward()              # Step 4: Backpropagation — compute gradients
        optimizer.step()             # Step 5: Update weights

        # --- Track metrics ---
        loss_value = loss.item()
        epoch_loss += loss_value
        running_loss += loss_value

        # Calculate accuracy for this batch
        # outputs.max(1) returns (max_values, argmax_indices) along dim 1
        # argmax gives the predicted class (the digit with highest score)
        _, predicted_indices = outputs.max(1)
        batch_size = targets.size(0)
        total_predictions += batch_size
        num_correct_predictions += predicted_indices.eq(targets).sum().item()

        # Print progress periodically
        if (batch_idx + 1) % 134 == 0 or (batch_idx + 1) == total_batches:
            avg_running_loss = running_loss / 134
            accuracy = 100. * num_correct_predictions / total_predictions
            print(f'\tStep {batch_idx + 1}/{total_batches} - Loss: {avg_running_loss:.3f} | Acc: {accuracy:.2f}%')
            running_loss = 0.0
            num_correct_predictions = 0
            total_predictions = 0

    avg_epoch_loss = epoch_loss / total_batches
    return model, avg_epoch_loss


# ==================== STEP 9: EVALUATION FUNCTION ====================
#  CONCEPT: Evaluation = measuring how well the model performs on UNSEEN test data.
#   Key differences from training:
#   - model.eval() — disables dropout, uses running stats for batchnorm
#   - torch.no_grad() — don't compute gradients (saves memory, faster)
#   - No optimizer.step() — we're NOT updating weights, just measuring accuracy

def evaluate(model, test_loader, device):
    """Evaluates model accuracy on the test dataset."""
    model.eval()  #  Set to evaluation mode
    num_correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  #  Disable gradient tracking for efficiency
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)                    # Forward pass only
            _, predicted_indices = outputs.max(1)      # Get predicted class

            batch_size = targets.size(0)
            total_predictions += batch_size
            num_correct_predictions += predicted_indices.eq(targets).sum().item()

    accuracy_percentage = (num_correct_predictions / total_predictions) * 100
    print(f'\tAccuracy - {accuracy_percentage:.2f}%')
    return accuracy_percentage


# ==================== STEP 10: RUN THE TRAINING ====================
# Train for multiple epochs, evaluating after each one.

NUM_EPOCHS = 5
train_loss = []
test_acc = []

for epoch in range(NUM_EPOCHS):
    print(f'\n[Training] Epoch {epoch+1}:')
    trained_model, loss = train_epoch(model, loss_function, optimizer, train_loader, device)
    train_loss.append(loss)

    print(f'[Testing] Epoch {epoch+1}:')
    accuracy = evaluate(trained_model, test_loader, device)
    test_acc.append(accuracy)

# Visualize some predictions
helper_utils.display_predictions(trained_model, test_loader, device)

# Plot loss and accuracy curves over all epochs
helper_utils.plot_metrics(train_loss, test_acc)

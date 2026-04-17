"""
Lesson 1 - Module 2: Letter Detective — EMNIST Classification (letter_detective.py)
====================================================================================
 WHAT YOU'LL LEARN:
  • Loading EMNIST (Extended MNIST) — 26 letter classes instead of 10 digits
  • Custom image orientation correction (rotate + flip for EMNIST)
  • Building DataLoaders with a helper function
  • Designing a deeper network (3 layers) for a harder problem
  • Full training pipeline: init → train_epoch → evaluate → train_and_evaluate
  • Label adjustment (EMNIST letters are 1-indexed, PyTorch expects 0-indexed)
  • Using the trained model to decode a hidden message from images

 KEY CONCEPT:
  EMNIST "Letters" split has 26 classes (A–Z). The labels are 1-indexed (A=1, B=2, ... Z=26),
  but PyTorch's CrossEntropyLoss expects 0-indexed labels, so we subtract 1: targets = targets - 1

 MODEL ARCHITECTURE:
  Input (784) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(26)
  - Deeper than the digit model to handle the harder 26-class problem
"""

import os
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torchvision.transforms.functional as F

import helper_utils
import unittests
from pathlib import Path


# ==================== STEP 1: DEVICE SELECTION ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

data_path = Path.cwd() / "data/EMNIST_data"

# Avoid re-downloading if data already exists locally
if os.path.exists(data_path) and os.path.isdir(data_path):
    download = False
    print("EMNIST Data folder found locally. Loading from local.\n")
else:
    download = True
    print("EMNIST Data folder not found locally. Downloading data.\n")


# ==================== STEP 2: LOAD EMNIST DATASETS ====================
#  CONCEPT: EMNIST has multiple "splits" — 'letters' gives us 26 letter classes.
#   Other splits include 'digits', 'mnist', 'balanced', 'byclass', 'bymerge'.

train_dataset = datasets.EMNIST(
    root=data_path,
    split='letters',    # 26 letter classes (A–Z)
    train=True,         # Training split
    download=download
)

test_dataset = datasets.EMNIST(
    root=data_path,
    split='letters',
    train=False,        # Test split
    download=download
)

# Explore a raw sample
index = 90000
img, label = train_dataset[index]
print(f"\n Image type: {type(img)}")  # PIL Image — not yet a tensor


# ==================== STEP 3: APPLY TRANSFORMS ====================
# Pre-computed mean and std for EMNIST Letters dataset
mean = (0.1736,)
std = (0.3317,)

transform = transforms.Compose([
    transforms.ToTensor(),                    # PIL → Tensor, [0,255] → [0,1]
    transforms.Normalize(mean=mean, std=std)  # Standardize
])

# Assign transforms to both datasets
train_dataset.transform = transform
test_dataset.transform = transform

# Verify the transform worked
img_tensor, label = train_dataset[index]


# ==================== STEP 4: FIX EMNIST IMAGE ORIENTATION ====================
#  NOTE: EMNIST images are often rotated/mirrored compared to normal text.
#   This function corrects the orientation so letters look right-side-up.

def correct_image_orientation(image):
    """Rotate and flip an EMNIST image to the correct orientation."""
    rotated = F.rotate(image, 65)   # Rotate to correct angle
    flipped = F.vflip(rotated)      # Flip vertically
    return flipped

img_transformed = correct_image_orientation(img_tensor)
# helper_utils.visualize_image(img_transformed, label)  # Uncomment to see


# ==================== STEP 5: CREATE DATALOADERS ====================
def create_emnist_dataloaders(train_dataset, test_dataset, batch_size=64):
    """
    Creates DataLoader objects for training and testing.

     CONCEPT: DataLoader handles:
      - Batching: Grouping samples into batches for efficient GPU processing
      - Shuffling: Randomizing order each epoch (training only)
      - Parallel loading: Using multiple workers to load data while GPU trains
    """
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True       #  Shuffle training data to prevent order memorization
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False      #  No need to shuffle test data — we're just measuring
    )

    return train_dataloader, test_dataloader


train_loader, test_loader = create_emnist_dataloaders(train_dataset, test_dataset, batch_size=64)

print("--- Train Loader --- \n")
helper_utils.display_data_loader_contents(train_loader)
print("\n--- Test Loader --- \n")
helper_utils.display_data_loader_contents(test_loader)

unittests.exercise_1(create_emnist_dataloaders, data_path)


# ==================== STEP 6: DEFINE THE MODEL ====================
def initialize_emnist_model(num_classes=26):
    """
    Creates a 3-layer neural network for EMNIST letter classification.

     ARCHITECTURE:
      Flatten(28×28) → Linear(784, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 26)

     WHY DEEPER THAN MNIST DIGIT MODEL?
      26 classes is harder than 10 — the model needs more capacity (more neurons)
      to learn the subtle differences between similar letters (e.g., 'a' vs 'd').
    """
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Flatten(),            # [1, 28, 28] → [784]
        nn.Linear(784, 256),     # Hidden layer 1: 256 neurons
        nn.ReLU(),               # Non-linearity
        nn.Linear(256, 128),     # Hidden layer 2: 128 neurons
        nn.ReLU(),               # Non-linearity
        nn.Linear(128, num_classes)  # Output: 26 scores (one per letter)
    )

    # CrossEntropyLoss for multi-class classification
    loss_function = nn.CrossEntropyLoss()

    # Adam optimizer — adaptive learning rate, generally faster than SGD
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, loss_function, optimizer


your_model, loss_func, optimizer = initialize_emnist_model(num_classes=26)
print(f"Your model's architecture:\n\n{your_model}\n")
print(f"Your model's loss function: {loss_func}\n")
print(f"Your model's optimizer:\n\n{optimizer}\n")

unittests.exercise_2(initialize_emnist_model)


# ==================== STEP 7: TRAINING FUNCTION (ONE EPOCH) ====================
def train_epoch(model, loss_function, optimizer, train_loader, device, verbose=True):
    """
    Trains the model for one epoch.

     KEY DETAIL: EMNIST 'letters' labels are 1-indexed (A=1, B=2, ..., Z=26).
      PyTorch's CrossEntropyLoss expects 0-indexed labels (0, 1, ..., 25).
      So we subtract 1: targets = targets - 1
    """
    model.to(device)
    model.train()

    running_loss = 0.0
    num_correct_predictions = 0
    total_predictions = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        #  CRITICAL: Shift labels from 1-indexed to 0-indexed
        targets = targets - 1

        # --- Standard training loop ---
        optimizer.zero_grad()               # Clear gradients
        outputs = model(inputs)             # Forward pass
        loss = loss_function(outputs, targets)  # Compute loss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update weights

        # Track metrics
        running_loss += loss.item()

        _, predicted_indices = outputs.max(1)  # Get predicted class
        num_correct_predictions += predicted_indices.eq(targets).sum().item()
        total_predictions += targets.size(0)

    average_loss = running_loss / len(train_loader)
    accuracy_percentage = (num_correct_predictions / total_predictions) * 100

    if verbose:
        print(f"Epoch Loss (Avg): {average_loss:.3f} | Epoch Acc: {accuracy_percentage:.2f}%")

    return model, average_loss


# Quick test of one epoch
model, loss_function, optimizer = initialize_emnist_model(num_classes=26)
model_one_train_epoch, _ = train_epoch(model, loss_function, optimizer, train_loader, DEVICE)

unittests.exercise_3(train_epoch, model, loss_function, optimizer, train_loader, data_path)


# ==================== STEP 8: EVALUATION FUNCTION ====================
def evaluate(model, test_loader, device, verbose=True):
    """
    Evaluates the model on test data.

     KEY DIFFERENCES from training:
      - model.eval() instead of model.train()
      - torch.no_grad() — no gradient computation
      - No optimizer.step() — weights don't change
    """
    model.eval()
    num_correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  #  Save memory — no gradient tracking needed
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets - 1  # Adjust 1-indexed → 0-indexed

            outputs = model(inputs)
            _, predicted_indices = outputs.max(1)

            num_correct_predictions += predicted_indices.eq(targets).sum().item()
            total_predictions += targets.size(0)

        accuracy_percentage = (num_correct_predictions / total_predictions) * 100

    if verbose:
        print(f'Test Accuracy: {accuracy_percentage:.2f}%')

    return accuracy_percentage


model_evaluate = evaluate(model, test_loader, DEVICE)
unittests.exercise_4(evaluate, model, test_loader, data_path)


# ==================== STEP 9: FULL TRAINING LOOP ====================
def train_and_evaluate(model, train_loader, test_loader, num_epochs, loss_function, optimizer, device):
    """
    Trains and evaluates the model for the specified number of epochs.

     CONCEPT: Each epoch:
      1. Train on all training batches (model learns)
      2. Evaluate on all test batches (measure performance)
      This cycle repeats — loss should decrease, accuracy should increase.
    """
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch+1}")

        # Train for one epoch
        trained_model, _ = train_epoch(model, loss_function, optimizer, train_loader, device)

        # Evaluate on test set
        accuracy = evaluate(trained_model, test_loader, device)

    return trained_model


# --- Run the training ---
NUM_EPOCHS = 10

emnist_model, loss_function, optimizer = initialize_emnist_model(num_classes=26)

trained_model = train_and_evaluate(
    model=emnist_model,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=NUM_EPOCHS,
    loss_function=loss_function,
    optimizer=optimizer,
    device=DEVICE
)

# Evaluate accuracy per letter class
class_accuracies = helper_utils.evaluate_per_class(trained_model, test_loader, DEVICE)

for letter, accuracy in class_accuracies.items():
    print(f"Accuracy for {letter}: {(accuracy*100):.2f} %")

unittests.exercise_5(class_accuracies)

# Save the trained model
trained_file_path = Path.cwd() / 'data/trained_student_model.pth'
helper_utils.save_student_model(model=trained_model, file_path=trained_file_path)


# ==================== STEP 10: DECODE A HIDDEN MESSAGE ====================
#  FUN PART: Use your trained model to read a secret message hidden in images!
#   The model acts as an "OCR" (Optical Character Recognition) system.

message_imgs = helper_utils.load_hidden_message_images(Path.cwd() / 'module2/hidden_message_images.pkl')


def decode_word_imgs(word_imgs, model, device):
    """
    Decodes a word from a list of character images using the trained model.

     PROCESS:
      For each character image:
        1. Add batch dimension: [1, 28, 28] → [1, 1, 28, 28]
        2. Model predicts a class index (0–25)
        3. Convert index to letter: chr(ord('a') + index)
    """
    model.eval()
    decoded_chars = []

    with torch.no_grad():
        for char_img in word_imgs:
            # Add batch dimension and move to device
            char_img = char_img.unsqueeze(0).to(device)

            # Predict the character
            output = model(char_img)
            _, predicted = output.max(1)
            predicted_label = predicted.item()

            # Convert label index to letter (0 → 'a', 1 → 'b', etc.)
            lowercase_char = chr(ord("a") + predicted_label)
            decoded_chars.append(lowercase_char)

    # Join characters into a word
    decoded_word = "".join(decoded_chars)
    return decoded_word


# Decode each sentence from the hidden message images
for sentence_imgs in message_imgs:
    decoded_sentence = []

    for word_imgs in sentence_imgs:
        decoded_word = decode_word_imgs(word_imgs, trained_model, DEVICE)
        decoded_sentence.append(decoded_word)

    print(" ".join(decoded_sentence))

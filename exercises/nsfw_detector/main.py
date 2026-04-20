# main.py

import copy

import torch
from torch import nn
from torch import optim

from data_loader import get_dataloaders
from cnn import CNNBlock, SimpleCNN
import helper_utils

train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    batch_size=32,
    val_fraction=0.15,
    test_fraction=0.2,
    )


print('=== Train Loader ===')
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of samples in train_dataset: {len(train_loader.dataset)}")
print(f"Transforms applied to train_dataset: {train_loader.dataset.transform}")

print('\n=== Test Loader ===')
print(f"Number of batches in test_loader: {len(test_loader)}")
print(f"Number of samples in test_dataset: {len(test_loader.dataset)}")
print(f"Transforms applied to test_dataset: {test_loader.dataset.transform}")


# ==================== VERIFY MODEL ARCHITECTURE ====================
# Quick sanity check: make sure the model produces the right output shape.

# Test CNNBlock
verify_cnn_block = CNNBlock(in_channels=3, out_channels=16)
dummy_input = torch.randn(1, 3, 224, 224)   # 1 image, 3 channels, 224x224
output = verify_cnn_block(dummy_input)
print(f"CNNBlock: Input shape {dummy_input.shape} -> Output shape {output.shape}")
# Output: [1, 16, 112, 112] -- channels doubled, spatial size halved


# Test SimpleCNN
verify_simple_cnn = SimpleCNN(num_classes=15)
dummy_input = torch.randn(64, 3, 224, 224)   # Batch of 64 images
output = verify_simple_cnn(dummy_input)
print(f"SimpleCNN: Input shape {dummy_input.shape} -> Output shape {output.shape}")
# Output: [64, 15] -- one score per class for each image


# ==================== STEP 5: DEFINE LOSS AND OPTIMIZER ====================
# CrossEntropyLoss: standard for multi-class classification
# Adam with weight_decay: weight_decay adds L2 regularization (penalizes large weights)
#   This helps prevent overfitting by encouraging smaller, more distributed weights.


def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


# ==================== VALIDATION FUNCTION ====================

def validate_epoch(model, val_loader, loss_function, device):
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
    epoch_accuracy = 100.0 * correct / total

    return epoch_val_loss, epoch_accuracy


# ==================== FULL TRAINING LOOP WITH BEST MODEL SAVING ====================
# KEY CONCEPT: We track the best validation accuracy and save that model's weights.
#   The model at the LAST epoch isn't always the best -- it might have started
#   overfitting. We restore the best checkpoint at the end.

def training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
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
model = SimpleCNN(num_classes=num_classes)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


trained_model, training_metrics = training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    num_epochs=5,
    device=device
)

# Plot training curves (loss and accuracy over epochs)
helper_utils.plot_training_metrics(training_metrics)




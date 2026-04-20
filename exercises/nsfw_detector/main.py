# ============================================================
# main.py - OPTIMIZED FOR YOUR GTX 1650 + Ryzen 5 5600X
# ============================================================

import copy
import torch
from torch import nn
from torch import optim
from torch.amp import autocast, GradScaler

from data_loader import get_dataloaders
from cnn import SimpleCNN
import helper_utils

# ==================== HYPERPARAMETERS (Tuned for your PC) ====================
NUM_EPOCHS = 40                    # More epochs = much better accuracy
BATCH_SIZE = 64                    # Best size for GTX 1650 4GB VRAM
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1              # Helps accuracy on your GPU

train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    batch_size=BATCH_SIZE,
    val_fraction=0.15,
    test_fraction=0.2,
)

print(f"✅ Using batch size {BATCH_SIZE} (optimized for GTX 1650 4GB)")

# ==================== DEVICE & SPEED OPTIMIZATIONS ====================
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True        # Big speed boost on GTX 1650
    print("🚀 Using NVIDIA GTX 1650 with cuDNN benchmark + Mixed Precision")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS")
else:
    device = torch.device("cpu")
    print("Using CPU (slow)")

# ==================== MODEL, LOSS, OPTIMIZER, SCHEDULER ====================
model = SimpleCNN(num_classes=num_classes)

loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

scaler = GradScaler()                                 # Mixed Precision scaler

# Optional: torch.compile (extra speed on PyTorch 2.0+)
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("⚡ Model compiled with torch.compile()")
except Exception:
    pass

# ==================== TRAINING FUNCTIONS (same as before but faster) ====================
def train_epoch(model, train_loader, loss_function, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            outputs = model(images)
            loss = loss_function(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(train_loader.dataset)


def validate_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_val_loss = 0.0
    correct = total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type=device.type):
                outputs = model(images)
                val_loss = loss_function(outputs, labels)

            running_val_loss += val_loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (running_val_loss / len(val_loader.dataset)), (100.0 * correct / total)


# ==================== TRAINING LOOP WITH EARLY STOPPING ====================
def training_loop(model, train_loader, val_loader, loss_function, optimizer, scheduler,
                  num_epochs, device, scaler):
    model.to(device)
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    patience = 8
    patience_counter = 0

    train_losses, val_losses, val_accuracies = [], [], []

    print("\n" + "="*70)
    print("🚀 TRAINING STARTED - Optimized for your GTX 1650")
    print("="*70)

    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device, scaler)
        epoch_val_loss, epoch_accuracy = validate_epoch(model, val_loader, loss_function, device)

        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_accuracy:6.2f}% | LR: {current_lr:.6f}")

        scheduler.step()

        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("  → New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break

    print("\n" + "="*70)
    print(f"✅ TRAINING FINISHED - Best Val Accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}")
    print("="*70)

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, [train_losses, val_losses, val_accuracies]


# ==================== RUN TRAINING ====================
trained_model, training_metrics = training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=NUM_EPOCHS,
    device=device,
    scaler=scaler
)

helper_utils.plot_training_metrics(training_metrics)

# ==================== SAVE MODEL ====================
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'num_classes': num_classes,
    'val_accuracy': max(training_metrics[2]),
    'epoch': training_metrics[2].index(max(training_metrics[2])) + 1,
}, "best_simple_cnn_gtx1650.pth")

print(f"\n✅ Model saved as 'best_simple_cnn_gtx1650.pth'")
print(f"   Best Validation Accuracy: {max(training_metrics[2]):.2f}%")
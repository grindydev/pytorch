"""
Lesson 2 - Module 1: Learning Rate Schedulers (scheduler/main.py)
=================================================================
WHAT YOU'LL LEARN:
  * Why a fixed learning rate is suboptimal
  * Three popular LR schedulers:
    - StepLR:          Reduce LR by a factor every N epochs
    - CosineAnnealingLR: Smoothly oscillate LR following a cosine curve
    - ReduceLROnPlateau: Reduce LR when validation metric stops improving
  * How schedulers help escape local minima and fine-tune convergence

KEY CONCEPT:
  Think of learning rate like step size when hiking down a mountain:
  - Large steps (high LR):  Fast progress initially, but you overshoot the bottom
  - Small steps (low LR):   Precise, but painfully slow
  - SCHEDULER:              Start with large steps, gradually take smaller ones
                           as you get closer to the bottom (minimum loss)
"""

import sys
import time
import warnings

# Suppress noisy warnings for cleaner output
class BlackHole:
    def write(self, message):
        pass
    def flush(self):
        pass
sys.stderr = BlackHole()
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import helper_utils
helper_utils.set_seed(42)

# Device selection (CPU for stability in this exercise)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device using: {device}")


# ==================== MODEL DEFINITION ====================
class SimpleCNN(nn.Module):
    """Same 2-block CNN from the learning_rate exercise."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==================== TRAINING WITH SCHEDULER ====================
def train_and_evaluate_with_scheduler(model, optimizer, scheduler, device, n_epochs=25, batch_size=128):
    """
    Trains a model with a learning rate scheduler.

    KEY DETAIL: After each epoch, we call scheduler.step() to update the LR.
    The scheduler modifies the optimizer's learning rate in-place.

    Returns: Dictionary with train/val loss, accuracy, and learning rate history.
    """
    helper_utils.set_seed(10)

    loss_fn = nn.CrossEntropyLoss()
    train_loader, val_loader = helper_utils.get_dataset_dataloaders(batch_size=batch_size)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    pbar = helper_utils.NestedProgressBar(
        total_epochs=n_epochs,
        total_batches=len(train_loader),
        epoch_message_freq=5,
        mode="train",
    )

    for epoch in range(n_epochs):
        pbar.update_epoch(epoch + 1)

        # --- Train ---
        train_loss, train_acc = helper_utils.train_epoch(
            model, train_loader, optimizer, loss_fn, device, pbar
        )
        # --- Validate ---
        val_loss, val_acc = helper_utils.evaluate_epoch(
            model, val_loader, loss_fn, device
        )

        # Get the current learning rate BEFORE updating
        current_lr = scheduler.get_last_lr()[0]

        # --- UPDATE LEARNING RATE ---
        # KEY CONCEPT: Different schedulers need different step() calls:
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau needs the metric to monitor
            # If val_acc doesn't improve for `patience` epochs, reduce LR
            scheduler.step(val_acc)
        else:
            # StepLR and CosineAnnealingLR just need step()
            scheduler.step()

        pbar.maybe_log_epoch(
            epoch=epoch+1,
            message=f"At epoch {epoch+1}: Train loss: {train_loss:.4f}, "
                    f"Train acc: {train_acc:.4f}, LR: {current_lr:.6f}"
        )
        pbar.maybe_log_epoch(
            epoch=epoch+1,
            message=f"At epoch {epoch+1}: Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

    pbar.close('Training complete!')
    return history


# ==================== COMPARING THREE SCHEDULERS ====================
n_epochs = 25
batch_size = 128
initial_lr = 0.005  # Start with a relatively high LR

# --- 1. StepLR ---
# KEY CONCEPT: Reduce LR by gamma every step_size epochs.
# Example: LR starts at 0.005
#   Epochs 1-10:  LR = 0.005
#   Epochs 11-20: LR = 0.005 * 0.2 = 0.001  (reduced by 80%)
#   Epochs 21-25: LR = 0.001 * 0.2 = 0.0002
print(f"------Training with StepLR------")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

history_LR = train_and_evaluate_with_scheduler(
    model, optimizer, scheduler_step, device, n_epochs=n_epochs, batch_size=batch_size
)

# --- 2. CosineAnnealingLR ---
# KEY CONCEPT: Smoothly decrease LR following a cosine curve from initial_lr
# down to eta_min, then back up. This creates a smooth, gradual reduction.
# T_max = period (number of epochs for one full cosine cycle)
print(f"------Training with CosineAnnealingLR------")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_epochs, eta_min=0.0002
)

history_cosine = train_and_evaluate_with_scheduler(
    model, optimizer, scheduler_cosine, device, n_epochs=n_epochs, batch_size=batch_size
)

# --- 3. ReduceLROnPlateau ---
# KEY CONCEPT: "Adaptive" -- only reduce LR when progress stalls.
# If validation accuracy doesn't improve for `patience` epochs, reduce LR by `factor`.
# This is the most "intelligent" scheduler -- it reacts to actual training progress.
print(f"------Training with ReduceLROnPlateau------")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max',   # 'max' because we monitor accuracy (higher is better)
    factor=0.2,               # Reduce LR to 20% of current value
    patience=3                # Wait 3 epochs with no improvement before reducing
)

history_plateau = train_and_evaluate_with_scheduler(
    model, optimizer, scheduler_plateau, device, n_epochs=n_epochs, batch_size=batch_size
)


# ==================== VISUALIZE COMPARISON ====================
labels = ['StepLR', 'CosineAnnealingLR', 'ReducedLRonPlateau']
colors = ['green', 'blue', 'purple']
training_curves_new = [history_LR, history_cosine, history_plateau]

# Plot how learning rate changes over epochs for each scheduler
helper_utils.plot_learning_rates_curves(training_curves_new, colors, labels)

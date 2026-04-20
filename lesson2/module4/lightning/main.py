"""
Lesson 2 - Module 4: PyTorch Lightning Basics (lightning/main.py)
=================================================================
WHAT YOU'LL LEARN:
  * What PyTorch Lightning is and why it simplifies training code
  * LightningDataModule: clean data loading in one class
  * LightningModule: model + training logic in one class
  * Built-in training loop: no more manual epoch loops
  * Profiling: measuring which operations are slowest
  * Reducing model complexity to improve training speed

KEY CONCEPT:
  PyTorch Lightning removes boilerplate code. Instead of writing the same
  training loop manually every time, you define:
    - training_step():   What happens for one batch during training
    - validation_step(): What happens for one batch during validation
    - configure_optimizers(): Which optimizer to use

  Lightning handles the rest: epoch loops, gradient zeroing, backprop,
  device management, logging, checkpointing, etc.
"""

import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import schedule
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from pathlib import Path

import helper_utils

torch.set_float32_matmul_precision('medium')


# ==================== PART 1: DATA MODULE ====================
# KEY CONCEPT: LightningDataModule encapsulates ALL data logic in one class.
#   - prepare_data(): Download data (called once, on 1 GPU)
#   - setup(): Create dataset objects (called on each GPU)
#   - train_dataloader() / val_dataloader(): Return DataLoaders

class CIFAR10DataModule(pl.LightningDataModule):
    """Encapsulates CIFAR-10 data loading logic."""

    def __init__(self, data_dir='./data', batch_size=128, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        """Download data if not present. Called ONCE on a single GPU."""
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Create dataset objects. Called on each GPU."""
        self.cifar_train = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
        self.cifar_val = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)


# ==================== PART 2: MODEL MODULE ====================
# KEY CONCEPT: LightningModule replaces nn.Module for training.
# It combines the model architecture with training/validation logic.

class CIFAR10LightningModule(pl.LightningModule):
    """
    A CNN for CIFAR-10 using Lightning.

    KEY METHODS TO IMPLEMENT:
      - forward():           The model's forward pass
      - training_step():     Loss computation for one batch
      - validation_step():   Metrics for one validation batch
      - configure_optimizers(): Return the optimizer
    """

    def __init__(self, learning_rate=1e-3, weight_decay=0.01,
                 conv_channels=(256, 512, 1024), linear_features=2048,
                 num_classes=10):
        super().__init__()
        # save_hyperparameters() stores all args in self.hparams
        # and logs them automatically -- very convenient for tracking experiments
        self.save_hyperparameters()

        # After 3 MaxPool layers: 32x32 -> 16x16 -> 8x8 -> 4x4
        flattened_size = self.hparams.conv_channels[-1] * 4 * 4

        self.model = nn.Sequential(
            # 3 convolutional blocks
            nn.Conv2d(3, self.hparams.conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(self.hparams.conv_channels[0], self.hparams.conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(self.hparams.conv_channels[1], self.hparams.conv_channels[2], kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            # Classifier
            nn.Flatten(),
            nn.Linear(flattened_size, self.hparams.linear_features),
            nn.ReLU(),
            nn.Linear(self.hparams.linear_features, self.hparams.num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx=None):
        """
        ONE training step: forward pass + loss computation.
        Lightning handles zero_grad, backward, and optimizer.step automatically!
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)

        # self.log() sends metrics to Lightning's logging system
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.train_accuracy(outputs, labels)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss  # Lightning uses this for backpropagation

    def validation_step(self, batch, batch_idx=None):
        """ONE validation step: forward pass + metric computation."""
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_accuracy(outputs, labels)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Return the optimizer. AdamW adds decoupled weight decay to Adam."""
        return optim.AdamW(self.parameters(),
                           lr=self.hparams.learning_rate,
                           weight_decay=self.hparams.weight_decay)


# ==================== PART 3: TRAINING (BASELINE) ====================
data_dir = Path.cwd() / 'data/cifar10'
dm_loader = CIFAR10DataModule(data_dir=data_dir, num_workers=0)

model_baseline = CIFAR10LightningModule()
baseline_results = helper_utils.run_full_training(model_baseline, dm_loader)

print("\nTraining Complete!")
print(f"  Training Accuracy:   {baseline_results['train_accuracy']}%")
print(f"  Validation Accuracy: {baseline_results['val_accuracy']}%")


# ==================== PART 4: PROFILING ====================
# KEY CONCEPT: Profiling measures how much time each operation takes.
# This helps you find bottlenecks and optimize your model.

log_dir = "./outputs/profiler"

profiler = PyTorchProfiler(
    dirpath=log_dir,
    filename="profile_report",
    # Schedule: skip first 2 steps, warm up for 2, then profile for 10
    schedule=schedule(wait=2, warmup=2, active=10, repeat=1),
    profile_memory=True
)

# Trainer is Lightning's main class -- it orchestrates everything
trainer = pl.Trainer(
    profiler=profiler,
    max_steps=14,
    accelerator="auto",   # Automatically pick GPU/CPU
    devices=1,
    logger=False,
    enable_model_summary=False,
    enable_checkpointing=False
)

trainer.fit(model_baseline, dm_loader)
print("\nProfiling Complete!")

# Show the top 10 slowest operations
helper_utils.display_profiler_logs(profiler, head=10)
helper_utils.display_model_computation_logs(profiler)


# ==================== PART 5: EFFICIENT MODEL ====================
# KEY CONCEPT: The baseline model is OVER-SIZED for CIFAR-10.
# Fewer channels = less computation = faster training (often with similar accuracy).

# Configure profiler for the efficient model
profiler_efficient = PyTorchProfiler(
    dirpath=log_dir,
    filename='light_exercise',
    schedule=schedule(wait=2, warmup=2, active=10, repeat=1),
    profile_memory=True
)

trainer_efficient = pl.Trainer(
    profiler=profiler_efficient,
    max_steps=14,
    accelerator="auto",
    devices=1,
    logger=False,
    enable_model_summary=False,
    enable_checkpointing=False
)

# Much smaller model: 256/512/1024 channels -> 32/64/128
model_efficient = CIFAR10LightningModule(
    conv_channels=(32, 64, 128),
    linear_features=512
)

trainer_efficient.fit(model_efficient, dm_loader)
print("\nEfficient Profiling Complete!")

helper_utils.display_profiler_logs(profiler_efficient, head=10)

# Compare baseline vs. efficient model
helper_utils.display_comparison_report(profiler, profiler_efficient)

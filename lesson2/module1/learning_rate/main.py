"""
Lesson 2 - Module 1: Learning Rate & Evaluation Metrics (learning_rate/main.py)
================================================================================
WHAT YOU'LL LEARN:
  * How learning rate affects training: too low = slow, too high = unstable
  * Evaluation metrics beyond accuracy: Precision, Recall, F1 Score
  * When accuracy is misleading: imbalanced datasets
  * Using torchmetrics for clean metric computation
  * How batch size interacts with learning rate on imbalanced data

KEY CONCEPT:
  Learning rate is THE most important hyperparameter. It controls how big
  each weight update step is. Finding the right learning rate can make or
  break your model's performance.

  ACCURACY is not enough! For imbalanced datasets (e.g., 99 cats, 1 dog),
  a model that always predicts "cat" has 99% accuracy but is useless.
  Precision, Recall, and F1 give a much better picture.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

import helper_utils
from pathlib import Path
helper_utils.set_seed(42)

# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

data_path = Path.cwd() / "data/rich_cifar10"
data_path_imbalanced = Path.cwd() / "data/cifar10_3class_imbalanced"


# ==================== STEP 1: DEFINE THE CNN ====================
class SimpleCNN(nn.Module):
    """
    A 2-block CNN for CIFAR-10 classification.

    Architecture:
      Input [3, 32, 32]
        -> Conv(3->16) -> ReLU -> MaxPool -> [16, 16, 16]
        -> Conv(16->32) -> ReLU -> MaxPool -> [32, 8, 8]
        -> Flatten -> FC(2048, 64) -> ReLU -> Dropout -> FC(64, 10)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [batch, 16, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))   # [batch, 32, 8, 8]
        x = x.view(-1, 32 * 8 * 8)             # Flatten: [batch, 2048]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==================== STEP 2: BASELINE ACCURACY FUNCTION ====================
def evaluate_accuracy(model, val_loader, device):
    """Simple accuracy: correct predictions / total predictions."""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)  # Class with highest score
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples


# ==================== STEP 3: TEST DIFFERENT LEARNING RATES ====================
# KEY CONCEPT: Learning rate controls the step size in gradient descent.
#   Too LOW (1e-5):  Model barely learns -- loss decreases very slowly
#   Too HIGH (0.1):  Model overshoots minima -- loss oscillates or diverges
#   JUST RIGHT (1e-3): Model converges quickly and stably

def train_and_evaluate(learning_rate, device, n_epochs=25, batch_size=128):
    """Train a model with a specific learning rate and return validation accuracy."""
    helper_utils.set_seed(42)

    model = SimpleCNN().to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader, val_dataloader = helper_utils.get_dataset_dataloaders(
        batch_size=batch_size, data_path=data_path, data_path_imbalanced=data_path_imbalanced
    )

    helper_utils.train_model(
        model=model, optimizer=optimizer, loss_fcn=loss_fcn,
        train_dataloader=train_dataloader, device=device, n_epochs=n_epochs
    )

    accuracy = evaluate_accuracy(model=model, val_loader=val_dataloader, device=device)
    print(f"Learning Rate: {learning_rate}, Accuracy: {accuracy:.4f}")
    return accuracy


# Test a range of learning rates from very small to very large
learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]

accuracies = []
for lr in learning_rates:
    acc = train_and_evaluate(learning_rate=lr, device=device)
    accuracies.append(acc)

# Plot: accuracy vs. learning rate (you'll see a "sweet spot" curve)
helper_utils.plot_results(learning_rates, accuracies)


# ==================== STEP 4: BEYOND ACCURACY -- PRECISION, RECALL, F1 ====================
# KEY CONCEPT: Why accuracy is not enough:
#
#   PRECISION: Of all samples predicted as class X, how many were actually X?
#              = TP / (TP + FP)  -- "How trustworthy are positive predictions?"
#
#   RECALL:    Of all actual class X samples, how many did we find?
#              = TP / (TP + FN)  -- "How complete are our predictions?"
#
#   F1 SCORE:  Harmonic mean of precision and recall (balances both)
#              = 2 * (precision * recall) / (precision + recall)
#
#   MACRO average: Compute metric for each class independently, then average.
#                  Treats all classes equally, regardless of class size.

def evaluate_metrics(model, val_dataloader, device, num_classes=10):
    """
    Evaluates model using Accuracy, Precision, Recall, and F1 Score.
    Uses the torchmetrics library for clean, batch-wise metric computation.
    """
    model.eval()

    # Initialize metrics (macro average: each class weighted equally)
    accuracy_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)

    precision_metric = torchmetrics.Precision(
        task="multiclass", average='macro', num_classes=num_classes
    ).to(device)

    recall_metric = torchmetrics.Recall(
        task="multiclass", average='macro', num_classes=num_classes
    ).to(device)

    f1_metric = torchmetrics.F1Score(
        task="multiclass", num_classes=num_classes
    ).to(device)

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Update each metric with this batch's predictions
            # torchmetrics accumulates results internally
            accuracy_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)

    # .compute() returns the final metric over all batches
    accuracy = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()

    return accuracy, precision, recall, f1


# Quick test with an untrained model
model = SimpleCNN().to(device)
train_dataloader, val_dataloader = helper_utils.get_dataset_dataloaders(
    batch_size=128, data_path=data_path, data_path_imbalanced=data_path_imbalanced
)

accuracy, precision, recall, f1 = evaluate_metrics(
    model=model, val_dataloader=val_dataloader, device=device
)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
      f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")


# ==================== STEP 5: METRICS VS. LEARNING RATE ====================
def train_and_evaluate_metrics(learning_rate, device, n_epochs=25, batch_size=128, imbalanced=False):
    """Train and evaluate, returning accuracy, precision, recall, and F1."""
    helper_utils.set_seed(42)

    model = SimpleCNN().to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader, val_dataloader = helper_utils.get_dataset_dataloaders(
        batch_size=batch_size, imbalanced=imbalanced,
        data_path=data_path, data_path_imbalanced=data_path_imbalanced
    )

    helper_utils.train_model(
        model=model, optimizer=optimizer, loss_fcn=loss_fcn,
        train_dataloader=train_dataloader, device=device, n_epochs=n_epochs
    )

    accuracy, precision, recall, f1 = evaluate_metrics(model, val_dataloader, device)
    print(f"LR: {learning_rate}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, "
          f"Rec: {recall:.4f}, F1: {f1:.4f}")
    return accuracy, precision, recall, f1


# Collect metrics across different learning rates
dict_metrics = []
for lr in learning_rates:
    acc, prec, rec, f1 = train_and_evaluate_metrics(
        learning_rate=lr, device=device, n_epochs=25, batch_size=128
    )
    dict_metrics.append({
        "learning_rate": lr,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    })

df_metrics = pd.DataFrame(dict_metrics)
helper_utils.plot_metrics_vs_learning_rate(df_metrics)


# ==================== STEP 6: BATCH SIZE EFFECTS ON IMBALANCED DATA ====================
# KEY CONCEPT: On imbalanced datasets, batch size matters more.
# Small batches may not contain samples from rare classes in every batch.

dict_metrics = []
batch_sizes = [32, 64, 128]
lr = 0.001
imbalanced = True  # Use the imbalanced version of the dataset

for bs in batch_sizes:
    acc, prec, rec, f1 = train_and_evaluate_metrics(
        batch_size=bs, n_epochs=25, learning_rate=lr, device=device, imbalanced=imbalanced
    )
    dict_metrics.append({
        "batch_size": bs,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    })

df_metrics = pd.DataFrame(dict_metrics)
helper_utils.plot_metrics_vs_batch_size(df_metrics)

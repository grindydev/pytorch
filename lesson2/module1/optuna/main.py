"""
Lesson 2 - Module 1: Hyperparameter Tuning with Optuna (optuna/main.py)
=======================================================================
WHAT YOU'LL LEARN:
  * What hyperparameter tuning is and why it matters
  * Using Optuna to automatically search for the best model architecture
  * Building a flexible CNN whose architecture is defined by hyperparameters
  * The objective function pattern: trial -> hyperparameters -> accuracy
  * Visualizing optimization history and parameter importance

KEY CONCEPT:
  Instead of manually guessing good hyperparameters (number of layers, filters,
  kernel sizes, dropout rate, etc.), Optuna systematically tries different
  combinations and finds the best one automatically.

  HOW OPTUNA WORKS:
  1. You define an "objective" function that takes a trial and returns a score
  2. Each trial samples a new combination of hyperparameters
  3. Optuna uses the results to guide future trials (smarter than random search)
  4. After N trials, it returns the best combination found
"""

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import helper_utils
import torch.nn.functional as F
from pprint import pprint

helper_utils.set_seed(15)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(device)


# ==================== STEP 1: FLEXIBLE CNN ARCHITECTURE ====================
# KEY CONCEPT: Unlike previous models with fixed architectures, this CNN's
# structure is DYNAMIC -- the number of layers, filters, and kernel sizes
# are determined by hyperparameters that Optuna will tune.

class FlexibleCNN(nn.Module):
    """
    A CNN whose architecture is defined by hyperparameters.

    The feature extractor has `n_layers` convolutional blocks, each with
    a configurable number of filters and kernel size. The classifier is
    built dynamically during the first forward pass (because the flattened
    size depends on the number of pooling operations).
    """

    def __init__(self, n_layers, n_filters, kernel_sizes, dropout_rate, fc_size):
        """
        Args:
            n_layers:     Number of conv blocks (1-3)
            n_filters:    List of output channels for each conv layer [16-128]
            kernel_sizes: List of kernel sizes for each conv layer [3 or 5]
            dropout_rate: Dropout probability for the classifier [0.1-0.5]
            fc_size:      Number of neurons in the hidden FC layer [64-256]
        """
        super(FlexibleCNN, self).__init__()

        # Build convolutional blocks dynamically
        blocks = []
        in_channels = 3  # Start with RGB input

        for i in range(n_layers):
            out_channels = n_filters[i]
            kernel_size = kernel_sizes[i]
            # "same" padding: keeps spatial dimensions unchanged before pooling
            padding = (kernel_size - 1) // 2

            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)  # Halve spatial dimensions
            )
            blocks.append(block)
            in_channels = out_channels  # Output channels become next block's input

        self.features = nn.Sequential(*blocks)
        self.dropout_rate = dropout_rate
        self.fc_size = fc_size
        self.classifier = None  # Built dynamically in forward()

    def _create_classifier(self, flattened_size, device):
        """Dynamically builds the classifier when we first know the feature size."""
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(flattened_size, self.fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_size, 10)  # 10 classes (CIFAR-10)
        ).to(device)

    def forward(self, x):
        device = x.device
        x = self.features(x)                       # Extract features
        flattened = torch.flatten(x, 1)             # Flatten for FC layers
        flattened_size = flattened.size(1)

        if self.classifier is None:
            # First forward pass: build classifier with the correct input size
            self._create_classifier(flattened_size, device)

        return self.classifier(flattened)


# ==================== STEP 2: DEFINE THE OBJECTIVE FUNCTION ====================
# KEY CONCEPT: This is the function Optuna will call repeatedly.
# Each call:
#   1. Samples new hyperparameters from the search space
#   2. Builds a model with those hyperparameters
#   3. Trains the model
#   4. Returns the validation accuracy (what Optuna tries to maximize)

def objective(trial, device):
    """
    Optuna objective: sample hyperparameters, train, return accuracy.
    """
    # --- Sample architecture hyperparameters ---
    # trial.suggest_int: pick an integer in the given range
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # Each layer can have a different number of filters
    n_filters = [trial.suggest_int(f"n_filters_{i}", 16, 128) for i in range(n_layers)]

    # Each layer can have kernel size 3 or 5
    kernel_sizes = [trial.suggest_categorical(f"kernel_size_{i}", [3, 5]) for i in range(n_layers)]

    # --- Sample classifier hyperparameters ---
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_size = trial.suggest_int("fc_size", 64, 256)

    # --- Build model ---
    model = FlexibleCNN(n_layers, n_filters, kernel_sizes, dropout_rate, fc_size).to(device)

    # IMPORTANT: Initialize the dynamic classifier by passing a dummy input
    # Otherwise the optimizer won't know about all parameters
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    model(dummy_input)

    # --- Train ---
    learning_rate = 0.001
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 128
    train_loader, val_loader = helper_utils.get_dataset_dataloaders(batch_size=batch_size)

    n_epochs = 10
    helper_utils.train_model(
        model=model, optimizer=optimizer, train_dataloader=train_loader,
        n_epochs=n_epochs, loss_fcn=loss_fcn, device=device
    )

    # --- Evaluate ---
    accuracy = helper_utils.evaluate_accuracy(model, val_loader, device)
    return accuracy


# ==================== STEP 3: RUN THE OPTIMIZATION ====================
# Create an Optuna study that tries to MAXIMIZE accuracy
study = optuna.create_study(direction='maximize')

# Run 20 trials (each trial trains a model from scratch)
# In practice, you'd use more trials (50-100+) for better results
n_trials = 20
study.optimize(lambda trial: objective(trial, device), n_trials=n_trials)

# View all trial results
df = study.trials_dataframe()

# Print the best result
best_trial = study.best_trial
print("Best trial:")
print(f"  Value (Accuracy): {best_trial.value:.4f}")
print("  Hyperparameters:")
pprint(best_trial.params)


# ==================== STEP 4: VISUALIZE RESULTS ====================
# How accuracy improved over trials
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.show()

# Which hyperparameters mattered most?
optuna.visualization.matplotlib.plot_param_importances(study)
plt.show()

# Parallel coordinate plot: see all hyperparameter combinations at once
ax = optuna.visualization.matplotlib.plot_parallel_coordinate(
    study, params=['n_layers', 'n_filters_0', 'kernel_size_0', 'dropout_rate', 'fc_size']
)
fig = ax.figure
fig.set_size_inches(12, 6, forward=True)
fig.tight_layout()

"""
Lesson 3 - Module 4: Pruning -- Optional Exercise
===================================================

WHY THIS MATTERS:
  This optional file lets you practice pruning on a real model (fruit/vegetable
  classifier) and compare accuracy before and after pruning.

WHAT YOU'LL LEARN:
  * Applying pruning to a production model
  * Measuring sparsity and accuracy impact
  * Comparing pruned vs unpruned model performance

KEY CONCEPTS:
  Sparsity -- Percentage of weights that are zero
  Fine-tuning after pruning -- Retrain to recover accuracy

HOW IT FITS:
  Optional companion to pruning/main.py. Complete main.py first to understand
  the pruning theory, then run this for hands-on practice.

PREREQUISITES:
  Complete pruning/main.py first.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pathlib import Path

import helper_utils

# ====================== DEVICE SETUP ======================
# LEARNING: We automatically choose the fastest device available on your Mac.
# MPS = Apple Silicon GPU → much faster than CPU for training.
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print(" Using MPS — Apple Silicon GPU acceleration!")
else:
    DEVICE = torch.device('cpu')
print(f"Using Device: {DEVICE}")

# ====================== DATA LOADING ======================
# LEARNING: Load the real fruit & vegetable dataset (28 classes).
# We use the same data for both unpruned and pruned models so the comparison is fair.
dataset_path = Path.cwd() / "data/fruit_and_vegetable_subset"
train_loader, val_loader = helper_utils.get_dataloaders(dataset_path)

# LEARNING: Get number of classes dynamically (better than hard-coding).
num_classes = len(train_loader.dataset.subset.dataset.classes)
print(f"Number of classes: {num_classes}")

num_epochs = 2   # We train both models for the same number of epochs

# ===================================================================
# PART 1: TRAIN THE UNPRUNED (BASELINE) MODEL
# ===================================================================
print("\n=== Training UNPRUNED Model (Baseline) ===")

model_path = Path.cwd() / 'data/pretrained_resnet18_weights/resnet18-f37072fd.pth'

# LEARNING: Load fresh ResNet18 with ImageNet pre-trained weights (transfer learning).
resnet18_model_unpruned = helper_utils.load_resnet18(path=model_path)

# LEARNING: Replace the final layer so the model outputs 28 classes instead of 1000.
model_unpruned = helper_utils.replace_final_layer(resnet18_model_unpruned, num_classes)

# LEARNING: Train the normal dense model. This is our reference point.
trained_unpruned_model, unpruned_metrics = helper_utils.training_loop(
    model_unpruned, train_loader, val_loader, num_epochs, DEVICE, num_classes
)

helper_utils.save_unpruned_model_and_metrics(trained_unpruned_model, unpruned_metrics)

print(f"""
-- After Training for {num_epochs} Epoch(s) --
Final Accuracy:           {unpruned_metrics['accuracy']:.2f}%
Final Precision (Macro):  {unpruned_metrics['precision']:.4f}
Final Recall (Macro):     {unpruned_metrics['recall']:.4f}
Final F1-Score (Macro):   {unpruned_metrics['f1_score']:.4f}""")

# ===================================================================
# PART 2: TRAIN THE PRUNED MODEL
# ===================================================================
print("\n=== Training PRUNED Model ===")

# LEARNING: Load a fresh copy of ResNet18. We will prune this version.
resnet18_model_pruned = helper_utils.load_resnet18(path=model_path)
model_pruned = helper_utils.replace_final_layer(resnet18_model_pruned, num_classes)

# -------------------------------------------------------------------
def analyze_model_sparsity(model):
    """
    LEARNING: This function counts how many weights are exactly zero.
    This is the real measure of "sparsity" after pruning.
    """
    total_params = 0
    total_zero_params = 0

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            total_params += module.weight.nelement()          # total weights
            total_zero_params += torch.sum(module.weight == 0).item()

    if total_params > 0:
        sparsity = 100. * float(total_zero_params) / float(total_params)
        print(f"Model Sparsity: {sparsity:.2f}%")
        print(f"Total parameters (in weighted layers): {total_params}")
        print(f"Total zero parameters: {total_zero_params}")
    else:
        print("No weighted layers found.")
# -------------------------------------------------------------------

# --- Analysis BEFORE any pruning ---
print("\n--- Analysis Before Pruning ---")
analyze_model_sparsity(model_pruned)

# LEARNING: Show some actual weights so you can see the difference later.
print("\n--- conv1 Layer Weights BEFORE Pruning ---\n")
helper_utils.show_weights(model_pruned, ['conv1'])

# LEARNING: Collect every Conv2d and Linear layer we want to prune.
# We skip the final 'fc' layer because we usually keep the classifier dense.
parameters_to_prune = []
for module_name, module in model_pruned.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and 'fc' not in module_name:
        parameters_to_prune.append((module, 'weight'))

# ====================== GLOBAL UNSTRUCTURED PRUNING ======================
# LEARNING: global_unstructured removes the smallest 50% of weights across the WHOLE model.
# This is more effective than pruning each layer separately.
print("\n--- Applying 50% Global Unstructured Pruning ---")
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,   # removes weights with smallest absolute values
    amount=0.5,                            # 50% of all weights will become zero
)
print("Global pruning applied (50% of weights set to zero).")

# --- Analysis AFTER pruning (still temporary) ---
print("\n--- Analysis After Pruning (temporary stage) ---")
analyze_model_sparsity(model_pruned)

print("\n--- conv1 Layer Weights AFTER Temporary Pruning ---\n")
helper_utils.show_weights(model_pruned, ['conv1'])

# ====================== FINE-TUNE THE PRUNED MODEL ======================
# LEARNING: After pruning, we fine-tune the model so it can recover accuracy.
trained_pruned_model, pruned_metrics = helper_utils.training_loop(
    model_pruned, train_loader, val_loader, num_epochs, DEVICE, num_classes
)

# ====================== MAKE PRUNING PERMANENT ======================
# LEARNING: prune.remove() bakes the zeros permanently into the weight tensor
# and removes the temporary mask. This is required before saving/exporting.
print("\n--- Making Pruning Permanent ---")
for module, param_name in parameters_to_prune:
    prune.remove(module, param_name)
print("Pruning is now permanent (zeros are baked into the weights).")

# --- Final analysis after permanent pruning ---
print("\n--- Final Analysis of Trained Model (Post-Permanent Pruning) ---")
analyze_model_sparsity(trained_pruned_model)

print("\n--- conv1 Layer Weights AFTER Permanent Pruning ---\n")
helper_utils.show_weights(trained_pruned_model, ['conv1'])

# Save the final pruned model
helper_utils.save_pruned_model_and_metrics(trained_pruned_model, pruned_metrics)

# ====================== COMPARISON REPORT ======================
# LEARNING: This helper shows you the real difference between the two models.
helper_utils.comparison_report(
    unpruned_state_dict_path="models/pruning/unpruned_model.pth",
    unpruned_metrics_path="outputs/unpruned_metrics.pkl",
    pruned_state_dict_path="models/pruning/pruned_model.pth",
    pruned_metrics_path="outputs/pruned_metrics.pkl",
    num_epochs=num_epochs,
    device=DEVICE
)
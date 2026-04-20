"""
Lesson 2 - Module 2: Transfer Learning (transfer_learning/main.py)
===================================================================
WHAT YOU'LL LEARN:
  * What transfer learning is and why it's so powerful
  * Three transfer learning strategies:
    1. Feature Extraction: Freeze all pre-trained layers, train only new head
    2. Fine-Tuning:        Unfreeze the top layers, train them + new head
    3. Full Retraining:    Unfreeze everything, train the whole model
  * Loading pre-trained models from torchvision (MobileNetV3, ResNet18)
  * Freezing layers with requires_grad = False
  * Replacing the final classification layer for a new task

KEY CONCEPT:
  Transfer learning: Instead of training from scratch (random weights), start
  with a model pre-trained on a large dataset (e.g., ImageNet with 14M images).
  The pre-trained model already knows how to detect edges, textures, shapes --
  you just need to adapt the final layer to YOUR specific task.

  WHY IT WORKS: Early layers learn generic features (edges, colors) useful for
  ANY image task. Only the later layers need task-specific adjustments.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

import helper_utils
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

emnist_data_path = Path.cwd() / 'data/EMNIST_data'


# ==================== STEP 1: DATA PREPARATION ====================
# KEY CONCEPT: Pre-trained models expect specific input format:
#   - 3 color channels (RGB)
#   - Size 224x224 (standard for ImageNet-trained models)
#   - Normalized with ImageNet mean and std
#
# EMNIST images are 28x28 grayscale, so we must:
#   1. Convert grayscale to 3 channels (Grayscale(num_output_channels=3))
#   2. Resize to 224x224
#   3. Apply augmentations (rotation, flip)
#   4. Normalize with ImageNet statistics

emnist_transformation = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),        # 1 channel -> 3 channels
    transforms.Resize((224, 224)),                      # 28x28 -> 224x224
    transforms.RandomRotation(degrees=(90, 90)),        # Data augmentation
    transforms.RandomVerticalFlip(p=1.0),               # Data augmentation
    transforms.ToTensor(),                               # PIL -> Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

train_loader, val_loader = helper_utils.create_emnist_dataloaders(
    emnist_data_path=emnist_data_path,
    batch_size=32,
    transform=emnist_transformation
)


# ==================== STEP 2: EXPLORE PRE-TRAINED MODELS ====================
# Load MobileNetV3 pre-trained on ImageNet (1000 classes)
mobilenet_model = tv_models.mobilenet_v3_small(weights='IMAGENET1K_V1').eval()
class_names = helper_utils.load_imagenet_classes('./outputs/imagenet_class_index.json')

# Load ResNet18 pre-trained on ImageNet
resnet18_model = tv_models.resnet18(weights='IMAGENET1K_V1')
print(resnet18_model)


# ==================== STRATEGY 1: FEATURE EXTRACTION ====================
# KEY CONCEPT: Freeze ALL pre-trained layers. Train ONLY the new classification head.
#
# WHEN TO USE: Small dataset, similar task to original training.
#
# HOW IT WORKS:
#   1. Load pre-trained model
#   2. Freeze all feature extraction layers (requires_grad = False)
#   3. Replace the final layer with a new one for YOUR classes
#   4. Train only the new layer

mobilenet_model = tv_models.mobilenet_v3_small(weights='IMAGENET1K_V1')

# Freeze ALL feature extraction layers
# requires_grad = False means: "don't compute gradients, don't update these weights"
for feature_parameter in mobilenet_model.features.parameters():
    feature_parameter.requires_grad = False

# Replace the output layer for 10 EMNIST digit classes (instead of 1000 ImageNet classes)
last_classifier_layer = mobilenet_model.classifier[-1]
num_features = last_classifier_layer.in_features  # Input size from the layer before
num_classes = 10

new_classifier = nn.Linear(in_features=num_features, out_features=num_classes)
mobilenet_model.classifier[-1] = new_classifier

print("Model's New Output Layer:")
print(mobilenet_model.classifier[-1])

# Only parameters with requires_grad=True will be optimized
# filter() ensures we only pass trainable parameters to the optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, mobilenet_model.parameters()),
    lr=0.001
)

# Train only the new head (1 epoch for demo)
num_epochs = 1
trained_model = helper_utils.training_loop(
    model=mobilenet_model,
    trainloader=train_loader,
    valloader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device
)

emnist_class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
helper_utils.show_predictions(trained_model, val_loader, device, emnist_class_names)


# ==================== STRATEGY 2: FINE-TUNING ====================
# KEY CONCEPT: Unfreeze the TOP (last) few layers of the feature extractor.
# Train them along with the new head, using a VERY LOW learning rate.
#
# WHEN TO USE: Medium dataset, somewhat different task.
#
# WHY LOW LR? The pre-trained weights are already good. We want small
# adjustments, not wholesale changes that destroy the learned features.

fine_tune_model = trained_model  # Start from Strategy 1's result

# Unfreeze ONLY the last feature block
for param in fine_tune_model.features[12].parameters():
    param.requires_grad = True

# Verify what's frozen vs. trainable
print(f"features[0] frozen:      {not fine_tune_model.features[0][0].weight.requires_grad}")   # Still frozen
print(f"features[12] unfrozen:   {fine_tune_model.features[12][0].weight.requires_grad}")      # Now trainable
print(f"classifier unfrozen:     {fine_tune_model.classifier[-1].weight.requires_grad}")        # Still trainable

# Use a MUCH LOWER learning rate for fine-tuning (10-100x smaller)
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, fine_tune_model.parameters()),
    lr=1e-5  # 100x smaller than Strategy 1's LR
)

num_epochs_fine_tune = 1
fine_tune_trained_model = helper_utils.training_loop(
    model=fine_tune_model,
    trainloader=train_loader,
    valloader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    num_epochs=num_epochs_fine_tune,
    device=device
)

helper_utils.show_predictions(fine_tune_trained_model, val_loader, device, emnist_class_names)


# ==================== STRATEGY 3: FULL RETRAINING ====================
# KEY CONCEPT: Unfreeze EVERYTHING and train the whole model.
#
# WHEN TO USE: Large dataset, very different task from original training.
#
# WARNING: Still use a lower LR than training from scratch, because
# the pre-trained weights are a good starting point.

full_retrain_model = fine_tune_trained_model

# Unfreeze ALL parameters
for param in full_retrain_model.parameters():
    param.requires_grad = True

print(f"features[0] unfrozen: {full_retrain_model.features[0][0].weight.requires_grad}")  # Now trainable

# Moderate LR -- higher than fine-tuning, lower than training from scratch
optimizer = torch.optim.SGD(full_retrain_model.parameters(), lr=1e-4)

num_epochs_full_retrain = 1
final_model = helper_utils.training_loop(
    model=full_retrain_model,
    trainloader=train_loader,
    valloader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    num_epochs=num_epochs_full_retrain,
    device=device
)

helper_utils.show_predictions(final_model, val_loader, device, emnist_class_names)

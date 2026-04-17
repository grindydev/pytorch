# From PyTorch to ONNX

# After investing significant effort to train a high-performing model, the next pivotal step is preparing it for deployment. In a real-world setting, your model needs to be efficient and capable of running on various platforms. However, a model is often a set of Python objects that only its native framework, like PyTorch, knows how to execute. This can make it incompatible with environments that cannot run Python or PyTorch directly, such as mobile apps or embedded devices.

# This is where  **[ONNX (Open Neural Network Exchange)](https://onnx.ai)** becomes essential. ONNX is an open standard designed to represent machine learning models, enabling them to be used across different frameworks and runtimes. By converting your model to the ONNX format, you make it **portable** and unlock a wide range of deployment possibilities.

# In this notebook, you will walk through the practical steps of this process. You will:
# * Take a fully trained PyTorch model and export it to the ONNX format.

# * Use the **ONNX Runtime** to perform inference with the newly converted `.onnx` file.

# * As a further demonstration of ONNX's flexibility, an optional section will also guide you through converting the ONNX model to a TensorFlow representation and running inference with it.



import sys
import warnings

# # Redirect stderr to a black hole to catch other potential messages
# class BlackHole:
#     def write(self, message):
#         pass
#     def flush(self):
#         pass
# sys.stderr = BlackHole()

# Ignore Python-level UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import torch
import torchvision.models as tv_models
import helper_utils
from pathlib import Path

# ====================== FIX FOR OPTIONAL TF CONVERSION (MODERN & CLEAN) ======================
# The old "onnx_tf" package is broken on Python 3.12 (requires deprecated tensorflow-addons).
# We now use the modern, actively maintained "onnx2tf" library instead.
# This achieves the exact same goal of the lab (ONNX → native TensorFlow SavedModel)
# but without any more import errors.
# Run this once in your terminal (inside tf_env):
#    pip uninstall onnx-tf -y
#    pip install onnx2tf
# =======================================================================================

from onnx2tf import convert   # ← modern replacement for onnx_tf

# ====================== DEVICE SETUP (optimized for your Mac) ======================
# LEARNING: On Apple Silicon (M1/M2/M3/M4), MPS (Metal) gives huge speed boost.
# The original lab code only checked CUDA — we improved it for real macOS performance.
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("🚀 Using MPS (Apple Silicon GPU) — much faster on your Mac!")
else:
    DEVICE = torch.device('cpu')
print(f"Using Device: {DEVICE}")
# =======================================================================================

# ====================== DATA LOADING AND PREPARATION ======================
# LEARNING: Real-world image datasets are folder-based (one folder = one class).
# helper_utils helps you inspect and load data with proper train/val split + augmentations.
dataset_path = Path.cwd() / "data/fruit_and_vegetable_subset"   # make sure this folder exists next to main.py
helper_utils.dataset_images_per_class(dataset_path)

classes = [
    'Apple (Healthy)', 'Apple (Rotten)', 'Banana (Healthy)', 'Banana (Rotten)',
    'Bellpepper (Healthy)', 'Bellpepper (Rotten)', 'Carrot (Healthy)', 'Carrot (Rotten)',
    'Cucumber (Healthy)', 'Cucumber (Rotten)', 'Grape (Healthy)', 'Grape (Rotten)',
    'Guava (Healthy)', 'Guava (Rotten)', 'Jujube (Healthy)', 'Jujube (Rotten)',
    'Mango (Healthy)', 'Mango (Rotten)', 'Orange (Healthy)', 'Orange (Rotten)',
    'Pomegranate (Healthy)', 'Pomegranate (Rotten)', 'Potato (Healthy)', 'Potato (Rotten)',
    'Strawberry (Healthy)', 'Strawberry (Rotten)', 'Tomato (Healthy)', 'Tomato (Rotten)'
]

train_loader, val_loader = helper_utils.get_dataloaders(dataset_path)
helper_utils.show_image_grid(train_loader, classes)
print("✅ Data loaded and visualized")
# =======================================================================================

# ====================== MODEL PREPARATION (Transfer Learning) ======================
# LEARNING: Transfer learning = take a strong pre-trained backbone (ResNet18) 
# and only retrain the final layer. This is how professionals build models fast.
resnet18_model = tv_models.resnet18(weights=None)

weights_path = Path.cwd() / 'data/pretrained_resnet18_weights/resnet18-f37072fd.pth'
state_dict = torch.load(weights_path, map_location=DEVICE)
resnet18_model.load_state_dict(state_dict)

num_classes = len(classes)
model = helper_utils.adapt_model_for_transfer_learning(resnet18_model, num_classes)

print("✅ ResNet18 loaded + adapted for 28 classes")
# =======================================================================================

# ====================== MODEL TRAINING ======================
# LEARNING: With transfer learning, even 1 epoch gives >75% accuracy.
# In real projects you would train longer and add more techniques.
num_epochs = 1
trained_model = helper_utils.training_loop(model, train_loader, val_loader, num_epochs, DEVICE)
print("✅ Training finished — model is ready!")
# =======================================================================================

# ====================== PyTorch → ONNX EXPORT ======================
# LEARNING: ONNX is the universal format that makes your model runnable
# anywhere (ONNX Runtime, TensorFlow, mobile, edge devices, browsers, etc.).
# This is the core skill of the entire lab.
trained_model.eval()
print("\nExporting PyTorch model to ONNX...")

dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

torch.onnx.export(
    trained_model,
    dummy_input,
    "fruit_veg_model.onnx",
    export_params=True,
    opset_version=11,                    # chosen for maximum compatibility
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("✅ Exported as fruit_veg_model.onnx")
# =======================================================================================

# ====================== INFERENCE WITH ONNX RUNTIME ======================
# LEARNING: ONNX Runtime is super fast and runs everywhere. 
# This shows you are no longer locked into PyTorch.
ort_session = ort.InferenceSession("fruit_veg_model.onnx", providers=['CPUExecutionProvider'])

input_name = ort_session.get_inputs()[0].name

val_iter = iter(val_loader)
images, labels = next(val_iter)
input_data = images[:9].cpu().numpy()
true_labels = labels[:9].numpy()

ort_outputs = ort_session.run(None, {input_name: input_data})
predictions = ort_outputs[0]

print("Displaying predictions from ONNX Runtime...\n")
helper_utils.show_prediction_grid(input_data, true_labels, predictions, classes)
print("✅ ONNX Runtime inference done!")
# =======================================================================================

# ====================== (OPTIONAL) ONNX → NATIVE TENSORFLOW ======================
# LEARNING: onnx2tf is the modern replacement for the old onnx_tf library used in the lab.
# The parameter name changed in newer versions of onnx2tf.
print("\n=== OPTIONAL: ONNX → Native TensorFlow (using modern onnx2tf) ===")
print("Converting ONNX to TensorFlow SavedModel...")

convert(
    input_onnx_file_path="fruit_veg_model.onnx",
    output_folder_path="./fruit_veg_tf_savedmodel",
    output_signaturedefs=True,                    # creates serving_default signature (needed for lab)
    keep_ncw_or_nchw_or_ncdhw_input_names=['input']   # ← corrected parameter name (this was the error)
)

print("✅ TensorFlow SavedModel exported to ./fruit_veg_tf_savedmodel")

# Test the native TF model
val_iter = iter(val_loader)
images, labels = next(val_iter)
input_data = images[:9].cpu().numpy()
true_labels = labels[:9].numpy()

loaded_tf_model = tf.saved_model.load("./fruit_veg_tf_savedmodel")
inference_func = loaded_tf_model.signatures["serving_default"]

input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
predictions_dict = inference_func(input=input_tensor)
tf_predictions_np = predictions_dict['output'].numpy()

print("Displaying predictions from NATIVE TensorFlow model...\n")
helper_utils.show_prediction_grid(input_data, true_labels, tf_predictions_np, classes)
print("✅ ONNX → TensorFlow conversion completed!")
# =======================================================================================

print("\n🎉 LAB COMPLETE!")
print("You have successfully:")
print("   • Used transfer learning with ResNet18")
print("   • Exported PyTorch → ONNX")
print("   • Ran inference with ONNX Runtime")
print("   • Converted ONNX → native TensorFlow (with modern tools)")
print("This workflow is used daily in real production ML deployments!")



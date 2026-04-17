"""
Lesson 3 - Module 4: ONNX Export & Cross-Platform Deployment (ONNX/main.py)
============================================================================
WHAT YOU'LL LEARN:
  * What ONNX (Open Neural Network Exchange) is and why it matters
  * Exporting a PyTorch model to the ONNX format
  * Running inference with ONNX Runtime (framework-independent)
  * Converting ONNX to TensorFlow SavedModel (cross-framework portability)
  * The full deployment pipeline: train in PyTorch -> export -> run anywhere

KEY CONCEPT:
  A trained PyTorch model is a set of Python objects that only PyTorch understands.
  ONNX is a universal format that lets your model run on:
    - Mobile phones (iOS/Android)
    - Web browsers (via ONNX.js)
    - Embedded devices
    - Other ML frameworks (TensorFlow, Caffe2, etc.)
    - Optimized runtimes (ONNX Runtime, TensorRT)

  DEPLOYMENT PIPELINE:
    1. Train model in PyTorch
    2. Export to ONNX format
    3. Run anywhere with ONNX Runtime or convert to other frameworks
"""

import sys
import warnings

# Ignore annoying warnings so the output is clean
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import torch
import torchvision.models as tv_models
import helper_utils
from pathlib import Path

# Modern replacement for the old (broken) onnx_tf package
from onnx2tf import convert


# ==================== STEP 1: DEVICE SETUP (Mac-friendly) ====================
# LEARNING: On Apple Silicon Macs we should use MPS (Metal GPU) for speed.
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("🚀 Using MPS — Apple Silicon GPU acceleration (much faster on your Mac!)")
else:
    DEVICE = torch.device('cpu')
print(f"Using Device: {DEVICE}")


# ==================== STEP 2: DATA LOADING & VISUALIZATION ====================
# LEARNING: Real-world datasets are folder-based (one folder = one class).
# The helper functions handle train/val split + data augmentation automatically.
dataset_path = Path.cwd() / "data/fruit_and_vegetable_subset"
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


# ==================== STEP 3: MODEL PREPARATION (Transfer Learning) ====================
# LEARNING: Transfer learning = reuse a powerful pre-trained backbone (ResNet18)
# and only train the final layer. This is the standard way to build models fast.
resnet18_model = tv_models.resnet18(weights=None)

weights_path = Path.cwd() / 'data/pretrained_resnet18_weights/resnet18-f37072fd.pth'
state_dict = torch.load(weights_path, map_location=DEVICE)
resnet18_model.load_state_dict(state_dict)

num_classes = len(classes)
model = helper_utils.adapt_model_for_transfer_learning(resnet18_model, num_classes)

print("✅ ResNet18 loaded with pre-trained weights + adapted for 28 classes")


# ==================== STEP 4: TRAINING (1 epoch is enough) ====================
# LEARNING: With transfer learning, even 1 epoch gives >75% accuracy.
# In real projects you would train longer + use learning rate scheduling, etc.
num_epochs = 1
trained_model = helper_utils.training_loop(model, train_loader, val_loader, num_epochs, DEVICE)
print("✅ Training finished — model is ready for deployment!")


# ==================== STEP 5: EXPORT PYTORCH → ONNX ====================
# LEARNING: This is the most important step of the lab.
# torch.onnx.export traces the model with a dummy input and saves it as .onnx

model_path = Path.cwd() / 'data/ONNX'
model_path.mkdir(parents=True, exist_ok=True)          # Create folder if it doesn't exist
print(f"✅ Export directory ready: {model_path}")

trained_model.eval()
print("\nExporting PyTorch model to ONNX...")

dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

torch.onnx.export(
    trained_model,                              # model to export
    dummy_input,                                # dummy input for tracing
    model_path / "fruit_veg_model.onnx",        # output file
    export_params=True,                         # save trained weights
    opset_version=13,                           # stable version (avoids old bugs)
    do_constant_folding=True,                   # optimization
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={                              # allow different batch sizes
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print("✅ Exported successfully as fruit_veg_model.onnx")


# ==================== STEP 6: INFERENCE WITH ONNX RUNTIME ====================
# LEARNING: ONNX Runtime runs the model without PyTorch at all.
# This is what makes ONNX so powerful for production deployment.

ort_session = ort.InferenceSession(
    model_path / "fruit_veg_model.onnx",
    providers=['CPUExecutionProvider']
)

input_name = ort_session.get_inputs()[0].name

# Take 9 images from validation set for nice visualization
val_iter = iter(val_loader)
images, labels = next(val_iter)
input_data = images[:9].cpu().numpy()
true_labels = labels[:9].numpy()

ort_outputs = ort_session.run(None, {input_name: input_data})
predictions = ort_outputs[0]

print("Displaying predictions from ONNX Runtime...\n")
helper_utils.show_prediction_grid(input_data, true_labels, predictions, classes)
print("✅ ONNX Runtime inference completed (no PyTorch needed!)")


# ==================== STEP 7: ONNX → NATIVE TENSORFLOW (TFLite) ====================
# LEARNING: onnx2tf converts ONNX to native TensorFlow format.
# IMPORTANT: Modern onnx2tf creates .tflite files (very efficient),
# not the old SavedModel with saved_model.pb. We load the .tflite file.

print("\n=== OPTIONAL: ONNX → Native TensorFlow (TFLite) ===")
print("Converting ONNX to TensorFlow Lite model...")

convert(
    input_onnx_file_path=model_path / "fruit_veg_model.onnx",
    output_folder_path=model_path / "fruit_veg_tf_savedmodel",
    output_signaturedefs=True,
    keep_ncw_or_nchw_or_ncdhw_input_names=['input']
)

print("✅ Conversion completed — TFLite model created")

# Load and run the TFLite model (this is the native TensorFlow way)
tflite_path = model_path / "fruit_veg_tf_savedmodel" / "fruit_veg_model_float32.tflite"

interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Use the same batch of images for fair comparison
val_iter = iter(val_loader)
images, labels = next(val_iter)
input_data = images[:9].cpu().numpy()
true_labels = labels[:9].numpy()

# Run inference with TFLite
tf_predictions = []
for i in range(input_data.shape[0]):
    interpreter.set_tensor(input_details[0]['index'],
                           np.expand_dims(input_data[i], axis=0).astype(np.float32))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    tf_predictions.append(pred[0])

tf_predictions_np = np.array(tf_predictions)

print("Displaying predictions from NATIVE TensorFlow Lite model...\n")
helper_utils.show_prediction_grid(input_data, true_labels, tf_predictions_np, classes)
print("✅ ONNX → TensorFlow Lite conversion completed!")


# ==================== FINAL SUMMARY ====================
print("\n🎉 LAB COMPLETE!")
print("You have successfully:")
print("   • Trained a model using transfer learning (ResNet18)")
print("   • Exported PyTorch model to ONNX format")
print("   • Ran inference with ONNX Runtime (framework-independent)")
print("   • Converted ONNX to native TensorFlow Lite")
print("\nThis exact workflow is used in real production deployments!")
print("Your model is now portable and can run anywhere.")
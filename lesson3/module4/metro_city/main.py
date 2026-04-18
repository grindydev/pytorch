"""
Lesson 3 - Module 4: Metro City -- Model Optimization Pipeline
================================================================
WHAT YOU'LL LEARN:
  * End-to-end optimization: Pruning + Quantization + Layer Fusion + QAT
  * Applying pruning to a real street classifier model
  * Dynamic quantization on a production model
  * Fusing Conv+BN+ReLU layers for faster inference
  * Quantization-Aware Training (QAT) to preserve accuracy after quantization
  * Benchmarking: comparing accuracy, size, and speed before/after optimization

KEY CONCEPT:
  This is the COMPLETE production optimization pipeline:
    1. Train a model normally (FP32)
    2. PRUNE: remove unnecessary weights (sparsity)
    3. FUSE: combine Conv+BN+ReLU into single ops
    4. QUANTIZE: reduce precision from FP32 to INT8
    5. QAT: fine-tune with fake quantization to recover accuracy

  The goal: smaller, faster models that run on edge devices with minimal
  accuracy loss. This is exactly what companies do before deploying models
  to mobile phones, embedded devices, or high-throughput servers.
"""

import copy
import torch
from torch.nn.utils import prune
import torch.nn as nn
import torch.ao.quantization as aoq

import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import helper_utils
import unittests
from pathlib import Path

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("🚀 Using MPS — Apple Silicon GPU acceleration!")
else:
    device = torch.device('cpu')
print(f"Using Device: {device}")


dataset_path = Path.cwd() / "data/data/"

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=train_transform)
dev_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'dev'), transform=eval_transform)
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'test'), transform=eval_transform)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(dev_dataset))
print("Number of test samples:", len(test_dataset))
print("\nClass mapping:", train_dataset.class_to_idx)

# helper_utils.display_some_images(test_dataset)


# Use the base model
model = helper_utils.resnet18_qat_ready_pretrained(num_classes=3, use_quant_stubs=False).to(device)
print(model)

# Load the final model checkpoint
model_path = Path.cwd() / 'data/street_classifier_weights.pt'
model_weights = torch.load(model_path, map_location="cpu")
model.load_state_dict(model_weights)

# Compute accuracy of the loaded model
base_accuracy = helper_utils.compute_accuracy(model, test_loader, device)
print(f"Model accuracy: {base_accuracy:.4f}")

def _iter_prunable_modules(model):
    """
    Iterate over modules that are eligible for pruning.

    Yields
    ------
    Tuple[str, nn.Module]
        Pairs of (fully-qualified module name, module) for layers that are
        prunable in this assignment: `nn.Conv2d` and `nn.Linear`.

    Notes
    -----
    - The qualified name comes from `model.named_modules()` and reflects the
      path within the module hierarchy (e.g., "block.0", "classifier.fc").
    - Use this generator to systematically apply pruning across the model.
    """
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m

def finalize_pruning(model):
    """
    Make pruning permanent by removing reparametrization wrappers.

    This converts any pruned parameter from the (`weight_orig`, `weight_mask`)
    reparametrization back to a regular `weight` `nn.Parameter` where the
    zeros are **materialized** in the stored tensor.
    """
    for _, module in _iter_prunable_modules(model):
        # Only remove if the parameter has been pruned
        if hasattr(module, "weight_orig") and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
    return model

# GRADED FUNCTION: prune_model

def prune_model(model, amount=0.3, mode="l1_unstructured"):
    """
    Apply pruning to **weights** of all `Conv2d` and `Linear` layers.

    This uses PyTorch's pruning reparametrization (adds `weight_orig` and
    `weight_mask`) without changing the tensor shape. To permanently embed
    zeros into the stored weights, call `finalize_pruning(model)` afterward.

    Parameters
    ----------
    model : nn.Module
        Model to prune. Pruning is applied **in-place** via
        `torch.nn.utils.prune`.
    amount : float, optional (default=0.3)
        Fraction in [0, 1] to prune.
        - For **unstructured** pruning: fraction of smallest-magnitude weights
          within each tensor.
        - For **structured (ln)** pruning: fraction of **output channels**
          (dimension 0) to remove using L2-norm (n=2).
    mode : {"l1_unstructured", "ln_structured"}, optional
        Pruning strategy:
        - `"l1_unstructured"` → `prune.l1_unstructured(..., name="weight", amount=amount)`
        - `"ln_structured"`   → `prune.ln_structured(..., name="weight", amount=amount, n=2, dim=0)`

    Returns
    -------
    nn.Module
        The same model instance with pruning **reparametrization** applied
        (not yet made permanent).
    """

    ### START CODE HERE ###

    if not (0 <= amount <= 1): # Check if amount is in [0,1]
        raise ValueError(f"amount must be in [0,1], got {amount}") 

    for _, module in _iter_prunable_modules(model):
        if not hasattr(module, 'weight'): # Check if module has "weight" attribute
            continue 

        if mode == "l1_unstructured": # Check if mode is "l1_unstructured"
            prune.l1_unstructured(module, 'weight', amount) # l1_unstructured from prune with module, name("weight"), and amount
        elif mode == "ln_structured": # Check elif mode is "ln_structured"
            prune.ln_structured(module, 'weight', amount, 2, 0) # ln_structured from prune with module, name("weight"), amount, n(2), and dim(0)
        else: 
            raise ValueError("mode must be 'l1_unstructured' or 'ln_structured'")
    ### END CODE HERE ###
    
    return model
    
# Verify your code here

# Create baseline model statistics
base = helper_utils.sparsity_report(model)
print("[BASE] global_sparsity:", base["global_sparsity"])
base_time = helper_utils.bench(model, device=device)
print("[BASE] time:", base_time)

# We prune 50% of the model
prune_model(model, amount=0.5, mode="l1_unstructured")

after = helper_utils.sparsity_report(model)
print("[AFTER PRUNE] global_sparsity:", after["global_sparsity"])

after_acc = helper_utils.compute_accuracy(model, test_loader, device=device)
print("[AFTER PRUNE] accuracy:", after_acc)

pruned_time = helper_utils.bench(model, device=device)

print(f"\nInference time comparison:")
print(f"Base model: {base_time:.4f} seconds per batch")
print(f"Pruned model: {pruned_time:.4f} seconds per batch") 
print(f"Speedup: {base_time / pruned_time:.2f}x")

# Test 1: Prune Model

unittests.exercise1(prune_model)


# GRADED FUNCTION: quantize_dynamic_linear

def quantize_dynamic_linear(model):
    """
    Return a **new** model where all nn.Linear layers are dynamically quantized to INT8.

    Requirements checked by the autograder
    --------------------------------------
    - Do NOT mutate the original model; use a deepcopy.
    - Quantize ONLY Linear modules (e.g., {nn.Linear}).
    - Use dynamic quantization with INT8 dtype.
    - Return the quantized model in eval() mode.
    - Should run on CPU-only environments (no CUDA, no calibration).

    Returns
    -------
    nn.Module
        An eval-mode copy of `model` with Linear layers using INT8 dynamic quantization.
    """
    ### START CODE HERE ###
    model_fp32 = copy.deepcopy(model).eval() # Create a deep copy of the model and set it to eval mode
    model_fp32.to('cpu')

    torch.backends.quantized.engine = 'qnnpack'
    print("   → Quantized engine set to 'qnnpack' (Mac compatible)")

    # # Ensure a sensible engine on CPU (x86). If unavailable, this line is harmless.
    # has_quantized = torch.backends.quantized # Check if torch.backends has quantized
    # has_engine = torch.backends.quantized.engine # Check if torch.backends.quantized has engine
    # if has_quantized and has_engine: # @KEEP
    #     try:
    #         torch.backends.quantized.engine = "fbgemm" 
    #     except Exception: 
    #         pass  # keep whatever the runtime supports 
    # Quantize only Linear layers to INT8
    quantized = torch.quantization.quantize_dynamic( # Use quantize_dynamic from quantization in torch to quantize the model_fp32 to INT8
        model_fp32, # The model to quantize
        {nn.Linear}, # The layers to quantize (only Linear layers)
        dtype=torch.qint8 # The dtype to quantize to qint8
    )
    
    quantized.eval() # Set the quantized model to eval mode

    return quantized # Return the quantized model

    ### END CODE HERE ###


# Verify your code here

# Use the base model to start fresh
model = helper_utils.resnet18_qat_ready_pretrained(num_classes=3, use_quant_stubs=True).to(device)
model_weights = torch.load(model_path, map_location="cpu")
model.load_state_dict(model_weights)
model.to("cpu")
model.eval()

# Set seed for reproducibility
torch.manual_seed(5)

# Quantize the model
qmodel = quantize_dynamic_linear(model)

# Evaluate the quantized model
qacc = helper_utils.compute_accuracy(qmodel, test_loader, device="cpu")
print(f"\nAccuracy on test dataset after quantization: {100*qacc:.2f}%")

# Benchmark the models  
t_fp32 = helper_utils.bench(model, device="cpu", shape=(32, 3, 224, 224))
t_int8 = helper_utils.bench(qmodel, device="cpu", shape=(32, 3, 224, 224))
print("\n[TIMING] avg forward per batch (CPU)")
print(f"  - FP32 : {t_fp32*1e3:.2f} ms")
print(f"  - INT8 : {t_int8*1e3:.2f} ms (↓ is better)")
print(f"  - Improvement: {((t_fp32 - t_int8)/t_fp32)*100:.1f}%")

# Verify your code here

unittests.exercise2(quantize_dynamic_linear)



# GRADED FUNCTION: fuse_model_inplace

def fuse_model_inplace(model: nn.Module) -> nn.Module:
    """
    Recursively apply best-effort eager fusion to:
      Conv+BN+ReLU, Conv+BN, Conv+ReLU, Linear+ReLU
    Only fuses *adjacent* modules inside nn.Sequential blocks.
    Modifies `model` in-place and returns the *same instance*.
    """
    ### START CODE HERE ###
    for _, child in model.named_children(): # Iterate over the named children of the model
        # Recurse first
        fuse_model_inplace(child) # Recursively apply best-effort eager fusion to the child

        # Then scan this child if it's a Sequential
        if (isinstance(child, nn.Sequential) and len(child) >= 2): # Check if the child is a Sequential and has at least 2 layers
            # BN folding prefers eval; don't mutate outer state permanently
            was_training = child.training # Get the training state of the child
            child.eval() # Set the child to eval mode
            i = 0 
            while i < len(child) - 1: # Iterate over the child layers - 1
                a, b = child[i], child[i + 1] # Get the two adjacent layers at i and i + 1
                c = child[i + 2] if i + 2 < len(child) else None

                # Conv + BN + ReLU
                # Check if the first layer is a Conv2d, the second layer is a BatchNorm2d, and the third layer is a ReLU
                if (isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d) and isinstance(c, nn.ReLU)):
                    torch.quantization.fuse_modules(child, [str(i), str(i+1), str(i+2)], inplace=True) # Try to fuse the three layers
                    i += 3 
                    continue 
                # Conv + BN
                if (isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d)): # Check if the first layer is a Conv2d and the second layer is a BatchNorm2d
                    torch.quantization.fuse_modules(child, [str(i), str(i+1)], inplace=True) # Try to fuse the two layers
                    i += 2 
                    continue 
                # Conv + ReLU
                if (isinstance(a, nn.Conv2d) and isinstance(b, nn.ReLU)): # Check if the first layer is a Conv2d and the second layer is a ReLU
                    torch.quantization.fuse_modules(child, [str(i), str(i+1)], inplace=True) # Try to fuse the two layers
                    i += 2 
                    continue 
                # Linear + ReLU
                if (isinstance(a, nn.Linear) and isinstance(b, nn.ReLU)): # Check if the first layer is a Linear and the second layer is a ReLU
                    torch.quantization.fuse_modules(child, [str(i), str(i+1)], inplace=True) # Try to fuse the two layers
                    i += 2 
                    continue 

                i += 1 

            if was_training: # Check if the child was training
                child.train() # Set the child to train mode

    # IMPORTANT: return the same object (tests check identity)
    return model # Return the model

    ### END CODE HERE ###


# Verify your code

# Create a toy model to test your code
torch.manual_seed(0)
device = torch.device("cpu")
toy = helper_utils.ToyNet().eval().to(device)

# Keep a copy for numerical comparison
toy_copy = helper_utils.ToyNet().eval().to(device)
toy_copy.load_state_dict(toy.state_dict())

# Show BEFORE
helper_utils.list_children(toy, "Before fusion")

# Forward pass BEFORE
x = torch.randn(2, 3, 32, 32, device=device)
with torch.no_grad():
    y_before = toy(x)

# Apply your fusion function (assumes fuse_model_inplace is defined + _try_fuse available)
ret_model = fuse_model_inplace(toy).eval()
# Show AFTER
helper_utils.list_children(toy, "After fusion")

# Forward pass AFTER
with torch.no_grad():
    y_after = toy(x)

# Report numerical closeness and fused-layer counts
max_abs_diff = (y_before - y_after).abs().max().item()
fused_counts = helper_utils.count_fused_layers(toy)


print("\n== Verification ==")
print(f"Max |y_before - y_after|: {max_abs_diff:.6g}  (expect ~0)")
print("Fused intrinsic layers found:", fused_counts if fused_counts else "{} (none)")

# sanity check on output shape
print("Output shapes -> before:", tuple(y_before.shape), ", after:", tuple(y_after.shape))


# Test your code!

unittests.exercise3(fuse_model_inplace)



# Wrapper for QAT
class QATWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.quant = aoq.QuantStub()
        self.m = m
        self.dequant = aoq.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.m(x)
        x = self.dequant(x)
        return x


# GRADED FUNCTION: prepare_qat

def prepare_qat(model, backend="fbgemm"):
    """
    Return a **QAT-ready copy** of `model`:
      - Sets quantized backend (default: 'fbgemm')
      - Applies best-effort fusion (Conv+BN(+Act))
      - Attaches a default QAT qconfig
      - Runs eager-mode prepare_qat to insert observers/fake-quant
      - Returns the prepared module in **train()** mode

    The original `model` **must not** be mutated.

    Parameters
    ----------
    model : nn.Module
        FP32 model to prepare for QAT.
    backend : str
        Quantized engine (use 'fbgemm' on x86; 'qnnpack' on ARM).

    Returns
    -------
    nn.Module
        A new, QAT-ready model (with observers) in training mode.
    """
    ### START CODE HERE ###
    # 1) Work on a copy; do not mutate the original
    qat = copy.deepcopy(model) # Create a deep copy of the model and set it to train mode
    qat.eval()

    # 2) Fuse eligible modules (best-effort; safe no-op if unsupported)
    fuse_model_inplace(qat) # Fuse the eligible modules (qat)

    # 3) Attach default QAT qconfig
    qat.qconfig = torch.quantization.get_default_qat_qconfig(backend)

    # 4) Prepare for QAT (insert observers/fake-quant)
    qat.train()
    torch.quantization.prepare_qat( # Prepare the model for QAT
        model=qat, # The model to prepare for QAT
        inplace=True, # Set the correct value for inplace
        ) 

    ### END CODE HERE ###
    return qat


# Verify your code here

# Use the base model to start fresh

model = helper_utils.resnet18_qat_ready_pretrained(num_classes=3, use_quant_stubs=False).to(device)
model_weights = torch.load(model_path, map_location="cpu")
model.load_state_dict(model_weights)
# wrap the base model in QATWrapper to add stubs for quantization
wrapped_model = QATWrapper(model)

print("Base Model loaded and wrapped")

# Prepare the QAT model
qat_model = prepare_qat(wrapped_model, backend="qnnpack")
print("Model prepared for qat")

# Fine-tune with fake-quant in the loop (can be on GPU)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD((p for p in qat_model.parameters() if p.requires_grad),
                            lr=1e-4, momentum=0.9, weight_decay=1e-4)

qat_model.to(device)

helper_utils.train_model(
    qat_model,
    train_loader,
    dev_loader,
    1,
    optimizer,
    device,
    save_path=  "fine_tuned_qat_model.pt")
    
qat_model.to("cpu")

# Convert to real INT8 (runs on CPU)
qat_model.eval()
int8_model = torch.quantization.convert(qat_model)
print("Model converted to int8")

# Save the quantized model with full state
torch.save({
    'model_state_dict': int8_model.state_dict(),
    'quantization_config': int8_model.state_dict()
}, "quantized_int8_model.pt")

print("Saved quantized model checkpoint to quantized_int8_model.pt")

# Evaluate int8 model on test data
int8_model.eval()
print("Testing model on cpu")
test_acc = helper_utils.compute_accuracy(int8_model, test_loader, device="cpu")
print(f"Test accuracy in base model: {base_accuracy:.2f}%")
print(f'\nInt8 model test accuracy: {test_acc:.2f}%')

# Measure inference time for both models on cpu
model.to("cpu")
int8_model.to("cpu")
base_time = helper_utils.bench(model, device="cpu", shape=(32, 3, 224, 224))
int8_time = helper_utils.bench(int8_model, device="cpu", shape=(32, 3, 224, 224))

# Calculate percentage improvement
time_improvement = ((base_time - int8_time) / base_time) * 100

print(f"\nInference time comparison:")
print(f"Base model: {base_time:.4f} seconds per batch")
print(f"Int8 model: {int8_time:.4f} seconds per batch") 
print(f"Speed improvement: {time_improvement:.1f}%")

# Save both models weights to compare sizes
torch.save(model.state_dict(), "base_model_weights.pt")
torch.save(int8_model.state_dict(), "int8_model_weights.pt")

# Get file sizes in MB
base_size = os.path.getsize("base_model_weights.pt") / (1024 * 1024)
int8_size = os.path.getsize("int8_model_weights.pt") / (1024 * 1024)

print(f"\nModel size comparison:")
print(f"Base model: {base_size:.2f} MB")
print(f"Int8 model: {int8_size:.2f} MB")
print(f"Size reduction: {((base_size - int8_size) / base_size * 100):.1f}%")

# Test your code!

unittests.exercise4(prepare_qat)


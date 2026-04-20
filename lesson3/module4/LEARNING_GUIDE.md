# Learning Guide: Lesson 3, Module 4 -- MLOps: Tracking, Deployment, and Optimization

## Module Overview

Training a model is only half the job. This module teaches you how to track
experiments (MLflow), export models for production (ONNX), compress models
(pruning, quantization), and run the full optimization pipeline end-to-end.

## Recommended Reading Order

1. **MLflow/main.py** -- Experiment tracking with MLflow + Lightning
2. **ONNX/main.py** -- Export PyTorch model to ONNX, run cross-platform
3. **pruning/main.py** -- Removing unnecessary weights (unstructured, structured, global)
4. **quantization/main.py** -- Reducing precision: dynamic, static, QAT
5. **metro_city/main.py** -- Full pipeline: prune + fuse + quantize + QAT

## Concept Map

```
Trained Model (FP32, full size)
   |
   v
Experiment Tracking (MLflow)
   |
   +--> Log parameters, metrics, artifacts
   +--> Compare runs in web UI
   +--> Register best model
   |
   v
Export (ONNX)
   |
   +--> PyTorch -> ONNX format
   +--> Run on ONNX Runtime (no PyTorch needed)
   +--> Convert to TensorFlow (cross-framework)
   |
   v
Optimization (make smaller and faster)
   |
   +--> Pruning: set weights to zero (sparsity)
   |    +--> L1 Unstructured: zero smallest weights
   |    +--> Ln Structured: zero entire channels
   |    +--> Global: consistent sparsity across layers
   |
   +--> Layer Fusion: Conv+BN+ReLU -> single operation
   |
   +--> Quantization: FP32 -> INT8 (4x smaller)
   |    +--> Dynamic: convert at runtime (simplest)
   |    +--> Static: calibrate with real data (better)
   |    +--> QAT: train with fake quantization (best accuracy)
   |
   v
Deployed Model (small, fast, runs anywhere)
```

## File Summaries

### MLflow/main.py
Uses PyTorch Lightning + MLflow to track a CIFAR-10 training run.
Logs hyperparameters, metrics, confusion matrix, and model artifacts.
Includes a custom Lightning Callback for MLflow integration.
Focus on: the MLflow UI for comparing experiments visually.

### ONNX/main.py
Exports a fruit/vegetable classifier from PyTorch to ONNX format.
Runs inference with ONNX Runtime (no PyTorch dependency) and optionally
converts to TensorFlow SavedModel.
Focus on: torch.onnx.export() parameters and the dynamic_axes argument.

### pruning/main.py
Teaches three pruning methods on a simple toy model:
- L1 Unstructured: zeros individual weights with smallest magnitude
- Ln Structured: zeros entire rows/channels
- Global: distributes sparsity uniformly across all layers
Shows the pruning workflow: apply -> remove (make permanent) -> fine-tune.
Focus on: the weight_orig/weight_mask reparametrization pattern.

### quantization/main.py
Compares three quantization methods on a CIFAR-10 CNN:
- Dynamic: simplest, converts Linear layers at runtime
- Static: calibrates with data, converts all layers
- QAT: trains with simulated quantization for best accuracy
Includes a final comparison table of all methods.
Focus on: QuantStub/DeQuantStub for marking quantization boundaries.

### metro_city/main.py
Full production optimization pipeline on a street classifier:
pruning -> layer fusion -> dynamic quantization -> QAT.
Benchmarking: accuracy, model size, inference speed before and after.
Focus on: the complete pipeline from trained model to optimized deployment.

### pruning/optional.py
Optional: applies pruning to the fruit/vegetable model with before/after
comparison of sparsity and accuracy.

### quantization/optional.py
Optional: advanced quantization techniques and edge cases.

## Common Questions

**Q: Why do I need ONNX if I already have a PyTorch model?**
A: PyTorch requires Python and the PyTorch library to run. ONNX models can run
on any platform with ONNX Runtime (C++, C#, Java, JavaScript, mobile). If you
want to deploy on a phone, browser, or embedded device, ONNX is the bridge.

**Q: Does pruning actually make the model faster?**
A: Unstructured pruning (zeroing individual weights) does NOT speed up inference
on standard hardware because the zeros still take up memory. Structured pruning
(removing entire channels) CAN speed things up. Real speedup from unstructured
pruning requires specialized sparse hardware or compilers.

**Q: What is the difference between static and dynamic quantization?**
A: Dynamic: weights are INT8 but activations are computed in FP32 at runtime.
Good for models with variable activation ranges (LSTMs, Transformers).
Static: both weights AND activations are INT8, calibrated on real data.
Best for CNNs where activation ranges are predictable.

**Q: When should I use QAT vs post-training quantization?**
A: Post-training (dynamic/static) is faster to apply and good enough when
accuracy drop is acceptable (< 1%). Use QAT when you need maximum accuracy
after quantization, especially for models sensitive to precision loss.

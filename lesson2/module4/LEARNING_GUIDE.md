# Learning Guide: Lesson 2, Module 4 -- PyTorch Lightning and Data Pipeline Optimization

## Module Overview

The training loop you have been writing manually (zero_grad, forward, loss,
backward, step) is boilerplate that repeats in every project. PyTorch Lightning
removes this boilerplate and gives you a clean, organized structure. This module
also teaches you how to optimize your data pipeline for faster training.

## Recommended Reading Order

1. **lightning/main.py** -- Lightning basics: DataModule, LightningModule, Trainer
2. **lightning/exercise.py** -- Practice exercise with Lightning
3. **optimizing_dataloader/main.py** -- Speeding up data loading
4. **advance_lightning/main.py** -- Advanced features: callbacks, early stopping
5. **diagnostic_assistant/main.py** -- Full end-to-end Lightning project

## Concept Map

```
PyTorch Lightning Structure
   |
   +--> LightningDataModule: all data logic in one class
   |    +--> prepare_data(): download
   |    +--> setup(): create datasets
   |    +--> train/val_dataloader(): return loaders
   |
   +--> LightningModule: model + training logic
   |    +--> forward(): model architecture
   |    +--> training_step(): one batch during training
   |    +--> validation_step(): one batch during validation
   |    +--> configure_optimizers(): optimizer setup
   |
   +--> Trainer: orchestrates everything
   |    +--> handles epoch loops, device management, logging
   |    +--> supports callbacks (early stopping, checkpoints)
   |
   v
Optimization
   +--> num_workers: parallel data loading
   +--> pin_memory: faster GPU transfer
   +--> Profiling: find bottlenecks
```

## File Summaries

### lightning/main.py
Introduction to PyTorch Lightning. Shows CIFAR10DataModule and
CIFAR10LightningModule. Includes profiling to measure performance.
Focus on: how Lightning replaces the manual training loop with three methods
(training_step, validation_step, configure_optimizers).

### exercise.py
Practice exercise building a Lightning module.
Focus on: implementing the required methods yourself.

### optimizing_dataloader/main.py
Shows how DataLoader settings affect training speed: num_workers, pin_memory,
batch_size. Includes benchmarking.
Focus on: the trade-off between data loading speed and memory usage.

### advance_lightning/main.py
Advanced Lightning features: custom callbacks, early stopping, learning rate
monitoring, model checkpointing.
Focus on: the Callback class and how to hook into the training lifecycle.

### diagnostic_assistant/main.py
Full end-to-end project using Lightning: data module, model, callbacks,
training, and evaluation. A practical medical diagnosis classification task.
Focus on: how all the Lightning pieces fit together in a real project.

## Common Questions

**Q: Should I always use Lightning instead of raw PyTorch?**
A: For learning, raw PyTorch is better because you see exactly what happens.
For projects, Lightning saves time and reduces bugs. The concepts transfer
directly -- Lightning just automates what you would write manually.

**Q: What is profiling?**
A: Measuring how much time each operation takes. Like a stopwatch for your
model's forward pass, backward pass, data loading, etc. It helps you find
which part is slow so you can optimize it.

**Q: How many workers should I use for DataLoader?**
A: Start with 2-4 workers. More workers = faster loading but more memory.
If your GPU is waiting for data (low utilization), increase workers. If you
run out of memory, decrease them.

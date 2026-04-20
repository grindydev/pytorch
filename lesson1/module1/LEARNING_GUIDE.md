# Learning Guide: Lesson 1, Module 1 -- Tensors and Linear Regression

## Module Overview

This is where your deep learning journey begins. You learn the fundamental data
structure (tensors) and the fundamental algorithm (training a model with gradient
descent). Everything in later modules builds on these two ideas.

## Recommended Reading Order

1. **tensor.py** -- Learn what tensors are and how to manipulate them
2. **tensor_excercise.py** -- Practice tensor operations with exercises
3. **leaner.py** -- Your first neural network: a simple linear model
4. **non_leaner.py** -- Adding hidden layers and non-linearity for curved data

## Concept Map

```
Tensors (data containers)
   |
   +--> Create from Python/NumPy/Pandas
   +--> Shape manipulation (reshape, unsqueeze, squeeze, transpose)
   +--> Indexing and slicing
   +--> Math operations and broadcasting
   |
   v
Linear Model (leaner.py)
   |
   +--> nn.Linear: y = weight * x + bias
   +--> Loss function: measures how wrong predictions are
   +--> Optimizer: adjusts weights to reduce loss
   +--> Training loop: zero_grad -> forward -> loss -> backward -> step
   |
   v
Non-Linear Model (non_leaner.py)
   |
   +--> Hidden layers: more capacity to learn complex patterns
   +--> ReLU activation: introduces non-linearity (bends the line)
   +--> Normalization: standardize input/output for stable training
   +--> De-normalization: convert predictions back to original scale
```

## File Summaries

### tensor.py
The alphabet of PyTorch. Covers tensor creation, shape manipulation, indexing,
math, and broadcasting. If tensors confuse you, everything else will too.
Focus on: shapes (the #1 bug source in deep learning) and broadcasting.

### tensor_excercise.py
Practice exercises for tensor operations. Uses a sales dataset.
Do this after tensor.py to check your understanding.
Focus on: indexing, slicing, and boolean masking exercises.

### leaner.py
Your first training loop. Predicts delivery time from distance using a single
linear layer. This is the SIMPLEST possible neural network.
Focus on: the 5-step training loop pattern -- this pattern repeats everywhere.

### non_leaner.py
Shows why we need hidden layers and non-linear activations. A straight line
cannot fit curved data. Adding ReLU lets the model learn curves.
Focus on: why normalization matters, and the intuition behind ReLU.

## Common Questions

**Q: Why tensors instead of NumPy arrays?**
A: Tensors can run on GPU (100x faster) and support automatic differentiation
(computing gradients automatically). NumPy does neither.

**Q: What is the training loop and why does it repeat?**
A: The loop tries different weights, measures how wrong they are (loss),
computes which direction to adjust (gradient), then nudges the weights.
Each iteration makes the model slightly better.

**Q: Why do we need non-linearity (ReLU)?**
A: Without it, stacking linear layers is just one big linear layer.
ReLU introduces a "bend" that lets the model learn curved relationships.

**Q: What is normalization and when should I use it?**
A: Normalization rescales data to mean~0, std~1. Use it when your input
features have very different scales (e.g., distance in miles vs time in minutes).
It helps the optimizer converge faster.

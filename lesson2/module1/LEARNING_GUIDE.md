# Learning Guide: Lesson 2, Module 1 -- Learning Rates, Schedulers, and Hyperparameter Tuning

## Module Overview

A good model architecture means nothing if you train it poorly. This module
teaches you how to tune the training process itself: finding the right learning
rate, using schedulers to adjust it over time, and using Optuna to automatically
search for the best hyperparameters.

## Recommended Reading Order

1. **learning_rate/main.py** -- How LR affects training, evaluation metrics
2. **scheduler/main.py** -- Three schedulers: StepLR, CosineAnnealing, ReduceLROnPlateau
3. **efficiency_performance/main.py** -- Comparing model architectures for speed vs accuracy
4. **optuna/main.py** -- Automatic hyperparameter search with Optuna

## Concept Map

```
Learning Rate (the most important hyperparameter)
   |
   +--> Too low: slow convergence
   +--> Too high: unstable, may diverge
   +--> Just right: fast, stable convergence
   |
   v
LR Schedulers (adjust LR during training)
   |
   +--> StepLR: reduce by factor every N epochs
   +--> CosineAnnealing: smooth cosine curve
   +--> ReduceLROnPlateau: reduce when progress stalls
   |
   v
Evaluation Metrics (beyond accuracy)
   |
   +--> Precision: of predicted positives, how many are correct?
   +--> Recall: of actual positives, how many did we find?
   +--> F1 Score: harmonic mean of precision and recall
   |
   v
Hyperparameter Tuning (automated search)
   |
   +--> Optuna: tries different combinations, finds the best
   +--> FlexibleCNN: architecture defined by hyperparameters
```

## File Summaries

### learning_rate/main.py
Tests different learning rates and measures accuracy, precision, recall, F1.
Shows why accuracy alone is misleading on imbalanced datasets.
Focus on: the torchmetrics library and the Precision/Recall/F1 formulas.

### scheduler/main.py
Compares three LR schedulers on the same model. Each scheduler adjusts the
learning rate differently over epochs.
Focus on: when to use each scheduler and the scheduler.step() call timing.

### efficiency_performance/main.py
Compares different model architectures (shallow vs deep, narrow vs wide) on
training speed and accuracy. Introduces the trade-off between model complexity
and performance.
Focus on: how to measure and compare training efficiency.

### optuna/main.py
Uses Optuna to automatically search for the best CNN architecture (number of
layers, filters, kernel sizes, dropout rate). Each trial trains a model with
different hyperparameters and returns the validation accuracy.
Focus on: the objective function pattern and visualization of results.

## Common Questions

**Q: What learning rate should I start with?**
A: For Adam optimizer, 0.001 (1e-3) is a good default. For SGD, 0.01 (1e-2).
Always do a quick sweep (try 1e-4, 1e-3, 1e-2) to find the right ballpark.

**Q: Which scheduler should I use?**
A: ReduceLROnPlateau is the safest default -- it reacts to actual training
progress. CosineAnnealing works well when you know the total number of epochs.
StepLR is simple but requires tuning the step_size parameter.

**Q: When is accuracy not enough?**
A: When your dataset is imbalanced (e.g., 99% cats, 1% dogs). A model that
always predicts "cat" has 99% accuracy but is useless. Precision, Recall, and
F1 give a complete picture of performance across all classes.

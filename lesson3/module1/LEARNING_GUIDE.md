# Learning Guide: Lesson 3, Module 1 -- Advanced Architectures: ResNet, DenseNet, Siamese Networks

## Module Overview

Simple CNNs hit a limit around 10-20 layers. ResNet and DenseNet introduced
architectural innovations that allow training 100+ layer networks. Siamese
networks solve a different problem: learning similarity instead of classification.
This module teaches you these advanced architectures.

## Recommended Reading Order

1. **resnet/main.py** -- Skip connections, residual blocks, very deep networks
2. **densenet/main.py** -- Dense connections, feature reuse, growth rate
3. **siamese_network/main.py** -- Similarity learning, contrastive/triplet loss
4. **classification/main.py** -- Comparing architectures on the same task

## Concept Map

```
Problem: Deep networks degrade (vanishing gradients)
   |
   v
ResNet Solution: Skip Connections
   |
   +--> output = F(x) + x  (addition)
   +--> Gradient flows directly through the shortcut
   +--> Enables 50, 101, 152 layer networks
   |
   v
DenseNet Solution: Dense Connections
   |
   +--> output = [x, F1(x), F2([x, F1(x)]), ...]  (concatenation)
   +--> Each layer receives ALL previous feature maps
   +--> Growth rate: new features per layer
   +--> Transition layers: reduce dimensions between blocks
   |
   v
Siamese Network: Different Problem
   |
   +--> Two identical branches sharing weights
   +--> Learn EMBEDDING (not classification)
   +--> Triplet loss: anchor + positive - negative
   +--> Use cases: signature verification, face recognition
```

## File Summaries

### resnet/main.py
Implements a ResNet from scratch with BasicBlock (two conv layers + skip
connection). Trains on Aerial Landscapes dataset.
Focus on: how the skip connection works in forward() -- adding input to output.

### densenet/main.py
Implements a DenseNet from scratch with DenseLayer and DenseBlock. Trains on
UC Merced Land Use dataset.
Focus on: how concatenation differs from ResNet's addition, and transition
layers that compress features.

### siamese_network/main.py
Builds a Siamese network for signature verification (detecting forgeries).
Uses triplet dataset (anchor, positive, negative pairs) and contrastive loss.
Focus on: the dataset structure for pair-based learning and how the two
branches share weights.

### classification/main.py
Compares different CNN architectures on a clothing classification task.
Uses torchinfo for architecture inspection.
Focus on: the trade-offs between model size, speed, and accuracy.

## Common Questions

**Q: What is the key difference between ResNet and DenseNet?**
A: ResNet ADDS the skip connection: output = F(x) + x.
DenseNet CONCATENATES all previous outputs: output = [x, F(x)].
Addition keeps the same size; concatenation grows the size (controlled by
growth rate).

**Q: Why use a Siamese network instead of regular classification?**
A: Classification requires a fixed set of classes. If you add a new person to
a face recognition system, you need to retrain. Siamese networks learn a
distance function: "are these two faces the same person?" New people can be
added without retraining.

**Q: What is a triplet in Siamese training?**
A: Three images: anchor (reference), positive (same person as anchor),
negative (different person). The model learns to make anchor+positive close
and anchor+negative far apart in embedding space.

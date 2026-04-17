# PyTorch Machine Learning Course Notes

## Overview

This project contains my hands-on notes from a PyTorch deep learning course, covering everything from basic tensors to deploying models in production. Each lesson builds on the previous one, progressing from fundamentals to advanced architectures.

---

## Course Structure

```
pytorch/
├── lesson1/          # PyTorch Fundamentals
│   ├── module1/      # Tensors & Linear Regression
│   ├── module2/      # Datasets, Transforms & Image Classification (MNIST/EMNIST)
│   ├── module3/      # Custom Datasets, Data Pipelines & Robust Loading
│   └── module4/      # CNN Basics, Debugging & Image Classification
│
├── lesson2/          # Training, Optimization & NLP
│   ├── module1/      # Learning Rates, Schedulers, Optuna Tuning
│   ├── module2/      # Transfer Learning & Data Preprocessing
│   ├── module3/      # NLP: Embeddings, Tokenizers, Text Classification, BERT
│   └── module4/      # PyTorch Lightning & Data Pipeline Optimization
│
├── lesson3/          # Advanced Architectures & Deployment
│   ├── module1/      # ResNet, DenseNet, Siamese Networks
│   ├── module2/      # Model Interpretability, Saliency Maps, Stable Diffusion
│   ├── module3/      # Transformers: Self-Attention, Encoder, Decoder, Translation
│   └── module4/      # MLOps: MLflow Experiment Tracking & ONNX Deployment
│
├── data/             # Datasets (auto-downloaded)
├── saved_models/     # Trained model checkpoints (.pth files)
├── best_model/       # Best-performing model checkpoints
└── intermediate_steps/  # Visualization steps (progressive training images)
```

---

## Quick Reference: What Each File Teaches

### Lesson 1 -- PyTorch Fundamentals

| Module | File | Topic | Key Concepts |
|--------|------|-------|-------------|
| 1.1 | `tensor.py` | Tensor basics | Creating tensors, shapes, indexing, slicing, math ops, broadcasting |
| 1.1 | `leaner.py` | Linear regression | nn.Sequential, MSELoss, SGD, training loop (5 steps), prediction |
| 1.1 | `non_leaner.py` | Non-linear regression | Hidden layers, ReLU activation, data normalization, de-normalization |
| 1.2 | `digit_detective.py` | MNIST classification | torchvision datasets, transforms, DataLoader, DNN, CrossEntropyLoss, Adam |
| 1.2 | `transform_dataset.py` | Computing dataset stats | Two-pass mean/std calculation, normalization from scratch |
| 1.2 | `letter_detective.py` | EMNIST letter classification | 26-class problem, label indexing, deeper network, decoding messages |
| 1.3 | `data_management/main.py` | Custom datasets | FlowerDataset, transforms pipeline, train/val/test split, augmentation |
| 1.3 | `data_pileline/main.py` | CSV-based dataset | PlantsDataset from CSV, dynamic mean/std, SubsetWithTransform |
| 1.4 | `cnn/cnn_block.py` | CNN building blocks | Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout, SimpleCNN |
| 1.4 | `cnn/main.py` | CNN training on CIFAR-100 | Separate train/val transforms, best model checkpointing, weight_decay |
| 1.4 | `debugging/main.py` | Debugging models | Shape mismatch debugging, activation statistics, parameter counting |
| 1.4 | `nature_classification/main.py` | End-to-end CNN | Prototyping with subset, scaling up, data flow tracing |

### Lesson 2 -- Training, Optimization & NLP

| Module | File | Topic | Key Concepts |
|--------|------|-------|-------------|
| 2.1 | `learning_rate/main.py` | LR & evaluation metrics | LR sweep, Precision, Recall, F1, torchmetrics, imbalanced data |
| 2.1 | `scheduler/main.py` | LR schedulers | StepLR, CosineAnnealingLR, ReduceLROnPlateau |
| 2.1 | `optuna/main.py` | Hyperparameter tuning | Optuna objective function, flexible CNN, parameter importance |
| 2.1 | `efficiency_performance/main.py` | Model efficiency | Comparing architectures, training speed vs. accuracy |
| 2.2 | `transfer_learning/main.py` | Transfer learning | Feature extraction, fine-tuning, full retraining, freezing layers |
| 2.2 | `pre_processing/main.py` | Image preprocessing | Oxford-IIIT Pets, augmentation visualization, dataset inspection |
| 2.3 | `embeddings/main.py` | Word embeddings | GloVe, word analogies, cosine similarity, training custom embeddings, BERT context |
| 2.3 | `tokenizer/main.py` | Tokenization | BERT tokenizer, wordpiece, subword tokenization |
| 2.3 | `simple_text_classifier/main.py` | Text classification | Recipe classifier, embedding layer, LSTM-style model |
| 2.3 | `pretrained/main.py` | Pretrained NLP models | HuggingFace transformers, BERT fine-tuning for classification |
| 2.4 | `lightning/main.py` | PyTorch Lightning | LightningModule, LightningDataModule, Trainer, profiling |
| 2.4 | `advance_lightning/main.py` | Advanced Lightning | Custom callbacks, early stopping, learning rate monitoring |
| 2.4 | `optimizing_dataloader/main.py` | DataLoader optimization | num_workers, pin_memory, prefetching, benchmarking |
| 2.4 | `diagnostic_assistant/main.py` | End-to-end Lightning | Full pipeline with callbacks, metrics, and checkpointing |

### Lesson 3 -- Advanced Architectures & Deployment

| Module | File | Topic | Key Concepts |
|--------|------|-------|-------------|
| 3.1 | `resnet/main.py` | ResNet | Residual connections, skip connections, skip connections |
| 3.1 | `densenet/main.py` | DenseNet | Dense connections, feature reuse, growth rate |
| 3.1 | `siamese_network/main.py` | Siamese networks | Contrastive loss, signature verification, similarity learning |
| 3.1 | `classification/main.py` | Advanced classification | Multiple architectures comparison |
| 3.2 | `interpreting/main.py` | Model interpretation | Convolution filters visualization, feature map inspection |
| 3.2 | `saliency_and_class_activation_map/main.py` | Saliency & CAM | Grad-CAM, saliency maps, class activation maps |
| 3.2 | `stable_diffusion/main.py` | Stable Diffusion | DDPM, denoising, image generation |
| 3.2 | `fruit_quality/main.py` | Practical application | Fruit quality classification using diffusion-augmented data |
| 3.3 | `self_attention/main.py` | Self-attention | Q/K/V, scaled dot-product, multi-head attention, next-word prediction |
| 3.3 | `decoder_block/main.py` | Transformer decoder | Causal masking, autoregressive generation |
| 3.3 | `transformer_encoder/main.py` | Transformer encoder | Full encoder block, positional encoding, text classification |
| 3.3 | `translation/main.py` | Machine translation | Encoder-decoder, attention visualization, seq2seq |
| 3.4 | `MLflow/main.py` | MLflow tracking | Experiment tracking, parameter logging, model registry, confusion matrix |
| 3.4 | `ONNX/main.py` | ONNX deployment | PyTorch->ONNX export, ONNX Runtime inference, ONNX->TensorFlow conversion |

---

## Key Concepts Glossary

### Fundamentals
- **Tensor**: Multi-dimensional array (like NumPy ndarray) that can run on GPU and supports autograd
- **Autograd**: PyTorch's automatic differentiation engine -- computes gradients for backpropagation
- **nn.Module**: Base class for all neural network components in PyTorch
- **nn.Sequential**: Container that chains layers in order (forward pass is automatic)

### Training
- **Loss Function**: Measures how wrong predictions are (MSE for regression, CrossEntropy for classification)
- **Optimizer**: Updates model weights to minimize loss (SGD, Adam, AdamW)
- **Learning Rate**: Step size for weight updates -- too high = unstable, too low = slow
- **Epoch**: One complete pass through the entire training dataset
- **Batch**: A subset of data processed together (32, 64, 128 are common sizes)
- **Backpropagation**: Computing gradients of the loss with respect to all weights

### Data
- **Transform**: Pipeline of operations applied to data (resize, normalize, augment)
- **Normalization**: (x - mean) / std -- standardizes input range for stable training
- **Data Augmentation**: Random transformations (flip, rotate, crop) to increase training variety
- **Train/Val/Test Split**: Training learns, validation tunes, test evaluates final performance

### Architectures
- **CNN (Convolutional Neural Network)**: Best for images -- learns local patterns (edges -> textures -> objects)
- **ResNet**: CNN with skip connections that allow training very deep networks
- **Transformer**: Attention-based architecture that powers modern NLP (GPT, BERT)
- **Self-Attention**: Mechanism where each word attends to all other words in a sequence
- **Transfer Learning**: Using a pre-trained model as starting point for a new task

### Deployment
- **ONNX**: Universal model format that runs across frameworks and platforms
- **MLflow**: Platform for tracking experiments, logging metrics, and managing models
- **PyTorch Lightning**: Framework that simplifies training code by removing boilerplate

---

## The Standard Training Loop (5 Steps)

Every deep learning training loop follows this pattern:

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()       # 1. Clear old gradients
        outputs = model(inputs)     # 2. Forward pass (make predictions)
        loss = loss_fn(outputs, y)  # 3. Compute loss (how wrong?)
        loss.backward()             # 4. Backward pass (compute gradients)
        optimizer.step()            # 5. Update weights (learn from mistakes)
```

---

## Prerequisites

- Python 3.10+
- PyTorch (with CUDA or MPS support for GPU acceleration)
- See `requirements.txt` for full dependencies

## How to Run

Most files can be run directly:
```bash
cd /path/to/pytorch
python lesson1/module1/tensor.py
```

Some files require data that is auto-downloaded on first run. Make sure you have internet access the first time you run each module.

## Virtual Environment

The project includes two environments:
- `venv/` -- Standard Python virtual environment
- `tf_env/` -- Environment with TensorFlow (needed for ONNX->TF conversion in Lesson 3 Module 4)

Activate before running:
```bash
source venv/bin/activate       # For most lessons
source tf_env/bin/activate     # For ONNX->TensorFlow conversion
```

---

## Model Checkpoints

Trained models are saved in:
- `saved_models/` -- General checkpoints (Siamese network, DenseNet)
- `best_model/` -- Best-performing checkpoints
- `fruit_veg_model.onnx` -- ONNX-exported fruit/vegetable classifier

---

## Timeline

- **Lesson 1** (Dec 17-20): Fundamentals -- tensors, datasets, CNNs
- **Lesson 2** (Dec 23 - Jan 31): Optimization, transfer learning, NLP, Lightning
- **Lesson 3** (Feb 4 - Apr 17): Advanced architectures, transformers, deployment

# PyTorch Machine Learning Course Notes

## Overview

This project contains my hands-on notes from a PyTorch deep learning course, covering everything from basic tensors to deploying models in production. Each lesson builds on the previous one, progressing from fundamentals to advanced architectures.

---

## If You Want To... (Goal-to-Learning Map)

Start here. Find the project you want to build, then follow the recommended reading path.

| If you want to... | Start with | Then study | Deep dive |
|---|---|---|---|
| **Build an image classifier** (detect sensitive content, filter inappropriate images, classify products by photo) | L1-M2 `digit_detective.py` -- your first image classifier | L1-M4 `cnn/main.py` -- CNNs for real images | L2-M2 `transfer_learning/main.py` (use pre-trained models), L3-M2 `saliency_and_class_activation_map/main.py` (see what model focuses on) |
| **Detect if two images are similar** (face recognition, signature verification, duplicate detection) | L1-M4 `cnn_block.py` -- learn how CNNs extract features | L2-M2 `transfer_learning/main.py` -- reuse powerful feature extractors | L3-M1 `siamese_network/main.py` -- learn similarity with contrastive/triplet loss |
| **Classify text** (spam detection, sentiment analysis, topic categorization, toxicity detection) | L2-M3 `embeddings/main.py` -- convert words to numbers | L2-M3 `simple_text_classifier/main.py` -- build your first text model | L2-M3 `pretrained/main.py` (fine-tune BERT), L3-M3 `transformer_encoder/main.py` (build from scratch) |
| **Translate between languages** | L2-M3 `tokenizer/main.py` -- how text becomes numbers | L3-M3 `self_attention/main.py` -- the core mechanism behind translation | L3-M3 `translation/main.py` -- full encoder-decoder with cross-attention |
| **Predict a number from features** (house prices, sales forecast, risk score) | L1-M1 `tensor.py` -- the data format | L1-M1 `leaner.py` -- linear regression, then `non_leaner.py` -- add hidden layers | L2-M1 `optuna/main.py` (tune hyperparameters automatically) |
| **Train a very deep network** (100+ layers that actually converge) | L1-M4 `cnn_block.py`, `cnn/main.py` -- standard CNNs | L3-M1 `resnet/main.py` -- skip connections solve the depth problem | L3-M1 `densenet/main.py` -- feature reuse for efficient deep networks |
| **Generate images from text prompts** (like DALL-E, Midjourney) | L1-M4 `cnn_block.py` -- understand convolutions first | L3-M2 `stable_diffusion/main.py` -- diffusion models explained | L3-M2 `fruit_quality/option.py` -- step-by-step denoising visualization |
| **Augment a small dataset with synthetic images** | L1-M3 `data_management/main.py` -- standard augmentation (flip, rotate) | L3-M2 `fruit_quality/main.py` -- generate synthetic training data with diffusion | L3-M2 `stable_diffusion/main.py` -- control what gets generated |
| **Understand what your model is doing** (explainability, debugging, trust) | L1-M4 `debugging/main.py` -- debug shapes and activations | L3-M2 `interpreting/main.py` -- visualize filters and feature maps | L3-M2 `saliency_and_class_activation_map/main.py` -- Grad-CAM shows where model looks |
| **Deploy a model to production** (API, mobile app, edge device, browser) | L1-M4 `cnn/main.py` -- train a model to export | L3-M4 `ONNX/main.py` -- export to universal format | L3-M4 `pruning/main.py` + `quantization/main.py` (shrink model), `metro_city/main.py` (full pipeline) |
| **Make a model smaller and faster** (run on phone, embedded device, or cheaper cloud) | L1-M4 `debugging/main.py` -- understand model structure and parameters | L3-M4 `pruning/main.py` -- remove unnecessary weights | L3-M4 `quantization/main.py` (FP32 to INT8), `metro_city/main.py` (prune + fuse + quantize end-to-end) |
| **Track and compare experiments** (which training run was best?) | L2-M1 `learning_rate/main.py` -- see why comparing runs matters | L2-M4 `lightning/main.py` -- clean training code | L3-M4 `MLflow/main.py` -- log parameters, metrics, artifacts, compare in web UI |
| **Use a pre-trained model for a new task** (save time, need less data) | L1-M4 `cnn/main.py` -- understand what gets trained | L2-M2 `transfer_learning/main.py` -- freeze, fine-tune, full retrain | L2-M2 `pre_processing/main.py` (prepare data for pre-trained models) |
| **Automate hyperparameter tuning** (stop guessing learning rate, batch size, architecture) | L2-M1 `learning_rate/main.py` -- see how LR affects training | L2-M1 `scheduler/main.py` -- adjust LR automatically | L2-M1 `optuna/main.py` -- let Optuna search for the best combination |
| **Build a search engine for similar images** | L1-M4 `cnn/main.py` -- CNNs produce feature vectors | L3-M1 `siamese_network/main.py` -- learn distance between images | L2-M3 `embeddings/main.py` (understand embedding spaces and cosine similarity) |
| **Detect anomalies or outliers** (fraud detection, defect inspection, rare disease) | L1-M1 `tensor.py` -- compute statistics on data | L3-M1 `siamese_network/main.py` -- learn what "normal" looks like | L2-M1 `learning_rate/main.py` (precision/recall/F1 for imbalanced data) |
| **Build a chatbot or text generator** | L2-M3 `tokenizer/main.py` -- text to numbers | L3-M3 `self_attention/main.py` -- attention mechanism | L3-M3 `decoder_block/main.py` -- autoregressive generation, one token at a time |
| **Classify medical images** (X-ray, MRI, pathology slides) | L1-M2 `digit_detective.py` -- first image classifier | L1-M4 `cnn/main.py` -- CNNs on real images | L2-M2 `transfer_learning/main.py` (fine-tune on medical data), L3-M2 `saliency_and_class_activation_map/main.py` (explain predictions to doctors) |
| **Speed up training with better data loading** | L1-M3 `data_pileline/main.py` -- custom DataLoaders | L2-M4 `optimizing_dataloader/main.py` -- num_workers, pin_memory, benchmarking | L2-M4 `lightning/main.py` -- Lightning automates the pipeline |
| **Write clean, organized training code** | L1-M1 `leaner.py` -- raw PyTorch training loop | L2-M4 `lightning/main.py` -- Lightning reduces boilerplate | L2-M4 `diagnostic_assistant/main.py` -- full end-to-end Lightning project |
| **Process text with BERT or GPT-style models** | L2-M3 `embeddings/main.py` -- word representations | L2-M3 `pretrained/main.py` -- fine-tune BERT | L3-M3 `transformer_encoder/main.py` (build Transformer from scratch) |
| **Classify audio or time-series data** | L1-M1 `tensor.py` -- 1D tensor operations | L1-M4 `cnn_block.py` -- 1D convolutions work on sequences | L3-M3 `self_attention/main.py` -- attention works on any sequence |

---

## Learning Paths by Goal

### Image Classification Path
The most common ML task. Start here if you want to work with images.
```
L1-M1 tensor.py (tensors)
  -> L1-M2 digit_detective.py (first classifier)
  -> L1-M4 cnn/main.py (CNNs)
  -> L2-M2 transfer_learning/main.py (pre-trained models)
  -> L2-M1 optuna/main.py (tune it)
  -> L3-M1 resnet/main.py (deeper architectures)
  -> L3-M2 saliency_and_class_activation_map/main.py (explain it)
```

### NLP / Text Path
Start here if you want to work with text, language, or chatbots.
```
L1-M1 tensor.py (tensors)
  -> L2-M3 embeddings/main.py (words to numbers)
  -> L2-M3 tokenizer/main.py (text to tokens)
  -> L2-M3 simple_text_classifier/main.py (first text model)
  -> L2-M3 pretrained/main.py (fine-tune BERT)
  -> L3-M3 self_attention/main.py (attention mechanism)
  -> L3-M3 transformer_encoder/main.py (full Transformer)
  -> L3-M3 translation/main.py (seq2seq)
```

### Deployment Path
Start here if you already have a trained model and want to ship it.
```
L3-M4 ONNX/main.py (export model)
  -> L3-M4 MLflow/main.py (track experiments)
  -> L3-M4 pruning/main.py (make smaller)
  -> L3-M4 quantization/main.py (make faster)
  -> L3-M4 metro_city/main.py (full pipeline)
```

### Similarity / Verification Path
Start here for face recognition, signature verification, or comparing images.
```
L1-M4 cnn/main.py (CNNs extract features)
  -> L2-M2 transfer_learning/main.py (powerful feature extractors)
  -> L3-M1 siamese_network/main.py (learn similarity)
  -> L2-M3 embeddings/main.py (embedding spaces)
```

---

## What You Will Learn in Each Module

### Lesson 1 -- PyTorch Fundamentals

**Module 1: Tensors & Linear Regression**

The alphabet of deep learning. Everything is a tensor.
- Create, reshape, slice, and compute on tensors (multi-dimensional arrays)
- Convert data from Python lists, NumPy, and Pandas into tensors
- Build a linear regression model that predicts a number from input features
- Add hidden layers with ReLU activation to learn non-linear patterns
- Write a complete training loop from scratch (the 5 steps that appear in every project)

When done, you can: load tabular data and train a model to predict continuous values.

**Module 2: Datasets, Transforms & Image Classification**

Your first image classifier. The standard data pipeline.
- Load image datasets from torchvision (MNIST handwritten digits, EMNIST letters)
- Apply transforms: resize, convert to tensor, normalize with mean/std
- Create DataLoaders that batch and shuffle data for training
- Build a neural network that classifies images into categories
- Compute dataset statistics (mean, std) from scratch for normalization

When done, you can: load an image dataset and train a model to classify images.

**Module 3: Custom Datasets & Data Pipelines**

Real data is messy. Handle it.
- Build custom Dataset classes with `__len__` and `__getitem__`
- Load images from folders, CSV files, and MATLAB .mat files
- Create train/validation/test splits with different transforms for each
- Handle corrupted images and bad data gracefully (skip errors, keep training)
- Use data augmentation (random flip, rotation, crop) to increase training variety

When done, you can: take any messy real-world dataset and turn it into a clean DataLoader.

**Module 4: CNN Basics & Image Classification**

Convolutions change everything for images.
- Understand Conv2d (sliding window that detects patterns), BatchNorm, MaxPool, Dropout
- Build a CNN block by block and understand tensor shapes at each layer
- Train on CIFAR-100 with proper train/val transforms and best-model checkpointing
- Debug common problems: shape mismatches, vanishing activations, wrong layer sizes
- Trace data flow through the network to catch errors early

When done, you can: design, build, debug, and train a CNN for any image classification task.

---

### Lesson 2 -- Training, Optimization & NLP

**Module 1: Learning Rates, Schedulers & Hyperparameter Tuning**

Good training is as important as a good model.
- Sweep learning rates and see how they affect convergence (too high = diverge, too low = crawl)
- Use evaluation metrics beyond accuracy: Precision, Recall, F1 (critical for imbalanced data)
- Apply LR schedulers: StepLR, CosineAnnealingLR, ReduceLROnPlateau
- Use Optuna to automatically search for the best hyperparameters (LR, batch size, architecture)
- Compare model architectures for speed vs accuracy trade-offs

When done, you can: tune any model's training process for maximum performance.

**Module 2: Transfer Learning & Data Preprocessing**

Don't train from scratch. Stand on the shoulders of giants.
- Load pre-trained models (ResNet, VGG) trained on ImageNet (millions of images)
- Strategy 1: Feature extraction -- freeze all layers, train only the head (fastest)
- Strategy 2: Fine-tuning -- unfreeze some layers, train with a small learning rate
- Strategy 3: Full retraining -- unfreeze everything, train end-to-end (slowest, most data needed)
- Prepare and inspect image datasets with proper preprocessing pipelines

When done, you can: take a pre-trained model and adapt it to your own dataset in minutes.

**Module 3: NLP -- Embeddings, Tokenizers & Text Classification**

Teach machines to understand text.
- Word embeddings: convert words to dense vectors (GloVe, custom-trained)
- Measure word similarity with cosine similarity, solve word analogies
- Tokenization: split text into tokens (BERT WordPiece tokenizer)
- Build a text classifier: embedding layer -> model -> prediction
- Fine-tune a pre-trained BERT model for text classification using HuggingFace

When done, you can: take raw text data and build a model that classifies it by topic, sentiment, or intent.

**Module 4: PyTorch Lightning & Data Pipeline Optimization**

Write less code, train faster.
- Replace the manual training loop with LightningModule (training_step, validation_step)
- Organize data logic into LightningDataModule (prepare_data, setup, dataloader)
- Use the Trainer to handle device management, epoch loops, and logging automatically
- Add callbacks: early stopping, learning rate monitoring, model checkpointing
- Optimize DataLoader speed: num_workers, pin_memory, benchmarking

When done, you can: write clean, professional training code and optimize data loading speed.

---

### Lesson 3 -- Advanced Architectures & Deployment

**Module 1: ResNet, DenseNet & Siamese Networks**

Go deeper and learn similarity.
- ResNet: skip connections (output = F(x) + x) allow training 50-152 layer networks
- DenseNet: dense connections (concatenate all previous outputs) for feature reuse
- Siamese networks: two identical branches that learn to compare (not classify)
- Contrastive and triplet loss: train on (anchor, positive, negative) image triplets
- Compare multiple architectures on the same task to see trade-offs

When done, you can: choose the right architecture for deep image tasks and similarity/comparison tasks.

**Module 2: Model Interpretability & Generative Models**

Open the black box and generate new data.
- Visualize convolutional filters: see what patterns each filter detects (edges, textures, objects)
- Feature maps: see how an image transforms at each layer of the network
- Saliency maps: which pixels matter most for the prediction (gradient-based)
- Grad-CAM: which image region the model focused on (activation-based, class-specific)
- Stable Diffusion: generate images from text prompts, understand denoising process
- Use diffusion to augment small datasets with synthetic training images

When done, you can: explain why a model made a prediction and generate synthetic images for data augmentation.

**Module 3: Transformers -- Self-Attention, Encoder, Decoder & Translation**

The architecture behind GPT, BERT, and modern AI.
- Self-attention: Q/K/V mechanism, scaled dot-product, multi-head attention
- Positional encoding: inject order information (attention is order-invariant by default)
- Encoder: bidirectional attention (sees all tokens), good for understanding text
- Decoder: causal masking (sees only past tokens), good for generating text
- Full encoder-decoder for machine translation with cross-attention between source and target

When done, you can: understand and build Transformer models for any sequence task (text, code, time-series).

**Module 4: MLOps -- MLflow, ONNX, Pruning, Quantization & Full Pipeline**

Ship your model to the real world.
- MLflow: log parameters, metrics, artifacts; compare runs in web UI; register models
- ONNX: export PyTorch model to universal format; run on ONNX Runtime without PyTorch
- Pruning: remove unnecessary weights (L1 unstructured, Ln structured, global)
- Quantization: reduce precision from FP32 to INT8 (dynamic, static, quantization-aware training)
- Full pipeline (metro_city): prune -> layer fusion -> quantize -> QAT with benchmarking

When done, you can: track experiments, export models, compress them, and prepare them for production deployment.

---

## Course Structure

```
pytorch/
+-- README.md              # You are here
+-- STRATEGY.md            # Learning improvement plan
+-- requirements.txt        # Dependencies
+-- requirement3.12.txt     # Dependencies (Python 3.12)
|
+-- lesson1/                # L1: PyTorch Fundamentals
|   +-- module1/            # Tensors & Linear Regression
|   |   +-- LEARNING_GUIDE.md
|   +-- module2/            # Datasets, Transforms & Image Classification
|   |   +-- LEARNING_GUIDE.md
|   +-- module3/            # Custom Datasets & Data Pipelines
|   |   +-- LEARNING_GUIDE.md
|   +-- module4/            # CNN Basics & Image Classification
|       +-- LEARNING_GUIDE.md
|
+-- lesson2/                # L2: Training, Optimization & NLP
|   +-- module1/            # Learning Rates, Schedulers, Optuna
|   |   +-- LEARNING_GUIDE.md
|   +-- module2/            # Transfer Learning & Preprocessing
|   |   +-- LEARNING_GUIDE.md
|   +-- module3/            # NLP: Embeddings, Tokenizers, Text Classification
|   |   +-- LEARNING_GUIDE.md
|   +-- module4/            # PyTorch Lightning & DataLoader Optimization
|       +-- LEARNING_GUIDE.md
|
+-- lesson3/                # L3: Advanced Architectures & Deployment
|   +-- module1/            # ResNet, DenseNet, Siamese Networks
|   |   +-- LEARNING_GUIDE.md
|   +-- module2/            # Interpretability, Saliency Maps, Stable Diffusion
|   |   +-- LEARNING_GUIDE.md
|   +-- module3/            # Transformers: Attention, Encoder, Decoder, Translation
|   |   +-- LEARNING_GUIDE.md
|   +-- module4/            # MLOps: MLflow, ONNX, Pruning, Quantization
|       +-- LEARNING_GUIDE.md
|
+-- models/                 # All saved model checkpoints
|   +-- checkpoints/        # MLflow epoch checkpoints
|   +-- cifar10/            # CIFAR-10 trained models
|   +-- densenet_siamese/   # DenseNet and Siamese network models
|   +-- metro_city/         # Metro city optimization pipeline models
|   +-- pruning/            # Pruned and unpruned model weights
|   +-- saved/              # Lightning diagnostic model
|
+-- outputs/                # All generated outputs
|   +-- figures/            # Plots and visualizations
|   +-- intermediate_steps/ # Stable Diffusion step-by-step denoising
|   +-- mlflow/             # MLflow tracking database and runs
|   +-- onnx_export/        # ONNX and TensorFlow exported models
|   +-- profiler/           # Lightning profiler reports
|   +-- dataloader_benchmarks/  # DataLoader speed benchmarks
|   +-- synthetic/          # Diffusion-generated synthetic images
|
+-- data/                   # Datasets (auto-downloaded)
+-- exercises/              # Practice exercises
+-- venv/                   # Standard Python virtual environment
+-- tf_env/                 # TensorFlow environment (ONNX conversion)
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
| 3.1 | `resnet/main.py` | ResNet | Residual connections, skip connections |
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
| 3.4 | `pruning/main.py` | Model pruning | L1 unstructured, Ln structured, global pruning, sparsity |
| 3.4 | `quantization/main.py` | Model quantization | Dynamic, static, and QAT quantization, INT8, layer fusion |
| 3.4 | `metro_city/main.py` | Full optimization pipeline | Pruning + fusion + quantization + QAT, benchmarking, production deployment |

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
- **Pruning**: Setting unnecessary weights to zero to create a sparse (smaller) model
- **Quantization**: Reducing weight precision from FP32 to INT8 for faster inference
- **Layer Fusion**: Combining Conv+BN+ReLU into a single operation for efficiency
- **QAT (Quantization-Aware Training)**: Training with simulated quantization to preserve accuracy

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

Trained models are organized in `models/`:
- `models/checkpoints/` -- MLflow epoch checkpoints (best model per epoch)
- `models/cifar10/` -- CIFAR-10 trained CNN (best and final)
- `models/densenet_siamese/` -- DenseNet and Siamese network models
- `models/metro_city/` -- Full optimization pipeline models (base, pruned, quantized, QAT)
- `models/pruning/` -- Pruned and unpruned model weights for comparison
- `models/saved/` -- Lightning diagnostic assistant model

Generated outputs are in `outputs/`:
- `outputs/figures/` -- Plots (confusion matrix, F1 comparison, diffusion timelapse)
- `outputs/mlflow/` -- MLflow tracking database and experiment runs
- `outputs/onnx_export/` -- ONNX and TensorFlow exported models
- `outputs/intermediate_steps/` -- Stable Diffusion step-by-step denoising images

---

## Timeline

- **Lesson 1** (Dec 17-20): Fundamentals -- tensors, datasets, CNNs
- **Lesson 2** (Dec 23 - Jan 31): Optimization, transfer learning, NLP, Lightning
- **Lesson 3** (Feb 4 - Apr 17): Advanced architectures, transformers, deployment

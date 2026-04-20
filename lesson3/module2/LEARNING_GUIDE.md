# Learning Guide: Lesson 3, Module 2 -- Model Interpretability and Generative Models

## Module Overview

Neural networks are often called "black boxes." This module opens the box:
you learn to visualize what the model sees (filters, feature maps, saliency maps)
and to generate new images using diffusion models (Stable Diffusion).

## Recommended Reading Order

1. **interpreting/main.py** -- Visualizing convolutional filters and feature maps
2. **saliency_and_class_activation_map/main.py** -- Saliency maps and Grad-CAM
3. **stable_diffusion/main.py** -- Image generation with diffusion models
4. **fruit_quality/main.py** -- Practical application: fruit quality classification
5. **fruit_quality/option.py** -- Optional: Stable Diffusion denoising visualization

## Concept Map

```
Interpretability (understanding what the model learned)
   |
   +--> Filter visualization: what pattern does each filter detect?
   +--> Feature maps: how does an image get transformed at each layer?
   +--> Saliency maps: which pixels matter most for the prediction?
   +--> Grad-CAM: which image region did the model focus on?
   |
   v
Generative Models (creating new data)
   |
   +--> Diffusion Models: add noise, then learn to remove it
   +--> DDPM: simple denoising diffusion
   +--> Stable Diffusion: text-guided image generation in latent space
   |
   v
Practical Application
   |
   +--> Fruit quality: healthy vs rotten classification
   +--> Forward hooks: tap into any layer's output
   +--> Diffusion augmentation: generate synthetic training data
```

## File Summaries

### interpreting/main.py
Visualizes what a CNN has learned: individual filter patterns, feature maps
at each layer, and max pooling effects. Uses a simple example image (ball).
Focus on: how early filters detect edges/textures and later filters detect
complex patterns.

### saliency_and_class_activation_map/main.py
Implements saliency maps (gradient-based) and Grad-CAM (activation-based) on
ResNet-50. Shows which image regions the model focused on for its prediction.
Focus on: the difference between gradient-based (pixel importance) and
activation-based (region importance) explanations.

### stable_diffusion/main.py
Loads and runs Stable Diffusion for text-to-image generation. Also shows
DDPM (basic diffusion) for generating images from pure noise.
Focus on: the forward process (adding noise) vs reverse process (denoising).

### fruit_quality/main.py
Practical application: classifying fruit as healthy or rotten using a pre-trained
model. Uses forward hooks to capture intermediate activations.
Focus on: the hook pattern for inspecting any layer without modifying model code.

### fruit_quality/option.py
Optional extension: visualizes the diffusion denoising process step-by-step and
generates a timelapse movie of images being created from noise.
Focus on: understanding how diffusion gradually refines noise into images.

## Common Questions

**Q: Why does model interpretability matter?**
A: If a medical AI says "tumor detected," you need to know WHY it thinks so.
Interpretability tools show you what the model focused on, helping build trust
and catch errors (e.g., the model might be looking at the hospital name on
an X-ray instead of the actual tissue).

**Q: How does Stable Diffusion work in simple terms?**
A: Imagine taking a photograph and slowly adding static/noise until it's pure
noise. Diffusion models learn to REVERSE this process: starting from noise
and gradually removing it to reveal a clean image. A text prompt guides which
image to generate.

**Q: What is a forward hook?**
A: A function you attach to any layer that runs whenever data passes through it.
It lets you capture intermediate outputs (activations) without changing the
model code. Like tapping a phone line to listen in on a conversation.

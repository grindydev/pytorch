"""
Lesson 3 - Module 2: Stable Diffusion -- Image Generation with Denoising
==========================================================================
WHAT YOU'LL LEARN:
  * How diffusion models work: adding noise then learning to remove it
  * DDPM (Denoising Diffusion Probabilistic Models) pipeline
  * Stable Diffusion: text-to-image generation with latent diffusion
  * How to load and run pre-trained diffusion pipelines from Hugging Face
  * The forward (noise-adding) and reverse (denoising) process

KEY CONCEPT:
  DIFFUSION MODELS work in two phases:
  1. FORWARD PROCESS: Gradually add Gaussian noise to an image until it's
     pure noise (this is fixed, not learned)
  2. REVERSE PROCESS: A neural network learns to predict and remove the
     noise, step by step, eventually generating a clean image

  STABLE DIFFUSION does this in a compressed "latent space" (not pixel space),
  making it much more efficient. A text encoder guides the denoising so the
  generated image matches the text prompt.
"""

import sys
import warnings

from diffusers import StableDiffusionPipeline, DDPMPipeline
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import numpy as np

import helper_utils
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

def load_model_pipeline(model_id, pipeline_class, device=None, **kwargs):
    """
    Loads a Hugging Face pipeline with caching logic.
    
    Args:
        model_id (str): The repository ID (e.g., "stabilityai/stable-diffusion-2-base").
        pipeline_class (class): The class to load (e.g., StableDiffusionPipeline).
        device (torch.device, optional): Device to move the pipeline to.
        **kwargs: Additional arguments for from_pretrained (e.g., torch_dtype, variant).
    
    Returns:
        The loaded pipeline.
    """
    # Define the directory for caching model files
    cache_dir = Path.cwd() / "data/models"
    
    # Construct the specific cache directory name based on the model ID
    model_cache_dir_name = f"models--{model_id.replace('/', '--')}"
    
    # Create the full path to the model cache
    model_cache_path = cache_dir / model_cache_dir_name

    # Verify model snapshot integrity or perform extraction if missing
    helper_utils.check_model_snapshot(model_cache_path)

    # Instantiate the pipeline with local files and cache settings
    pipe = pipeline_class.from_pretrained(
        pretrained_model_name_or_path=model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        **kwargs  # Forward additional configuration arguments
    )

    # Transfer the pipeline to the specified device if one is provided
    if device:
        pipe = pipe.to(device)

    return pipe

model_id = "stabilityai/stable-diffusion-2-base"

# Load the pipeline using the helper function
pipe = load_model_pipeline(
    model_id=model_id,                       # The specific model repository ID (Stable Diffusion 2)
    pipeline_class=StableDiffusionPipeline,  # The class definition for the pipeline you want to load
    device=device,                           # Moves the pipeline to the detected hardware (CUDA/MPS/CPU)
    # Extra arguments specific to this model:
    torch_dtype=torch.float16,               # Uses half-precision (float16) to reduce memory usage
    variant="fp16"                           # Loads the specific fp16 weights file for efficiency
)

print("\nLoading Complete!")

# Set the seed for reproducibility
generator = torch.Generator(device=device).manual_seed(42)

prompt = "A puppy riding a skateboard in Times Square."

# # Using 40 steps provides a good balance between quality and speed
# # (Typical range: 20-50 for fast generation, 50-150 for high quality)
# images = pipe(
#     prompt,                 # What you want the model to create
#     num_inference_steps=40, # How many denoising steps to use (more steps = more detail/compute)
#     generator=generator     # Ensures reproducible noise/randomness
# ).images

# helper_utils.display_images(images[0])

# images = pipe(
#     prompt,                 # What you want the model to create
#     num_inference_steps=40, # How many denoising steps to use (more steps = more detail/compute)
#     generator=generator     # Ensures reproducible noise/randomness
# ).images


# helper_utils.display_images(images[0])

# # First run
# generator = torch.Generator(device=device).manual_seed(42)

# images = pipe(
#     prompt,                 # What you want the model to create
#     num_inference_steps=40, # How many denoising steps to use (more steps = more detail/compute)
#     generator=generator     # Ensures reproducible noise/randomness
# ).images

# helper_utils.display_images(images[0])

# # Second run
# generator = torch.Generator(device=device).manual_seed(42)
# images = pipe(
#     prompt,                 # What you want the model to create
#     num_inference_steps=40, # How many denoising steps to use (more steps = more detail/compute)
#     generator=generator     # Ensures reproducible noise/randomness
# ).images

# helper_utils.display_images(images[0])

# image_list = pipe(
#     prompt,                    # What you want the model to create
#     num_inference_steps=40,    # How many denoising steps to use (more steps = more detail/compute)
#     generator=generator,       # Ensures reproducible noise/randomness
#     num_images_per_prompt = 3  # Number of images to generate
# ).images

# helper_utils.display_images(image_list)


# prompt = "a cute dog with a red bandana, sitting in a lush park"

# # Redefining the generator
# generator = torch.Generator(device=device).manual_seed(42)

# # Fewer steps (fast, less detail)
# image_fast = pipe(prompt, num_inference_steps=10, generator=generator).images[0]

# # Default/standard (balanced quality)
# image_standard = pipe(prompt, num_inference_steps=50, generator=generator).images[0]

# # More steps (slower, more detail)
# image_high_quality = pipe(prompt, num_inference_steps=200, generator=generator).images[0]


# images = [image_fast, image_standard, image_high_quality]
# titles = ["10 steps (fast)", "50 steps (standard)", "200 steps (high quality)"]

# helper_utils.display_images(images, titles=titles)


# prompt = ("A surreal landscape with floating clocks, melting trees, "
#           "and a purple sky, in the style of Salvador Dalí")

# guidance_scales = [5, 7.5, 12]

# # Generate all images first (so the generator seed doesn't advance unpredictably)
# images = []
# for gs in guidance_scales:
#     # It's important to re-create the generator for reproducibility
#     generator = torch.Generator(device=device).manual_seed(42)
#     print(f"Generating image for guidance scale = {gs}")
#     img = pipe(prompt, generator=generator, guidance_scale=gs).images[0]
#     images.append(img)
#     # Adding line space
#     print()

# helper_utils.display_images(
#     images, 
#     titles=[f"guidance_scale = {gs}" for gs in guidance_scales]
# )


# prompt = "A group of realistic teddy bears eating pizza at a birthday party"
# negative_prompt = "pepperoni, deformed hands, extra limbs, blurry, out of focus, text, watermark"

# # Generate without negative prompt
# generator = torch.Generator(device=device).manual_seed(42)
# result_no_neg = pipe(
#     prompt, 
#     generator=generator, 
#     num_inference_steps=30
# )

# # Re-seed to ensure the same starting noise for fair comparison
# generator = torch.Generator(device=device).manual_seed(42)
# result_with_neg = pipe(
#     prompt, 
#     negative_prompt=negative_prompt, 
#     generator=generator, 
#     num_inference_steps=30
# )

# imgs = [result_no_neg.images[0], result_with_neg.images[0]]
# titles = ['Without negative_prompt', 'With negative_prompt']

# helper_utils.display_images(imgs, titles=titles)

# def save_intermediate_steps(step_index, timestep, latents):
#     """
#     Callback function to save intermediate images during the denoising process.

#     Args:
#         step_index: The current step number in the generation process.
#         timestep: The specific timestep value associated with the current step.
#         latents: The latent tensor representation at the current step.
#     """
#     with torch.no_grad():
#         # Scale the latents using the VAE scaling factor specific to Stable Diffusion
#         # Rescale latents to the range expected by the VAE decoder
#         latents_input = latents / 0.18215
        
#         # Decode the latent representation into an image using the VAE
#         image = pipe.vae.decode(latents_input).sample
        
#         # Normalize the image data to the range [0, 1]
#         image = (image / 2 + 0.5).clamp(0, 1)
        
#         # Move image to CPU, rearrange dimensions, and convert to numpy array
#         image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    
#     # Convert the numpy array to a PIL Image object
#     pil_image = Image.fromarray((image * 255).astype("uint8"))
    
#     # Define the output directory for saving intermediate steps
#     outdir = "intermediate_steps"
    
#     # Create the directory if it does not already exist
#     os.makedirs(outdir, exist_ok=True)
    
#     # Save the current step's image to the output directory
#     pil_image.save(f"{outdir}/step_{step_index:02d}.png")
    
#     # Append the step index and image to the global list for later grid plotting
#     all_steps.append((step_index, pil_image))


# # List to store intermediate images
# all_steps = []

# generator = torch.Generator(device).manual_seed(42)

# prompt = "A puppy dog riding a skateboard in Times Square"
# negative_prompt = "cars"

# # Clear previous
# all_steps.clear()

# # Run the pipe with the callback
# pipe(
#     prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=40,
#     generator=generator,
#     callback=save_intermediate_steps,
#     callback_steps=1  # Every step
# )

# # Extract images and create titles
# imgs = [img for step, img in all_steps]
# titles = [f"Step {step+1}" for step, img in all_steps]

# helper_utils.display_grid(
#     images=imgs,
#     titles=titles,
#     n_rows=8, 
#     n_cols=5,
#     figsize=(15, 24),
#     main_title='Stable Diffusion 2: Denoising Steps'
# )

model_id = "google/ddpm-ema-bedroom-256"  # Bedroom images (256x256)

ddpm_pipeline = load_model_pipeline(
    model_id=model_id,
    pipeline_class=DDPMPipeline,
    device=device
)

print("\nLoading Complete!")

# Using 1000 steps for highest quality DDPM generation
# (DDPM typically requires more steps than Stable Diffusion for best results)
num_inference_steps = 1000

print(f"Using model: {model_id}")
print(f"Image resolution: 256x256 pixels")
print(f"Steps: {num_inference_steps}")

# Access the model and scheduler
model = ddpm_pipeline.unet
scheduler = ddpm_pipeline.scheduler

# Set up for visualization
scheduler.set_timesteps(num_inference_steps)

# --- Run pixel-space DDPM comparison ---
num_inference_steps = 100

print("Generating DDPM denoising comparison...")

gradual_images, full_removal_images = helper_utils.visualize_ddpm_denoising(
    ddpm_pipeline, 
    num_inference_steps=num_inference_steps
)

num_splits = 5

# Use the actual step IDs saved
actual_step_ids = [gi[0] for gi in gradual_images]
num_inference_steps = actual_step_ids[-1]  # last step
step_indices = [int(np.round(i * num_inference_steps / (num_splits - 1))) for i in range(num_splits - 1)]
step_indices.append(num_inference_steps)

print("Plotted step indices:", step_indices)

# Speed up access: build {step: image} dictionaries
grad_dict = {s: img for (s, img) in gradual_images}
full_dict = {s: img for (s, img) in full_removal_images}

# Use closest available step if step is not present
def get_closest_img(dct, step):
    # Find the closest available key in the dict
    best = min(dct.keys(), key=lambda k: abs(k-step))
    return dct[best]

images_full = [get_closest_img(full_dict, s) for s in step_indices]
images_grad = [get_closest_img(grad_dict, s) for s in step_indices]

# --- PLOTTING SECTION ---

# Prepare Titles
# Create a flat list of titles corresponding to (Row 1 images) + (Row 2 images)
titles_row1 = [f"Step {s}:\nPredicted Clean Image" for s in step_indices]
titles_row2 = [f"Step {s}:\nActual Noisy State" for s in step_indices]
all_titles = titles_row1 + titles_row2

# Pass a list of lists [[row1], [row2]] so the function knows to create 2 rows
helper_utils.display_grid(
    images=[images_full, images_grad],
    titles=all_titles,
    row_labels=["Model Prediction", "Actual Noise"],
    main_title='Pixel-Space DDPM: Denoising Analysis',
    figsize=(3.5 * len(step_indices), 7)
)


helper_utils.load_widget(pipe)
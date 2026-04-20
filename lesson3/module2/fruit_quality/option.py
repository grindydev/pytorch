"""
Lesson 3 - Module 2: Fruit Quality -- Optional Diffusion Visualization
=========================================================================

WHY THIS MATTERS:
  Seeing diffusion models in action makes the theory click. This optional
  file lets you generate images with Stable Diffusion and watch the
  step-by-step denoising process as a timelapse.

WHAT YOU'LL LEARN:
  * Loading and running Stable Diffusion 2 pipeline
  * Generating images from text prompts
  * Visualizing the denoising process step-by-step
  * Creating a timelapse of image generation

KEY CONCEPTS:
  Denoising -- Removing noise from an image, step by step
  Latent space -- Compressed representation where diffusion operates
  Text guidance -- Prompt controls what image is generated

HOW IT FITS:
  Optional companion to fruit_quality/main.py. Run this after understanding
  the basics of diffusion models from stable_diffusion/main.py.

PREREQUISITES:
  Complete stable_diffusion/main.py first.
"""

import gc
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

import helper_utils

# ====================== DEVICE ======================
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# ====================== LOAD PIPELINE ======================
def load_sd_pipeline(device):
    cache_dir = Path.cwd() / "data/models"
    
    print(" Loading Stable Diffusion 2-base from community mirror...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="sd2-community/stable-diffusion-2-base",
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
        safety_checker=None,
        use_safetensors=False,
    ).to(device)
    
    return pipe


# ====================== INITIALIZE PIPE ======================
if "pipe" in globals():
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

try:
    pipe = load_sd_pipeline(device)
    print("\n Pipeline loaded successfully!")
except Exception as e:
    print(f" Error while loading pipeline:\n{e}")
    pipe = None


# ====================== GENERATE IMAGE ======================
def generate_sd_image(pipe, prompt, negative_prompt, seed, steps, save_dir="outputs/synthetic"):
    if pipe is None:
        print(" Pipeline is not loaded.")
        return None
    
    device = pipe.device
    generator = torch.Generator(device=device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        generator=generator,
    ).images[0] 

    # Save image
    slug = "_".join(prompt.lower().split()[:3]) 
    out_dir = Path(save_dir) / slug 
    out_dir.mkdir(parents=True, exist_ok=True) 
    out_path = out_dir / f"img_{seed}.png" 
    
    image.save(out_path)
    print(f"\nImage saved to {out_path}\n")

    return image


# ====================== TEST GENERATION ======================
prompt = "A mango with a small hole made by a worm in the middle."
negative_prompt = "Fresh, intact."
seed = 42
steps = 50

if pipe is not None:
    try:
        img = generate_sd_image(
            pipe=pipe,
            prompt=prompt, 
            negative_prompt=negative_prompt,
            seed=seed, 
            steps=steps
        )
        
        # Fixed plotting
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        
    except Exception as e:
        print(f"Error during generation:\n{e}")
else:
    print("Cannot generate: pipe is None")


# ====================== DENOISING MOVIE (FIXED) ======================
def denoising_movie(pipe, prompt, seed, steps, capture_steps, save_grid_path="outputs/figures/stable_diffusion_denoising_timelapse.png"):
    if pipe is None:
        print(" Pipeline is not loaded.")
        return None

    frames = {}

    def grab_frame(pipeline, step_idx, timestep, callback_kwargs): 
        if step_idx in capture_steps:
            # FIXED: correct key + proper postprocessing
            latents = callback_kwargs["latents"]
            
            with torch.no_grad():
                img = pipe.vae.decode(
                    latents / pipe.vae.config.scaling_factor,
                    return_dict=False
                )[0] 
            
            pil = pipe.image_processor.postprocess(img, output_type="pil")[0]
            frames[step_idx] = pil
            
        return callback_kwargs

    generator = torch.Generator(pipe.device).manual_seed(seed)

    _ = pipe( 
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
        callback_on_step_end=grab_frame,
        # Required for newer diffusers versions
        callback_on_step_end_tensor_inputs=["latents"],
    ) 

    ordered_frames = [frames[s] for s in capture_steps]
    
    # Build 2x2 grid
    w, h = ordered_frames[0].size 
    grid = Image.new("RGB", (w * 2, h * 2)) 
    for idx, frame in enumerate(ordered_frames): 
        row, col = divmod(idx, 2) 
        grid.paste(frame, (col * w, row * h)) 

    grid.save(save_grid_path) 
    print(f"Timelapse grid saved to {save_grid_path}") 

    return ordered_frames


# Run denoising
if pipe is not None:
    try:
        prompt = "A healthy mango."
        seed = 42
        steps = 50
        capture_steps = [0, 10, 20, 30, 50]

        ordered_frames = denoising_movie(
            pipe=pipe,
            prompt=prompt,
            seed=seed, 
            steps=steps,
            capture_steps=capture_steps
        )

        # Display the timelapse grid
        grid_image = plt.imread("outputs/figures/stable_diffusion_denoising_timelapse.png")
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(grid_image)
        plt.show()
        
    except Exception as e:
        print(f"Error during denoising movie:\n{e}")
else:
    print("Cannot run denoising: pipe is None")
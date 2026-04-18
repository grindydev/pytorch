"""
Lesson 3 - Module 2: Fruit Quality Classification with Diffusion-Augmented Data
==================================================================================
WHAT YOU'LL LEARN:
  * Practical application: classifying fruit as healthy or rotten
  * Using forward hooks to capture intermediate layer activations
  * Applying pre-trained models to real-world quality control tasks
  * Generating synthetic training data with diffusion models

KEY CONCEPT:
  This is a practical application combining transfer learning (ResNet-50 for
  fruit classification) with diffusion model data augmentation. Forward hooks
  let you tap into any layer's output without modifying the model code.
"""

import torch
from torch.nn import functional as F

import gc
from pathlib import Path

from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

import helper_utils
import unittests

from pathlib import Path

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


# Root path for healthy and rotten apple, mango, and tomato images
dataset_path = Path.cwd() / "data/fruits_subset/"

# Visualize random healthy and rotten samples from the dataset
helper_utils.plot_samples_from_dataset(dataset_path)


# Load the pre-trained weights for the ResNet-50 model
fruits_model = helper_utils.load_model(Path.cwd() / "data/models/fruits_quality_model.pth", device)
# Move the model to device
fruits_model = fruits_model.to(device)

# Display the model architecture
helper_utils.print_model_architecture(fruits_model)

# Launch the prediction interface
helper_utils.predict_fruit_quality(fruits_model, dataset_path, device)

def grab(activations, name):
    """
    Creates a forward hook function to capture and store the output of a specific layer.

    Arguments:
        activations: A dictionary where the captured layer output will be stored.
        name: The key under which the output tensor will be saved in the dictionary.

    Returns:
        _hook: The closure function to be registered as a hook.
    """
    # Define the internal hook function following the PyTorch hook signature (module, input, output)
    def _hook(_, __, out): 
        # Detach the output tensor from the gradient graph and store it in the dictionary
        activations[name] = out.detach()
        
    # Return the closure to be registered as a hook
    return _hook


def cnn_feature_hierarchy(img, model):
    """
    Visualizes the feature hierarchy of a CNN by capturing feature maps 
    from specific layers during a forward pass.

    This function attaches hooks to key convolutional layers in a ResNet-style 
    architecture to extract intermediate representations.

    Arguments:
        img: The input tensor (image) to process.
        model: The pretrained neural network module to use for feature extraction.

    Returns:
        activations: A dictionary mapping layer names to their captured 
                     feature-map tensors.
    """

    # Initialize an empty dictionary to store the captured activations
    activations = {}
    
    ### START CODE HERE ###

    # Define a dictionary mapping descriptive names to the specific model layers to probe
    layers = { 
        # Register forward hook for the first convolution layer
        "conv1": model.conv1,
        # Register forward hook for the first convolution layer in the first layer of the model
        "layer1": model.layer1[0].conv1,
         # Register forward hook for the first convolution layer in the second layer of the model
        "layer2": model.layer2[0].conv1,
        # Register forward hook for the first convolution layer in the third layer of the model
        "layer3": model.layer3[0].conv1,
        # Register forward hook for the first convolution layer in the fourth layer of the model
        "layer4": model.layer4[0].conv1,
    } 

    ### END CODE HERE ###

    # Initialize a list to track the registered hook handles for cleanup
    hooks = []
    
    ### START CODE HERE ###

    # Iterate through the dictionary to register hooks on each selected layer
    for name, layer in layers.items():
        
        # Create a specific hook closure for the current layer using the 
        # `grab` helper function
        hook_function = grab(activations, name)
        
        # Register the forward hook on the layer and store the returned handle
        hook_handle = layer.register_forward_hook(hook_function)
        
        # Append the `hook_handle` to `hooks` list
        hooks.append(hook_handle)
    
    ### END CODE HERE ###

    # Perform the forward pass to trigger the hooks and capture data
    with torch.no_grad():
        _ = model(img) 

    # Remove all hooks to clean up the model and prevent memory leaks
    for h in hooks:  
        h.remove() 

    return activations


# Verify your implementation

# Load and preprocess a sample image
image_path = Path.cwd() / "data/fruits_subset/Apple_Healthy/FreshApple_3.jpg"
img = helper_utils.preprocess_image(image_path, device)

# Compute the activations
activations = cnn_feature_hierarchy(
    img=img,
    model=fruits_model
)

# Check all keys and shapes
print("Activations Keys and Shapes:\n")
for name, tensor in activations.items():
    print(f"{name}:\t{tensor.shape}")


# Test your code!
unittests.exercise_1(cnn_feature_hierarchy)

# Display the feature hierarchy for the given image
helper_utils.display_feature_hierarchy(activations, img)

# GRADED FUNCTION: feature_map_strip

def feature_map_strip(img, model):
    """
    Processes an image through a model to extract, select, and upsample 
    representative feature maps from specific layers.

    This function retrieves raw feature activations, identifies the most 
    active channel based on mean activation, and upsamples it to match 
    the original image resolution for visualization purposes.

    Arguments:
        img: The input image tensor.
        model: The pretrained neural network module used for feature extraction.

    Returns:
        upsampled: A list of tensors, each representing the most active 
                   channel from a specific layer, resized to 224x224 and 
                   normalized to [0, 1].
    """

    ### START CODE HERE ###
    
    # Capture the raw feature maps using `cnn_feature_hierarchy` (exercise 1)
    feats = cnn_feature_hierarchy(img, model)
    
    # Initialize list to store processed maps
    upsampled = [] 

    # Iterate through the specific layers to visualize
    for name in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
        
        # Extract the tensor for the current layer
        fm = feats[name]
        
        # Select the most "active" feature channel
        # Calculate the mean activation per channel (averaging over Height and Width dims)
        avg_activation = fm.mean(dim=(2, 3))
        
        # Find the index of the channel with the highest average activation
        idx = avg_activation.argmax().item()
        
        # Slice the tensor to keep only that specific channel (keep dims 4D: B, C, H, W)
        sel = fm[:, idx:idx+1] 
        
        # 4. Upsample the small feature map to the original image size (224x224)
        sel = F.interpolate(sel, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 5. Normalize values to [0, 1] for visualization
        sel = (sel - sel.min()) / (sel.max() - sel + 1e-8)
        
        # Append `sel` to list `upsampled`
        upsampled.append(sel)

    ### END CODE HERE ###

    return upsampled


# Verify your implementation

# Load and preprocess a sample image
image_path = Path.cwd() / "data/fruits_subset/Tomato_Rotten/rottenTomato_8.jpg"
img = helper_utils.preprocess_image(image_path, device)

# Extract and resize the most significant activation channel from each key layer
upsampled = feature_map_strip(
    img=img,
    model=fruits_model
)

# Verify that all extracted maps have been upsampled to the input size (224x224)
print("Shape of the upsampled feature maps:\n")
print(f"conv1:  {upsampled[0].shape}")
print(f"layer1: {upsampled[1].shape}")
print(f"layer2: {upsampled[2].shape}")
print(f"layer3: {upsampled[3].shape}")
print(f"layer4: {upsampled[4].shape}")

# Test your code!
unittests.exercise_2(feature_map_strip)

# # Display a horizontal sequence of the five processed feature maps
helper_utils.visual_strip(upsampled)


# GRADED FUNCTION: saliency_map

def saliency_map(model, image_tensor, class_idx):
    """
    Generate a saliency map for a single image and class.

    This function computes the gradients of the target class score with respect 
    to the input image pixels. The resulting map highlights which pixels usually 
    influence the model's prediction the most.

    Arguments:
        model: A trained CNN model instance; should be in evaluation mode.
        image_tensor: Input image tensor with shape (1, 3, H, W). Must be pre-processed 
                      consistently with the model's training data.
        class_idx: The integer index of the specific target class logit to explain.

    Returns:
        heatmap: A torch.Tensor 2-D saliency heat-map normalised to the 
                 range [0, 1] with shape (H, W).
    """ 

    ### START CODE HERE ###

    # Create a clone of the input tensor to avoid modifying the original data
    image_tensor = image_tensor.clone()
    # Detach the tensor from the current computation graph to start a new tracking history
    image_tensor = image_tensor.detach()
    # Enable gradient tracking for the input tensor to allow backpropagation to the pixels
    image_tensor.requires_grad_(True)

    # Perform a forward pass of the image through the model
    output = model(image_tensor)
    # Extract the logit (raw score) corresponding to the target class index
    target_logit = output[0, class_idx]

    ### END CODE HERE ###

    # Clear any existing gradients in the model parameters
    model.zero_grad()

    ### START CODE HERE ###
    
    # Perform the backward pass to compute gradients of the target logit w.r.t the input
    target_logit.backward()

    # Compute the absolute value of the gradients and sum across the color channels (C dim)
    grads = image_tensor.grad.abs().sum(dim=1)[0]

    # Normalize the gradients to the [0, 1] range for visualization
    # Subtract the minimum value to shift the range to start at 0
    grads -= grads.min()
    # Divide by the maximum value (plus a small epsilon) to scale to [0, 1]
    grads /= grads.max() + 1e-8

    ### END CODE HERE ###

    # Detach the resulting heatmap from the computation graph
    heatmap = grads.detach()

    return heatmap


# Verify your implementation

# Load and preprocess a sample image
image_path = Path.cwd() / "data/fruits_subset/Apple_Rotten/rottenApple_7.jpg"
img = helper_utils.preprocess_image(image_path, device)

# Define the target category for explanation (1 corresponds to 'rotten')
class_idx = 1

# Compute saliency map
heatmap = saliency_map(
    model=fruits_model,
    image_tensor=img,
    class_idx=class_idx
)

# Confirm the heatmap matches input dimensions and is normalized to [0, 1]
print("Shape and Range of the heatmap:\n")
print(f"Shape: {heatmap.shape}")
print(f"Range: min = {heatmap.min()}, max = {heatmap.max()}")

# Test your code!
unittests.exercise_3(saliency_map)

# Display saliency map
helper_utils.display_saliency(image_tensor=img, heatmap=heatmap)

# GRADED FUNCTION: simplified_cam

def simplified_cam(model, image_tensor, class_idx):
    """
    Generates a simplified Class Activation Map (CAM) for a specific image and class.

    This function extracts the feature maps from the final convolutional layer 
    and computes a weighted sum using the weights from the final fully connected 
    layer. The resulting map highlights regions of the image that contributed 
    most to the prediction of the target class.

    Arguments:
        model: A trained ResNet-style neural network module.
        image_tensor: The input image tensor (1, 3, H, W), normalized for the model.
        class_idx: The integer index of the target class to explain.

    Returns:
        heatmap: A 2-D tensor representing the class activation heatmap, 
                 scaled to [0, 1] with the same spatial dimensions as the input.
    """

    # Initialize an empty dictionary to store the captured feature maps
    fmap_holder = {}

    # Define a hook function to detach and store the layer output during the forward pass
    def save_fmap(_, __, output): 
        fmap_holder["feat"] = output.detach()

    ### START CODE HERE ###

    # Register the forward hook on the final convolutional layer to capture features
    hook = model.layer4[-1].conv3.register_forward_hook(save_fmap)

    ### END CODE HERE ###

    # Perform a forward pass with the image to trigger the hook
    with torch.no_grad():
        _ = model(image_tensor) 

    # Remove the hook to clean up the model and stop capturing data
    hook.remove() 

    ### START CODE HERE ###
    
    # Retrieve the captured feature maps from the dictionary
    feats = fmap_holder["feat"]
    # Extract the weight vector corresponding to the target class from the FC layer
    weight_vec = model.fc.weight[class_idx]

    # Compute the weighted sum of feature maps along the channel dimension
    # Uses Einstein summation: 'c' (channels), 'chw' (features) -> 'hw' (spatial map)
    cam = torch.einsum("c,chw->hw", weight_vec, feats.squeeze(0))

    # Apply ReLU to retain only positive contributions to the class score
    cam = F.relu(cam) 
    # Normalize the activation map values to the range [0, 1]
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    ### END CODE HERE ###

    # Retrieve the spatial dimensions (Height, Width) of the original input
    H, W = image_tensor.shape[2:]

    ### START CODE HERE ###

    # Upsample the low-resolution activation map to match the input image size
    cam_up = F.interpolate( 
        # Add batch and channel dimensions required for interpolation (1, 1, H, W)
        cam.unsqueeze(0).unsqueeze(0),
        # Specify the target output size matching the input image
        size=(H, W),
        # Use bilinear interpolation for smooth resizing
        mode="bilinear", 
        # Disable corner alignment to align the geometric centers of pixels
        align_corners=False, 
    )[0, 0] 

    ### END CODE HERE ###

    # Detach the result from the graph and move to CPU if necessary
    heatmap = cam_up.cpu().detach()

    return heatmap


# Verify your implementation

# Load and preprocess a sample image
image_path = Path.cwd() / "data/fruits_subset/Apple_Rotten/rottenApple_5.jpg"
img = helper_utils.preprocess_image(image_path, device)

# Define the target category for explanation (1 corresponds to 'rotten')
class_idx = 1

# Compute CAM
heatmap = simplified_cam(
    model=fruits_model, 
    image_tensor=img, 
    class_idx=class_idx
)

# Verify shape and range
print("Shape and Range of the CAM:\n")
print(f"Shape: {heatmap.shape}")
print(f"Range: min = {heatmap.min()}, max = {heatmap.max()}")

# Test your code!
unittests.exercise_4(simplified_cam)

# Display CAM
helper_utils.display_cam(img, heatmap)

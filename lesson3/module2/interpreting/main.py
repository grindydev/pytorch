import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms

import helper_utils
from pathlib import Path

# Getting a ball image to illustrate
ball_image = helper_utils.get_ball()
ball_tensor = torch.tensor(ball_image, dtype=torch.float32)

helper_utils.plot_image(ball_tensor)

out_channels=16
kernel_size=3 
stride=1
padding=1

# Define a 2D convolutional layer with specified parameters.
conv_layer = nn.Conv2d(
    in_channels=3, 
    out_channels=out_channels, 
    kernel_size=kernel_size, 
    stride=stride, 
    padding=padding
)

# Apply the convolutional layer to the input tensor.
output_conv_layer = conv_layer(ball_tensor)

# Print the shape of the tensor before and after convolution for comparison.
print(f"Input tensor shape: {ball_tensor.shape}")
print(f"Output tensor shape after convolution: {output_conv_layer.shape}")

# Loop over each output channel (filter)
for i in range(out_channels):
    # Determine the grid size based on total number of filters
    grid_size = int(np.ceil(np.sqrt(out_channels)))
    # Add a subplot in a grid layout for each filter
    plt.subplot(grid_size, grid_size, i + 1)
    # Detach the tensor from the computation graph, convert to numpy array for visualization
    plt.imshow(output_conv_layer[i].detach().numpy(), cmap='gray')
    # Remove axis for a cleaner look
    plt.axis('off')
    # Set the title for each filter with proper formatting
    plt.title(f'Filter {i+1}', fontsize=10, pad=10)  

# Adjust layout to prevent overlap
plt.tight_layout()  
# Display the plot with all filters
plt.show()
# Adding space
print()

# Define the parameters for the max pooling layer
pool_kernel_size = 2  # Size of the pooling window
pool_stride = 2       # Stride of the pooling window

# Create a MaxPool2d layer with specified kernel size and stride
max_pool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

# Apply the max pooling layer to the input tensor
pooled_tensor = max_pool_layer(ball_tensor)

# Print the tensor shape before and after pooling to observe the dimensionality reduction
print(f"Shape of input tensor before pooling: {ball_tensor.shape}")
print(f"Shape of tensor after pooling: {pooled_tensor.shape}\n")

# Visualize the original and pooled images side by side
plt.figure(figsize=(12, 6))  # Set figure size for clarity

# Plot original image
plt.subplot(1, 2, 1)
helper_utils.plot_image(ball_tensor, title='Original', aspect='equal')

# Plot pooled image
plt.subplot(1, 2, 2)
helper_utils.plot_image(pooled_tensor, title='Pooled', aspect='equal')

# Adjust layout to prevent overlap and display the figure
plt.tight_layout()
plt.show()


class ThreeLayerCNN(nn.Module):
    """
    A three-layer Convolutional Neural Network (CNN).

    This class defines a sequential architecture consisting of three convolutional
    blocks. Each block performs a convolution, applies a non-linear activation, 
    and reduces spatial dimensions via pooling.
    """
    def __init__(self):
        """
        Initialize the ThreeLayerCNN model.
        """
        # Initialize the parent class
        super().__init__()

        # Define the container for the sequence of layers
        self.layers = nn.ModuleList([
            # First convolutional block
            nn.Sequential(
                # Convolutional layer with 3 input channels and 16 output filters
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
                # Apply ReLU activation function to introduce non-linearity
                nn.ReLU(),
                # Max pooling to reduce spatial dimensions by a factor of 2
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            # Second convolutional block
            nn.Sequential(
                # Convolutional layer taking 16 inputs and producing 32 output filters
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                # Apply ReLU activation function
                nn.ReLU(),
                # Max pooling to reduce spatial dimensions by half
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            # Third convolutional block
            nn.Sequential(
                # Convolutional layer taking 32 inputs and producing 64 output filters
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                # Apply ReLU activation function
                nn.ReLU(),
                # Final max pooling layer for size reduction
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ])
    
    def forward(self, x):
        """
        Define the forward pass of the network.

        Args:
            x: The input tensor containing image data.

        Returns:
            The output tensor after passing through all network layers.
        """
        # Iterate through each layer in the module list
        for layer in self.layers:
            # Pass the input through the current layer
            x = layer(x)
        # Return the final processed tensor
        return x


# Dictionary to store the captured feature maps
activations = {}

def grab(name):
    """
    A function for creating forward hooks to capture intermediate activations.

    This utility generates a closure that allows for the interception of 
    layer outputs during the forward pass, using the provided name as a 
    key for storage.

    Args:
        name: The string identifier used as a key in the activations dictionary.

    Returns:
        A callable hook function suitable for registration on a neural network module.
    """
    def hook(model, input, output):
        """
        Execute the hook logic to process layer outputs.

        Args:
            model: The neural network layer or module triggering the hook.
            input: The input tensor(s) provided to the layer.
            output: The resulting output tensor(s) generated by the layer.
        """
        # Detach the output tensor from the computation graph and save it to the global dictionary
        activations[name] = output.detach()

    # Return the inner hook function to be registered
    return hook

model = ThreeLayerCNN()

# Register forward hooks on specific layers to capture activations during a forward pass
# This helps us visualize what features each layer detects
model.layers[0].register_forward_hook(grab('layer1'))  # Hook for layer1
model.layers[1].register_forward_hook(grab('layer2'))  # Hook for layer2
model.layers[2].register_forward_hook(grab('layer3'))  # Hook for layer3

# Passing the ball tensor to the network
output = model(ball_tensor.unsqueeze(0))
print(output.shape)

# Printing the shapes after each layer
for layer, output_tensor in activations.items():
    print(f"Output shape after {layer}: {output_tensor.shape}")

helper_utils.visualize_all_layers_grids(activations)

rfinfo = helper_utils.calculate_receptive_field(model, input_size=224)
helper_utils.plot_receptive_field_summary(rfinfo)

# Load pre-trained ResNet50
torch.hub.set_dir(Path.cwd() / 'data/pretrained_model')
model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2).eval()

rfinfo = helper_utils.calculate_receptive_field(model, input_size=224)
helper_utils.plot_receptive_field_summary(rfinfo)

# Reset activations dictionary
activations = {}
# To register a forward hook, you need to call the following method on each layer you want to register
model.conv1.register_forward_hook(grab('conv1'))           # First layer
model.layer1[0].conv1.register_forward_hook(grab('layer1'))  # Early block
model.layer2[0].conv1.register_forward_hook(grab('layer2'))  # Middle block
model.layer3[0].conv1.register_forward_hook(grab('layer3'))  # Later block
model.layer4[0].conv1.register_forward_hook(grab('layer4'))  # Deep block

# Load and preprocess an image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load a sample image (let's say we have a cat image)
image = Image.open(Path.cwd() / 'data/images/cat.jpg')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Forward pass through the model
with torch.no_grad():
    output = model(input_tensor)

# Visualize the input image and feature maps
plt.figure(figsize=(15, 10))

# Display original image
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

helper_utils.visualize_all_layers_grids(activations)

helper_utils.plot_rfinfo_over_image(rfinfo, Path.cwd() / 'data/images/cat.jpg', input_size=224)

file_path = Path.cwd()/ 'data/imagenet_classes.txt'
images_folder = Path.cwd()/ 'data/images'

helper_utils.plot_widget(model, file_path, images_folder)


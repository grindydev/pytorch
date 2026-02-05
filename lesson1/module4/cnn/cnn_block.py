
import torch
import torch.nn as nn


# GRADED CLASS: CNNBlock

class CNNBlock(nn.Module):
    """
    Defines a single convolutional block for a CNN.

    This block consists of a convolutional layer, batch normalization,
    a ReLU activation, and a max-pooling layer, bundled as a sequential module.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        """
        Initializes the layers of the CNNBlock.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
            padding (int, optional): Zero-padding added to both sides of the input. Defaults to 1.
        """
        # Initialize the parent nn.Module class.
        super(CNNBlock, self).__init__()
        
        ### START CODE HERE ###
        
        # Define the sequential container for the block's layers.
        self.block = nn.Sequential(
            # 2D convolutional layer to apply learnable filters to the input.
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            # Batch normalization to stabilize and accelerate training.
            nn.BatchNorm2d(num_features=out_channels),
            # ReLU activation function to introduce non-linearity.
            nn.ReLU(),
            # Max pooling layer to downsample the feature map and reduce spatial dimensions.
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) 
        
        ### END CODE HERE ###

    def forward(self, x):
        """
        Defines the forward pass for the CNNBlock.

        Args:
            x: The input tensor for the block.

        Returns:
            The output tensor after passing through the block.
        """
        
        ### START CODE HERE ###
        
        # Pass the input tensor through the sequential block of layers.
        x = self.block(x)
        return x
    
        ### END CODE HERE ###


# GRADED CLASS: SimpleCNN

class SimpleCNN(nn.Module):
    """
    Defines a simple CNN architecture using modular CNNBlocks.

    This model stacks three reusable convolutional blocks followed by a fully
    connected classifier to perform image classification.
    """
    def __init__(self, num_classes):
        """
        Initializes the layers of the SimpleCNN model.

        Args:
            num_classes (int): The number of output classes for the classifier.
        """
        # Initialize the parent nn.Module class.
        super(SimpleCNN, self).__init__()
        
        ### START CODE HERE ###

        # Define the first convolutional block.
        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)
        # Define the second convolutional block.
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)
        # Define the third convolutional block.
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)

        # Define the fully connected classifier block.
        self.classifier = nn.Sequential(
            # Flatten the 3D feature map (channels, height, width) into a 1D vector.
            nn.Flatten(start_dim=1),
            # First fully connected (linear) layer that maps the flattened features to a hidden layer.
            nn.Linear(4*4*128, 512),
            # ReLU activation function to introduce non-linearity.
            nn.ReLU(),
            # Dropout layer to prevent overfitting by randomly setting a fraction of inputs to zero.
            nn.Dropout(p=0.6),
            # Final fully connected (linear) layer that maps the hidden layer to the output classes.
            nn.Linear(512, num_classes)
        ) 
        
        ### END CODE HERE ###

    def forward(self, x):
        """
        Defines the forward pass of the SimpleCNN model.

        Args:
            x (torch.Tensor): The input tensor containing a batch of images.

        Returns:
            torch.Tensor: The output tensor with logits for each class.
        """
        
        ### START CODE HERE ###
        
        # Pass the input through the first convolutional block.
        x = self.conv_block1(x)
        # Pass the result through the second convolutional block.
        x = self.conv_block2(x)
        # Pass the result through the third convolutional block.
        x = self.conv_block3(x)

        # Pass the final feature map through the classifier.
        x = self.classifier(x)
        
        ### END CODE HERE ###
        
        # Return the final output tensor.
        return x
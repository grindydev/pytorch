import copy 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import helper_utils
import unittests
from pathlib import Path

from cnn_block import CNNBlock, SimpleCNN

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

data_path = Path.cwd() / 'data/cifar_100'

# Pre-calculated mean for each of the 3 channels of the CIFAR-100 dataset
cifar100_mean = (0.5071, 0.4867, 0.4408)
# Pre-calculated standard deviation for each of the 3 channels of the CIFAR-100 dataset
cifar100_std = (0.2675, 0.2565, 0.2761)

# GRADED FUNCTION: define_transformations

def define_transformations(mean, std):
    """
    Creates image transformation pipelines for training and validation.

    Args:
        mean (list or tuple): A sequence of mean values for each channel.
        std (list or tuple): A sequence of standard deviation values for each channel.

    Returns:
        train_transformations (torchvision.transforms.Compose): The training
                                                                transformation pipeline.
        val_transformations (torchvision.transforms.Compose): The validation
                                                                transformation pipeline.
    """
    
    ### START CODE HERE ###
    
    # Define the sequence of transformations for the training dataset.
    
    train_transformations = transforms.Compose([
        # Randomly flip the image horizontally with a 50% probability.
        transforms.RandomHorizontalFlip(p=0.5),
        # Randomly flip the image vertically with a 50% probability.
        transforms.RandomVerticalFlip(p=0.5),
        # Rotate the image by a random angle between -15 and +15 degrees.
        transforms.RandomRotation(degrees=15),
        # Convert the image from a PIL Image or NumPy array to a PyTorch tensor.
        transforms.ToTensor(),
        # Normalize the tensor image with the given mean and standard deviation.
        transforms.Normalize(mean=mean, std=std),
    ]) 
    
    # Define the sequence of transformations for the validation dataset.
    val_transformations = transforms.Compose([
        # Convert the image from a PIL Image or NumPy array to a PyTorch tensor.
        transforms.ToTensor(),
        # Normalize the tensor image with the given mean and standard deviation.
        transforms.Normalize(mean=mean, std=std),
    ]) 
    
    ### END CODE HERE ###

    # Return both transformation pipelines.
    return train_transformations, val_transformations


# Verify the Transformations
print("--- Verifying define_transformations ---\n")
train_transform_verify, val_transform_verify = define_transformations(cifar100_mean, cifar100_std)


print("Training Transformations:")
print(train_transform_verify)
print("-" * 30)
print("\nValidation Transformations:")
print(val_transform_verify)

# Test your code!
unittests.exercise_1(define_transformations)
print("-"*45)

# Create and store the training and validation transformation pipelines
train_transform, val_transform = define_transformations(cifar100_mean, cifar100_std)

# Define the full class list.
all_target_classes = [
    # Flowers
    'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
    # Mammals
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    # Insects
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'
]

# Load the full datasets.
train_dataset, val_dataset = helper_utils.load_cifar100_subset(all_target_classes, train_transform, val_transform, root=data_path)

# Set the number of samples to be processed in each batch
batch_size = 64

# Create a data loader for the training set, with shuffling enabled
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Create a data loader for the validation set, without shuffling
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# # Visualize a grid of random training images
# helper_utils.visualise_images(train_loader, grid=(3, 5))


# Verify the CNNBlock
print("--- Verifying CNNBlock ---\n")

# Instantiate the block with 3 input channels and 16 output channels
verify_cnn_block = CNNBlock(in_channels=3, out_channels=16)
print("Block Structure:\n")
print(verify_cnn_block)

# Verify the output shape after a forward pass
# Create a dummy input tensor (batch_size=1, channels=3, height=32, width=32)
dummy_input = torch.randn(1, 3, 32, 32)
print(f"\nInput tensor shape:  {dummy_input.shape}")

# Pass the dummy tensor through the block
output = verify_cnn_block(dummy_input)
print(f"Output tensor shape: {output.shape}")


# Test your code!
unittests.exercise_2(CNNBlock)


print("-"*45)
# Verify the SimpleCNN
print("--- Verifying SimpleCNN ---\n")

# Verify the structure of the model
# Instantiate the model with 15 output classes
verify_simple_cnn = SimpleCNN(num_classes=15)
print("Model Structure:\n")
print(verify_simple_cnn)

# Verify the output shape after a forward pass
# Create a dummy input tensor (batch_size=64, channels=3, height=32, width=32)
dummy_input = torch.randn(64, 3, 32, 32)
print(f"\nInput tensor shape:  {dummy_input.shape}")

# Pass the dummy tensor through the model
output = verify_simple_cnn(dummy_input)
print(f"Output tensor shape: {output.shape}")

# Test your code!
unittests.exercise_3(SimpleCNN, CNNBlock)

# Get the number of classes
num_classes = len(train_dataset.classes)

# Instantiate the model
model = SimpleCNN(num_classes)

print("-"*45)


# Loss function
loss_function = nn.CrossEntropyLoss()

# Optimizer for the model with weight_decay
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)

# GRADED FUNCTION: train_epoch

def train_epoch(model, train_loader, loss_function, optimizer, device):
    """
    Performs a single training epoch.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training data.
        loss_function (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device (CPU or GPU) to perform training on.

    Returns:
        float: The average training loss for the epoch.
    """
    # Set the model to training mode
    model.train()
    running_loss = 0.0
    # Iterate over batches of data in the training loader
    for images, labels in train_loader:
        # Move images and labels to the specified device
        images, labels = images.to(device), labels.to(device)
        
        ### START CODE HERE ###
        
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
        # Perform a forward pass to get model outputs
        outputs = model(images)
        # Calculate the loss
        loss = loss_function(outputs, labels)
        # Perform a backward pass to compute gradients
        loss.backward()
        # Update the model parameters
        optimizer.step()
        
        ### END CODE HERE ###
        
        # Accumulate the training loss for the batch
        running_loss += loss.item() * images.size(0)
        
    # Calculate and return the average training loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


# Use a helper function to perform a sanity check on the train_epoch implementation
helper_utils.verify_training_process(SimpleCNN, train_loader, loss_function, train_epoch, device)

# Test your code!
unittests.exercise_4(train_epoch)

print("-"*45)


# GRADED FUNCTION: validate_epoch

def validate_epoch(model, val_loader, loss_function, device):
    """
    Performs a single validation epoch.

    Args:
        model (torch.nn.Module): The neural network model to validate.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation data.
        loss_function (callable): The loss function.
        device (torch.device): The device (CPU or GPU) to perform validation on.

    Returns:
        tuple: A tuple containing the average validation loss and validation accuracy.
    """
    # Set the model to evaluation mode
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    
    ### START CODE HERE ###
    
    # Disable gradient calculations for validation
    with torch.no_grad():
        
    ### END CODE HERE ###
    
        # Iterate over batches of data in the validation loader
        for images, labels in val_loader:
            # Move images and labels to the specified device
            images, labels = images.to(device), labels.to(device)
            
            ### START CODE HERE ###
            
            # Perform a forward pass to get model outputs
            outputs = model(images)
            
            # Calculate the validation loss for the batch
            val_loss = loss_function(outputs, labels)
            # Accumulate the validation loss
            running_val_loss += val_loss.item() * images.size(0)
            
            # Get the predicted class labels
            _, predicted = torch.max(outputs, 1)
            
            ### END CODE HERE ###
            
            # Update the total number of samples
            total += labels.size(0)
            # Update the number of correct predictions
            correct += (predicted == labels).sum().item()
            
    # Calculate the average validation loss and accuracy for the epoch
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_accuracy = 100.0 * correct / total
    
    return epoch_val_loss, epoch_accuracy


# Use a helper function to perform a sanity check on the validate_epoch implementation
helper_utils.verify_validation_process(SimpleCNN, val_loader, loss_function, validate_epoch, device)

# Test your code!
unittests.exercise_5(validate_epoch)

print("-"*45)

def training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    """
    Trains and validates a PyTorch neural network model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        loss_function (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        num_epochs (int): The total number of epochs to train for.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run training on.

    Returns:
        tuple: A tuple containing the best trained model and a list of metrics
               (train_losses, val_losses, val_accuracies).
    """
    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    
    # Initialize variables to track the best performing model
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Initialize lists to store training and validation metrics
    train_losses, val_losses, val_accuracies = [], [], []
    
    print("--- Training Started ---")
    
    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        # Perform one epoch of training
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        train_losses.append(epoch_loss)
        
        # Perform one epoch of validation
        epoch_val_loss, epoch_accuracy = validate_epoch(model, val_loader, loss_function, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)
        
        # Print the metrics for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")
        
        # Check if the current model is the best one so far
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            # Save the state of the best model in memory
            best_model_state = copy.deepcopy(model.state_dict())
            
    print("--- Finished Training ---")
    
    # Load the best model weights before returning
    if best_model_state:
        print(f"\n--- Returning best model with {best_val_accuracy:.2f}% validation accuracy, achieved at epoch {best_epoch} ---")
        model.load_state_dict(best_model_state)
    
    # Consolidate all metrics into a single list
    metrics = [train_losses, val_losses, val_accuracies]
    
    # Return the trained model and the collected metrics
    return model, metrics


# Start the training process by calling the training loop function
trained_model, training_metrics = training_loop(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    loss_function=loss_function, 
    optimizer=optimizer, 
    num_epochs=50, 
    device=device
)

# Visualize the training metrics (loss and accuracy)
print("\n--- Training Plots ---\n")
helper_utils.plot_training_metrics(training_metrics)



print("-"*45)


# Import the preview function that demonstrates concepts from the next course
from c2_preview.c2_preview import course_2_preview

# This helper function runs a training loop using a powerful strategy that will be taught
# in the next course. Run this cell to see the improved results in action.
trained_model = course_2_preview(
    train_dataset, 
    val_dataset, 
    loss_function,
    device,
    num_epochs=5
    )
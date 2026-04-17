# 1. **Unstructured Pruning**:
#     * **Best Practices**: Unstructured pruning is often used when you want to maximize sparsity while maintaining flexibility. It is best suited for models where the architecture should remain unchanged, and you only need to prune less important weights.
#     * **Use Cases**: This technique is useful in scenarios where hardware support for sparsity is present, allowing the sparsely pruned model to be efficiently executed. It's suitable for environments like server-side deployments where computational resources are relatively abundant but memory savings are crucial.
# 2. **Structured Pruning**:
#     * **Best Practices**: Use structured pruning when you need to improve inference speed and reduce the model's memory footprint effectively. This technique can significantly accelerate the model since entire structures like channels or neurons are removed.
#     * **Use Cases**: It's ideal for edge devices or mobile applications where computational resources are limited, as structured pruning can help maintain a balance between model size and accuracy while enhancing computational efficiency.
# 3. **Global Pruning**:
#     * **Best Practices**: Global pruning is advantageous when aiming for a consistent level of sparsity across multiple layers. It minimizes the risk of over-pruning critical parts of the model by evaluating the importance of weights globally rather than locally.
#     * **Use Cases**: This approach is beneficial in situations where the overall model performance is more critical than the performance of individual layers. It's particularly effective for creating uniformly sparse models that can adapt to various processing environments without heavily compromising accuracy.

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import helper_utils

# Create a simple model for demonstration
class SimpleModel(nn.Module):
    """
    A simple convolutional neural network defined for demonstration purposes.
    """
    def __init__(self):
        """
        Initializes the SimpleModel with a convolutional layer and a fully connected layer.
        """
        # Call the parent class initializer
        super(SimpleModel, self).__init__()
        # Define the first convolutional layer with 3 input channels and 16 output channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        # Define the ReLU activation function
        self.relu1 = nn.ReLU()
        # Define the fully connected layer
        # The input size is calculated based on a 6x6 input image: (6-3+1) = 4x4 spatial output
        self.fc1 = nn.Linear(16 * 4 * 4, 10)  # Assuming input is 6x6

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: The input tensor containing the image data.

        Returns:
            The output tensor after passing through the network.
        """
        # Apply the convolutional layer followed by the ReLU activation
        x = self.relu1(self.conv1(x))
        # Flatten the tensor dimensions for the fully connected layer
        x = x.view(x.size(0), -1)
        # Pass the flattened tensor through the fully connected layer
        x = self.fc1(x)
        return x

model = SimpleModel()

# Check state before pruning
print("BEFORE UNSTRUCTURED PRUNING:\n")

# Model Parameters
print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
# Model Buffers
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")


print("--- conv1 Weights Before Pruning ---\n")
helper_utils.show_weights(model, ['conv1'])

# Apply unstructured pruning to conv1
prune.l1_unstructured(model.conv1, name="weight", amount=0.3)

# Check state after pruning
print("AFTER UNSTRUCTURED PRUNING:\n")
print(f"Model parameters:", [name for name, _ in model.named_parameters()])
print(f"Model buffers:", [name for name, _ in model.named_buffers()])

# --- Show numerical weights after pruning ---
print("\n--- conv1 Weights After Pruning ---\n")
helper_utils.show_weights(model, ['conv1'])

# Check if the pruning mask exists.
if hasattr(model.conv1, 'weight_mask'):
    
    # Count weights where the mask is 0.
    pruned_count = torch.sum(model.conv1.weight_mask == 0).item()
    
    # Get the total number of weights.
    total_count = model.conv1.weight_mask.numel()
    
    # Print statistics.
    print(f"\nPruning statistics:")
    print(f"Total weights in conv1: {total_count}")
    print(f"Pruned weights (set to zero) in conv1: {pruned_count}")
    print(f"Sparsity: {pruned_count/total_count:.2%}")

    # Verify the change in weight attribute types.
    print(f"\nWeight type: {type(model.conv1.weight)}")
    print(f"Weight_orig type: {type(model.conv1.weight_orig)}")
else:
    # Handle pruning failure.
    print("Pruning did not work as expected! No weight_mask found.")


# Make pruning permanent for conv1
prune.remove(model.conv1, 'weight')

# Check state to see if pruning is permanent
print("AFTER MAKING UNSTRUCTURED PRUNING PERMANENT:\n")

print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")
print(f"\nIs conv1 still considered pruned? {prune.is_pruned(model.conv1)}\n")

helper_utils.show_weights(model, ['conv1'])


print('\n' + '='*50 + '\n')

model = SimpleModel()

# Check state before pruning
print("BEFORE STRUCTURED PRUNING:\n")
# Model State
print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")

print("\n--- fc1 Weights Before Pruning ---\n")
helper_utils.show_weights(model, ['fc1'])

# Apply structured pruning to fc1
prune.ln_structured(model.fc1, name="weight", amount=0.5, n=2, dim=0)

# Check state after pruning
print("AFTER STRUCTURED PRUNING:\n")
print(f"Model parameters:", [name for name, _ in model.named_parameters()])
print(f"Model buffers:", [name for name, _ in model.named_buffers()])

# --- Show numerical weights after pruning ---
print("\n--- fc1 Weights After Pruning ---\n")
helper_utils.show_weights(model, ['fc1'])


# Check if the pruning mask exists for fc1.
if hasattr(model.fc1, 'weight_mask'):
    # For structured pruning, you count entire rows of zeros.
    # Determine if each row in the weight tensor is all zeros.
    zero_rows = torch.sum(model.fc1.weight == 0, dim=1) == model.fc1.weight.shape[1]
    # Count the number of all-zero rows.
    pruned_count = torch.sum(zero_rows).item()
    # Get the total number of rows (output neurons).
    total_count = model.fc1.weight.shape[0]
    
    # Print the statistics.
    print(f"\nPruning statistics:\n")
    print(f"Total output neurons in fc1: {total_count}")
    print(f"Pruned output neurons (set to zero) in fc1: {pruned_count}")
    print(f"Sparsity: {pruned_count/total_count:.2%}")
    
    # Verify the change in attribute types.
    print(f"\nWeight type: {type(model.fc1.weight)}")
    print(f"Weight_orig type: {type(model.fc1.weight_orig)}")
else:
    # Handle pruning failure.
    print("\nStructured pruning did not work as expected!")

# Make structured pruning permanent for fc1
prune.remove(model.fc1, 'weight')

# Check if structured pruning is permanent
print("AFTER MAKING STRUCTURED PRUNING PERMANENT:\n")
print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")
print(f"\nIs fc1 still considered pruned? {prune.is_pruned(model.fc1)}\n")

helper_utils.show_weights(model, ['fc1'])

print('\n' + '='*50 + '\n')

model = SimpleModel()

# Check the model's initial state before pruning.
print("BEFORE GLOBAL PRUNING:\n")

print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")

# Show the initial weights of both layers.
print("\n--- Initial Weights of conv1 and fc1 ---\n")
helper_utils.show_weights(model, ['conv1', 'fc1'])

# Define the collection of parameters to be pruned globally.
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.fc1, 'weight'),
)

# Apply global unstructured pruning across the specified parameters.
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

# Verify the model's state after pruning.
print("AFTER GLOBAL PRUNING:\n")

print(f"Model parameters:", [name for name, _ in model.named_parameters()])
print(f"Model buffers:", [name for name, _ in model.named_buffers()])

print("\n--- Weights After Global Pruning ---\n")
helper_utils.show_weights(model, ['conv1', 'fc1'])

# Check global pruning results.
print("--- Sparsity per Layer After Global Pruning ---\n")

# Verify sparsity in each layer individually.
conv1_sparsity = 100. * float(torch.sum(model.conv1.weight == 0)) / float(model.conv1.weight.nelement())
fc1_sparsity = 100. * float(torch.sum(model.fc1.weight == 0)) / float(model.fc1.weight.nelement())

print(f"Sparsity in conv1: {conv1_sparsity:.2f}%")
print(f"Sparsity in fc1: {fc1_sparsity:.2f}%")

# Make the pruning permanent for all pruned layers by iterating through the list.
for module, param_name in parameters_to_prune:
    prune.remove(module, param_name)

print("\nAFTER MAKING GLOBAL PRUNING PERMANENT:\n")

print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")

print(f"\nIs conv1 still considered pruned? {prune.is_pruned(model.conv1)}")
print(f"Is fc1 still considered pruned? {prune.is_pruned(model.fc1)}\n")

helper_utils.show_weights(model, ['conv1', 'fc1'])




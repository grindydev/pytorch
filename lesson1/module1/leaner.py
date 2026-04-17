"""
Lesson 1 - Module 1: Linear Regression with PyTorch (leaner.py)
================================================================
 WHAT YOU'LL LEARN:
  • How to define a simple neural network (a single linear layer)
  • The 4-step training loop: zero_grad → forward → loss → backward → step
  • What loss functions and optimizers do
  • How to make predictions with a trained model
  • How to inspect learned weights and bias

 KEY CONCEPT:
  **Linear Regression** is the simplest neural network: y = weight * x + bias.
  The model learns the relationship between input (distance) and output (time)
  by iteratively adjusting weight and bias to minimize prediction error (loss).

 PROBLEM:
  Predict delivery time (minutes) given distance (miles).
  Short distances use bikes → roughly linear relationship: time ≈ w * distance + b
"""

import torch
import torch.nn as nn       # Neural network building blocks (layers, loss functions)
import torch.optim as optim  # Optimization algorithms (SGD, Adam, etc.)

import helper_utils

# ==================== STEP 1: SET RANDOM SEED ====================
#  CONCEPT: Setting a seed ensures reproducibility — you get the same
#   random weight initialization every time you run this script.
torch.manual_seed(42)


# ==================== STEP 2: PREPARE THE DATA ====================
# Input: distances in miles (shape: [4, 1] — 4 samples, 1 feature each)
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

# Target: delivery times in minutes (shape: [4, 1] — must match input shape)
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)


# ==================== STEP 3: DEFINE THE MODEL ====================
# nn.Sequential chains layers in order. Here we have just one:
#   nn.Linear(1, 1) → y = w * x + b  (1 input feature → 1 output)
#
#  CONCEPT: nn.Linear applies an affine transformation:
#   output = weight @ input + bias
#   weight is initialized randomly, bias is initialized to zero.
model = nn.Sequential(nn.Linear(1, 1))


# ==================== STEP 4: DEFINE LOSS FUNCTION & OPTIMIZER ====================
#  CONCEPT: Loss Function measures how wrong the model's predictions are.
#   - MSELoss (Mean Squared Error): averages the squared differences
#     between predictions and targets. Lower = better.
loss_function = nn.MSELoss()

#  CONCEPT: Optimizer updates the model's parameters (weights & bias)
#   to minimize the loss.
#   - SGD (Stochastic Gradient Descent): updates parameters in the direction
#     that reduces loss, scaled by the learning rate (lr).
#   - lr=0.01 means each step changes weights by 1% of the gradient.
optimizer = optim.SGD(model.parameters(), lr=0.01)


# ==================== STEP 5: TRAINING LOOP ====================
#  THE CORE OF ALL DEEP LEARNING — this 4-step cycle repeats for every epoch:
#   1. zero_grad()  — Clear old gradients from the previous step
#   2. forward()    — Make predictions using current weights
#   3. loss         — Compare predictions to actual values
#   4. backward()   — Calculate gradients (how much each weight contributed to error)
#   5. step()       — Update weights using the gradients

NUM_EPOCHS = 500  # Number of times to iterate over the entire dataset

for epoch in range(NUM_EPOCHS):
    # Step 1: Clear accumulated gradients from the previous iteration
    #  WHY: Gradients accumulate by default in PyTorch. If you don't zero them,
    #   they'd be summed across iterations, giving wrong updates.
    optimizer.zero_grad()

    # Step 2: Forward pass — the model predicts outputs from inputs
    #   Internally: outputs = weight * distances + bias
    outputs = model(distances)

    # Step 3: Calculate loss — how far off are the predictions?
    loss = loss_function(outputs, times)

    # Step 4: Backward pass — compute gradients for all parameters
    #  CONCEPT: Backpropagation calculates ∂loss/∂weight and ∂loss/∂bias.
    #   These tell us the direction and magnitude to adjust each parameter.
    loss.backward()

    # Step 5: Update parameters — nudge weights in the direction that reduces loss
    optimizer.step()

    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")


# ==================== STEP 6: VISUALIZE THE RESULTS ====================
# Plot the training data (orange dots) and the model's learned line (green)
helper_utils.plot_results(model, distances, times)


# ==================== STEP 7: MAKE A PREDICTION ====================
#  CONCEPT: torch.no_grad() tells PyTorch "we're not training, don't track gradients."
#   This saves memory and speeds up computation during inference.
distance_to_predict = 7.0

with torch.no_grad():
    # The model expects a 2D tensor: shape [1, 1] (1 sample, 1 feature)
    new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)

    # Forward pass on the new data point
    predicted_time = model(new_distance)

    # .item() extracts the Python float from the 1-element tensor
    print(f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time.item():.1f} minutes")

    # Decision based on the prediction
    if predicted_time.item() > 30:
        print("\nDecision: Do NOT take the job. You will likely be late.")
    else:
        print("\nDecision: Take the job. You can make it!")


# ==================== STEP 8: INSPECT LEARNED PARAMETERS ====================
# The model learned weight (w) and bias (b) such that: time ≈ w * distance + b
layer = model[0]  # Access the first (and only) layer in the Sequential model

weights = layer.weight.data.numpy()  # Shape: [1, 1] — one weight
bias = layer.bias.data.numpy()       # Shape: [1] — one bias

print(f"\nLearned Weight (slope): {weights}")
print(f"Learned Bias (intercept): {bias}")


# ==================== STEP 9: TEST ON NON-LINEAR DATA ====================
#  KEY INSIGHT: The linear model works well for bike deliveries (short distances).
#   But what if we add car deliveries (longer distances)? The relationship
#   becomes non-linear — a straight line CAN'T fit this curved data well.

new_distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

new_times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=torch.float32)

# Use the already-trained linear model to predict
with torch.no_grad():
    predictions = model(new_distances)

# The loss will be MUCH higher — linear model can't capture the curve
new_loss = loss_function(predictions, new_times)
print(f"\nLoss on new, combined data: {new_loss.item():.2f}")

#  LOOK AT THE PLOT: The green line is straight, but the orange dots curve.
#   This motivates the next file: non_leaner.py, which uses a hidden layer + ReLU.
helper_utils.plot_nonlinear_comparison(model, new_distances, new_times)

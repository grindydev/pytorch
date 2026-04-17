"""
Lesson 1 - Module 1: Non-Linear Regression with PyTorch (non_leaner.py)
=======================================================================
 WHAT YOU'LL LEARN:
  • Why linear models fail on non-linear data (motivation for deep learning)
  • Data normalization — why and how to standardize input/output
  • Building a model with hidden layers and ReLU activation
  • How non-linear activations let neural networks learn curves
  • De-normalizing predictions back to the original scale

 KEY CONCEPT:
  A **hidden layer + non-linear activation (ReLU)** gives the model the ability
  to learn non-linear relationships. Without non-linear activations, stacking
  linear layers would still only produce a linear function.

 PROBLEM:
  Predict delivery time for a mix of bike (short) and car (long) deliveries.
  The time-distance curve bends — bikes are slow, cars plateau around 90 min.
  y = ReLU(w₁·x + b₁)·w₂ + b₂  can approximate this curve.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import helper_utils


# ==================== STEP 1: PREPARE THE DATA ====================
# Combined dataset: bikes for short distances, cars for longer ones.
#  OBSERVE: Time grows quickly at first (bike speed) then levels off (car speed).
#   This is NON-LINEAR — a straight line can't fit it well.

distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=torch.float32)


# ==================== STEP 2: NORMALIZE THE DATA ====================
#  CONCEPT: Normalization (also called standardization or z-score scaling)
#   transforms data to have mean≈0 and std≈1:
#     normalized = (value - mean) / std
#
#  WHY NORMALIZE?
#   1. Features with large ranges dominate gradient updates
#   2. The optimizer converges MUCH faster with normalized data
#   3. Prevents numerical instability (very large or very small values)

distances_mean = distances.mean()  # Average distance across all samples
distances_std = distances.std()    # How spread out the distances are

times_mean = times.mean()          # Average delivery time
times_std = times.std()            # How spread out the times are

# Apply z-score normalization: (x - μ) / σ
distances_norm = (distances - distances_mean) / distances_std
times_norm = (times - times_mean) / times_std

# Plot the normalized data (values now centered around 0)
helper_utils.plot_data(distances_norm, times_norm, normalize=True)


# ==================== STEP 3: BUILD A NON-LINEAR MODEL ====================
#  CONCEPT: This is a 2-layer neural network:
#   Input (1) → Hidden Layer (11 neurons) → ReLU → Output Layer (1)
#
#   - nn.Linear(1, 11): 1 input feature → 11 hidden neurons (each learns a pattern)
#   - nn.ReLU(): Rectified Linear Unit — max(0, x). Introduces non-linearity.
#     Without this, two linear layers would collapse into one linear transformation.
#   - nn.Linear(11, 1): 11 hidden features → 1 output (predicted time)
#
#  WHY ReLU? It's simple (fast), avoids vanishing gradients, and works
#   well in practice. f(x) = max(0, x) → negatives become 0, positives stay.

torch.manual_seed(27)  # Seed for reproducibility

model = nn.Sequential(
    nn.Linear(1, 11),   # Hidden layer: 1 input → 11 neurons
    nn.ReLU(),           #  Non-linear activation — THE KEY to learning curves!
    nn.Linear(11, 1)     # Output layer: 11 neurons → 1 prediction
)


# ==================== STEP 4: TRAINING LOOP ====================
# Same 4-step training loop as leaner.py, but now with a non-linear model.

loss_function = nn.MSELoss()                   # Mean Squared Error
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

NUM_EPOCHS = 10000  # More epochs needed because the problem is harder

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()             # Step 1: Clear old gradients
    outputs = model(distances_norm)   # Step 2: Forward pass (on normalized data)
    loss = loss_function(outputs, times_norm)  # Step 3: Compute loss
    loss.backward()                   # Step 4: Backpropagation
    optimizer.step()                  # Step 5: Update weights

    # Visualize training progress every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        helper_utils.plot_training_progress(
            epoch=epoch,
            loss=loss,
            model=model,
            distances_norm=distances_norm,
            times_norm=times_norm
        )

print("\nTraining Complete.")
print(f"\nFinal Loss: {loss.item()}")


# ==================== STEP 5: VISUALIZE THE FINAL FIT ====================
# Plot the model's predictions against the ORIGINAL (de-normalized) data.
# The model was trained on normalized values, so we de-normalize for display:
#   actual_value = (normalized_value * std) + mean
helper_utils.plot_final_fit(model, distances, times, distances_norm, times_std, times_mean)


# ==================== STEP 6: MAKE A PREDICTION ====================
#  IMPORTANT: When using a model trained on normalized data, you MUST:
#   1. Normalize the input before feeding it to the model
#   2. De-normalize the output to get the actual prediction

distance_to_predict = 5.1

with torch.no_grad():
    # Step 1: Create the input tensor
    distance_tensor = torch.tensor([[distance_to_predict]], dtype=torch.float32)

    # Step 2: Normalize the input using the SAME mean and std from training
    new_distance_norm = (distance_tensor - distances_mean) / distances_std

    # Step 3: Get the normalized prediction from the model
    predicted_time_norm = model(new_distance_norm)

    # Step 4: De-normalize the output back to real minutes
    predicted_time_actual = (predicted_time_norm * times_std) + times_mean

    print(f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time_actual.item():.1f} minutes")

    # Decision logic: check if delivery is feasible within 45 minutes
    if predicted_time_actual.item() > 45:
        print("\nDecision: Do NOT promise the delivery in under 45 minutes.")
    else:
        # Decide vehicle based on distance (bikes for short, cars for long)
        if distance_to_predict <= 3:
            print(f"\nDecision: Yes, delivery is possible. Since the distance is {distance_to_predict} miles (<= 3 miles), use a bike.")
        else:
            print(f"\nDecision: Yes, delivery is possible. Since the distance is {distance_to_predict} miles (> 3 miles), use a car.")

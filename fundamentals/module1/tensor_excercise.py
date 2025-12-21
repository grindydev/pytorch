import torch
import numpy as np
import pandas as pd

### Exercise 1: Analyzing Monthly Sales Data
# Sales data for 3 products over 4 months (Jan, Feb, Mar, Apr).
sales_data = torch.tensor([[100, 120, 130, 110],   # Product A
                           [ 90,  95, 105, 125],   # Product B
                           [140, 115, 120, 150]    # Product C
                          ], dtype=torch.float32)

print("ORIGINAL SALES DATA:\n\n", sales_data)
print("-" * 45)

### START CODE HERE ###

# 1. Calculate total sales for Product B.
total_sales_product_b = sales_data[1].sum()

# 2. Find months where sales for Product C were > 130.
high_sales_mask_product_c = sales_data[2] > 130

# 3. Get sales for Feb and Mar for all products.
sales_feb_mar = sales_data[:, 1:3]

### END CODE HERE ###

print("\nTotal Sales for Product B:                   ", total_sales_product_b)
print("\nMonths with >130 Sales for Product C (Mask): ", high_sales_mask_product_c)
print("\nSales for Feb & Mar:\n\n", sales_feb_mar)


### Exercise 2: Image Batch Transformation
print("-" * 45)
# A batch of 4 grayscale images, each 3x3
image_batch = torch.rand(4, 3, 3)

print("ORIGINAL BATCH SHAPE:", image_batch.shape)

### START CODE HERE ###

# 1. Add a channel dimension at index 1.
image_batch_with_channel = image_batch.unsqueeze(1)

# 2. Transpose the tensor to move the channel dimension to the end.
# Swap dimension 1 (channels) with dimension 3 (the last one).
image_batch_transposed = image_batch_with_channel.transpose(1, 3)

### END CODE HERE ###
print("\nSHAPE AFTER UNSQUEEZE:", image_batch_with_channel.shape)
print("SHAPE AFTER TRANSPOSE:", image_batch_transposed.shape)


### Exercise 3: Combining and Weighting Sensor Data
print("-" * 45)
# Sensor readings (5 time steps)
temperature = torch.tensor([22.5, 23.1, 21.9, 22.8, 23.5])
humidity = torch.tensor([55.2, 56.4, 54.8, 57.1, 56.8])

print("TEMPERATURE DATA: ", temperature)
print("HUMIDITY DATA:    ", humidity)

### START CODE HERE ###

# 1. Concatenate the two tensors.
# Note: You need to unsqueeze them first to stack them vertically.
temperature = temperature.unsqueeze(0)
humidity = humidity.unsqueeze(0)

combined_data =  torch.cat((temperature, humidity), 0)

# 2. Create the weights tensor.
weights = torch.tensor([0.6, 0.4]).reshape(2, 1)

# 3. Apply weights using broadcasting.
# You need to reshape weights to [2, 1] to broadcast across columns.
weighted_data = combined_data * weights

# 4. Calculate the weighted average for each time step.
#    (A true average = weighted sum / sum of weights)
weighted_sum = torch.sum(weighted_data, dim=0)
weighted_average = weighted_sum / torch.sum(weights)

### END CODE HERE ###
print("\nCOMBINED DATA (2x5):\n\n", combined_data)
print("\nWEIGHTED DATA:\n\n", weighted_data)
print("\nWEIGHTED AVERAGE:", weighted_average)


## Excercise 4: 
print("-" * 55)
# Data for 8 taxi trips: [distance, hour_of_day]
trip_data = torch.tensor([
    [5.3, 7],   # Not rush hour, not long
    [12.1, 9],  # Morning rush, long trip -> RUSH HOUR LONG
    [15.5, 13], # Not rush hour, long trip
    [6.7, 18],  # Evening rush, not long
    [2.4, 20],  # Not rush hour, not long
    [11.8, 17], # Evening rush, long trip -> RUSH HOUR LONG
    [9.0, 9],   # Morning rush, not long
    [14.2, 8]   # Morning rush, long trip -> RUSH HOUR LONG
], dtype=torch.float32)

print("ORIGINAL TRIP DATA (Distance, Hour):\n\n", trip_data)

### START CODE HERE ###

# 1. Slice the main tensor to get 1D tensors for each feature.
distances = trip_data[:, 0]
hours = trip_data[:, 1]

# 2. Create boolean masks for each condition.
is_long_trip = distances > 10
is_morning_rush = (hours >= 8) & (hours < 10)
is_evening_rush =  (hours >= 17) & (hours < 19)

# 3. Combine masks to identify rush hour long trips.
# A trip is a rush hour long trip if it's (a morning OR evening rush) AND a long trip.
is_rush_hour_long_trip_mask = is_long_trip & (is_morning_rush | is_evening_rush)

# 4. Reshape the new feature into a column vector and cast to float.
new_feature_col = is_rush_hour_long_trip_mask.float().unsqueeze(1)

### END CODE HERE ###

print("\n'IS RUSH HOUR LONG TRIP' MASK: ", is_rush_hour_long_trip_mask)
print("\nNEW FEATURE COLUMN (Reshaped):\n\n", new_feature_col)

# You can now concatenate this new feature to the original data
enhanced_trip_data = torch.cat((trip_data, new_feature_col), dim=1)
print("\nENHANCED DATA (with new feature at the end):\n\n", enhanced_trip_data)
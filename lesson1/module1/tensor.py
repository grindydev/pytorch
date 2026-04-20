"""
Lesson 1 - Module 1: PyTorch Tensors
=====================================

WHY THIS MATTERS:
  Every neural network, from a simple linear regression to GPT-4, operates on
  tensors. If you do not understand tensors, you cannot understand anything else
  in deep learning. This is the alphabet of ML -- learn it well and everything
  else becomes readable.

  In practice: when you load an image, it becomes a tensor. When you train a
  model, the weights are tensors. When you make a prediction, the output is a
  tensor. Tensors are everywhere.

WHAT YOU'LL LEARN:
  * Creating tensors from Python lists, NumPy arrays, and Pandas DataFrames
  * Shape manipulation: unsqueeze, squeeze, reshape, transpose, concatenate
  * Indexing and slicing: accessing elements, rows, columns, masks
  * Math operations: element-wise, matrix multiplication, broadcasting
  * Comparison and logical operators on tensors
  * Basic statistics: mean, std, and data type casting

KEY CONCEPTS:
  Tensor       -- A multi-dimensional array, like a NumPy array but GPU-compatible
  Shape        -- The dimensions of a tensor (e.g., [3, 4] = 3 rows, 4 columns)
  dtype        -- Data type of elements (float32, int64, etc.)
  Broadcasting -- PyTorch auto-stretches smaller tensors to match larger ones
  Gradient     -- Tensor property tracking derivatives for backpropagation

HOW IT FITS:
  Read this file FIRST in Module 1. Then practice with tensor_excercise.py.
  After that, move to leaner.py where tensors become inputs to a real model.

PREREQUISITES:
  Basic Python. No ML knowledge needed.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path


# ==================== PART 1: CREATING TENSORS ====================
#
# CONCEPT:
#   A tensor is a container for numbers arranged in N dimensions:
#     0D = scalar (single number)
#     1D = vector (list of numbers)
#     2D = matrix (table of numbers)
#     3D = cube of numbers (e.g., color image: channels x height x width)
#     4D = batch of cubes (e.g., batch of images)
#
# WHY:
#   Neural networks process numbers, not text or images. Before any ML, you
#   must convert your raw data into tensors.
#
# ANALOGY:
#   Think of a tensor as a spreadsheet. 1D = one column, 2D = full table,
#   3D = a stack of tables, 4D = a filing cabinet of stacks.
#
# COMMON MISTAKES:
#   - Forgetting dtype: torch.tensor([1,2,3]) gives int64, not float32.
#     Models usually need float32.
#   - Using torch.tensor() when you want to share memory with NumPy.
#     torch.tensor() always COPIES data. torch.from_numpy() SHARES memory.

# --- From a Python list ---
x = torch.tensor([1, 2, 3])
print("FROM PYTHON LISTS:", x)
print("TENSOR DATA TYPE:", x.dtype)  # Default: torch.int64 for integers

# --- From a NumPy array ---
# torch.from_numpy() shares memory -- changing one changes the other!
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
torch_tensor_from_numpy = torch.from_numpy(numpy_array)
print("TENSOR FROM NUMPY:\n\n", torch_tensor_from_numpy)

# --- From a Pandas DataFrame ---
# Step 1: Read CSV
df = pd.read_csv(Path.cwd() / 'module1/data.csv')
# Step 2: Extract values as NumPy array
all_values = df.values
# Step 3: Convert to tensor (this copies data)
tensor_from_df = torch.tensor(all_values)
print("ORIGINAL DATAFRAME:\n\n", df)
print("\nRESULTING TENSOR:\n\n", tensor_from_df)
print("\nTENSOR DATA TYPE:", tensor_from_df.dtype)

# --- Factory functions: zeros, ones, arange ---
zeros = torch.zeros(2, 3)           # 2x3 matrix of 0s
ones = torch.ones(2, 3)             # 2x3 matrix of 1s
range_tensor = torch.arange(0, 10, step=1)  # Like Python's range()
print("TENSOR WITH ZEROS:\n\n", zeros)
print("TENSOR WITH ONES:\n\n", ones)
print("ARANGE TENSOR:", range_tensor)


# ==================== PART 2: SHAPE MANIPULATION ====================
#
# CONCEPT:
#   Shape tells you the size of each dimension. For a 2D tensor, shape
#   is [rows, columns]. Reshaping changes how the same data is arranged
#   without changing the data itself.
#
# WHY:
#   Shape mismatches are the #1 source of bugs in deep learning. Model
#   layers expect specific input shapes. Understanding shape operations
#   is essential for debugging.
#
# COMMON MISTAKES:
#   - Trying to reshape to incompatible sizes (6 elements cannot become 2x4)
#   - Confusing reshape (any shape with same total) with transpose (swap dims)

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print("ORIGINAL TENSOR:\n\n", x)
print("TENSOR SHAPE:", x.shape)  # torch.Size([2, 3])

# --- unsqueeze(): Add a dimension of size 1 ---
# USE CASE: Adding a "batch" dimension. Models expect [batch_size, features].
#   Shape: [2, 3] -> [1, 2, 3]
expanded = x.unsqueeze(0)
print("\nTENSOR WITH ADDED DIMENSION AT INDEX 0:\n\n", expanded)
print("TENSOR SHAPE:", expanded.shape)

# --- squeeze(): Remove dimensions of size 1 ---
#   Shape: [1, 2, 3] -> [2, 3]
squeezed = expanded.squeeze()
print("\nTENSOR WITH DIMENSION REMOVED:\n\n", squeezed)
print("TENSOR SHAPE:", squeezed.shape)

# --- reshape(): Change shape, keep all data ---
#   Total elements must stay the same: 2*3=6 = 3*2=6
reshaped = x.reshape(3, 2)
print("\nAFTER PERFORMING reshape(3, 2):\n\n", reshaped)
print("TENSOR SHAPE:", reshaped.shape)

# --- transpose(): Swap two dimensions ---
#   Rows become columns, columns become rows
transposed = x.transpose(0, 1)
print("\nAFTER PERFORMING transpose(0, 1):\n\n", transposed)
print("TENSOR SHAPE:", transposed.shape)

# --- cat(): Concatenate tensors along an existing dimension ---
tensor_a = torch.tensor([[1, 2],
                         [3, 4]])
tensor_b = torch.tensor([[5, 6],
                         [7, 8]])
# dim=1 = side by side (horizontally). dim=0 = stacked (vertically).
concatenated_tensors = torch.cat((tensor_a, tensor_b), dim=1)
print("TENSOR A:\n\n", tensor_a)
print("\nTENSOR B:\n\n", tensor_b)
print("\nCONCATENATED TENSOR (dim=1):\n\n", concatenated_tensors)


# ==================== PART 3: INDEXING AND SLICING ====================
#
# CONCEPT:
#   Indexing = picking specific elements. Slicing = picking a range.
#   Very similar to NumPy and Python lists.
#
# WHY:
#   You constantly need to access specific parts of tensors: a single
#   prediction, a batch of labels, a subset of features.
#
# CODE PATTERN:
#   tensor[row, col]          -- single element
#   tensor[start:stop]        -- range (stop is exclusive)
#   tensor[:, col]            -- all rows, one column
#   tensor[mask]              -- elements where mask is True

x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print("ORIGINAL TENSOR:\n\n", x)

# --- Single element ---
single_element_tensor = x[1, 2]  # Row 1, Column 2 -> value 7
print("\nINDEXING SINGLE ELEMENT AT [1, 2]:", single_element_tensor)

# --- Row indexing ---
second_row = x[1]     # Entire row at index 1
print("\nINDEXING ENTIRE ROW [1]:", second_row)
last_row = x[-1]      # Last row (negative indexing works too)
print("\nINDEXING ENTIRE LAST ROW ([-1]):", last_row)

# --- Slicing ---
first_two_rows = x[0:2]    # Rows 0 and 1 (stop is exclusive)
print("\nSLICING FIRST TWO ROWS ([0:2]):\n\n", first_two_rows)
third_column = x[:, 2]     # All rows, column at index 2
print("\nSLICING THIRD COLUMN ([:, 2]]):", third_column)
every_other_col = x[:, ::2]  # All rows, every other column (step=2)
print("\nEVERY OTHER COLUMN ([:, ::2]):\n\n", every_other_col)
last_col = x[:, -1]        # All rows, last column
print("\nLAST COLUMN ([:, -1]):", last_col)

# --- Combined ---
combined = x[0:2, 2:]      # First 2 rows, columns from index 2 to end
print("\nFIRST TWO ROWS, LAST TWO COLS ([0:2, 2:]):\n\n", combined)

# --- .item(): Extract Python scalar from 1-element tensor ---
# IMPORTANT: Only works on tensors with exactly 1 element.
value = single_element_tensor.item()
print("\n.item() PYTHON NUMBER EXTRACTED:", value)
print("TYPE:", type(value))

# --- Boolean masking ---
mask = x > 6                          # True where condition is met
print("MASK (VALUES > 6):\n\n", mask)
mask_applied = x[mask]               # Returns flattened values where True
print("VALUES AFTER APPLYING MASK:", mask_applied)

# --- Fancy indexing ---
row_indices = torch.tensor([0, 2])
col_indices = torch.tensor([1, 3])
get_values = x[row_indices[:, None], col_indices]  # (0,1), (0,3), (2,1), (2,3)
print("\nSPECIFIC ELEMENTS USING INDICES:\n\n", get_values)


# ==================== PART 4: MATH OPERATIONS ====================
#
# CONCEPT:
#   Tensors support all standard arithmetic. Operations are applied
#   element-wise by default. Matrix multiplication uses @ or matmul.
#
# WHY:
#   Neural networks are essentially chains of matrix multiplications
#   and element-wise operations (additions, ReLU, etc.).
#
# COMMON MISTAKES:
#   - Confusing * (element-wise) with @ (matrix multiply)
#   - Shape mismatches in matmul: (A,B) @ (B,C) works, (A,B) @ (C,B) fails

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Element-wise operations
element_add = a + b       # [1+4, 2+5, 3+6] = [5, 7, 9]
print("\nELEMENT-WISE ADDITION:", element_add)
element_mul = a * b       # [1*4, 2*5, 3*6] = [4, 10, 18]
print("ELEMENT-WISE MULTIPLICATION:", element_mul)

# Dot product: sum of element-wise products
dot_product = torch.matmul(a, b)  # 1*4 + 2*5 + 3*6 = 32
print("DOT PRODUCT:", dot_product)

# --- Broadcasting ---
# PyTorch auto-stretches smaller tensors to match larger ones.
# Shape [3] + Shape [3,1] -> Shape [3,3]
b_col = torch.tensor([[1], [2], [3]])  # Shape: [3, 1]
c = a + b_col                           # Broadcasts a to [3,3], adds
print("\nBROADCASTING RESULT:\n\n", c)
print("SHAPE:", c.shape)


# ==================== PART 5: COMPARISON AND LOGICAL OPERATORS ====================
#
# CONCEPT:
#   Comparison operators return boolean tensors (True/False for each element).
#   Logical operators (& for AND, | for OR) combine boolean tensors.
#
# WHY:
#   Used for filtering data, creating masks, and conditional logic.

temperatures = torch.tensor([20, 35, 19, 35, 42])
is_hot = temperatures > 30
is_cool = temperatures <= 20
is_35_degrees = temperatures == 35
print("HOT (> 30 DEGREES):", is_hot)
print("COOL (<= 20 DEGREES):", is_cool)
print("EXACTLY 35 DEGREES:", is_35_degrees)

is_morning = torch.tensor([True, False, False, True])
is_raining = torch.tensor([False, False, True, True])
morning_and_raining = (is_morning & is_raining)   # Both must be True
morning_or_raining = is_morning | is_raining       # At least one True
print("MORNING & (AND) RAINING:", morning_and_raining)
print("MORNING | (OR) RAINING:", morning_or_raining)


# ==================== PART 6: STATISTICS AND DATA TYPES ====================
#
# CONCEPT:
#   Basic statistics (mean, std) are used to understand and normalize data.
#   Data types (dtype) affect precision and memory usage.
#
# WHY:
#   Normalization (using mean and std) is essential for stable training.
#   Using the wrong dtype can waste memory or cause precision errors.

data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
data_mean = data.mean()  # Average: 30.0
print("CALCULATED MEAN:", data_mean)
data_std = data.std()    # Standard deviation: measures spread
print("CALCULATED STD:", data_std)

# Type casting
# Models usually use float32. NumPy defaults to float64 (wastes memory).
print("DATA TYPE:", data.dtype)  # torch.float32
int_tensor = data.int()          # Cast to int (truncates decimals)
print("CASTED DATA:", int_tensor)
print("CASTED DATA TYPE:", int_tensor.dtype)  # torch.int32

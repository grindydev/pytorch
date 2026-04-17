"""
Lesson 1 - Module 1: PyTorch Tensors
=====================================
 WHAT YOU'LL LEARN:
  • What tensors are and how to create them from Python lists, NumPy, and Pandas
  • Tensor shapes: unsqueeze, squeeze, reshape, transpose, concatenate
  • Indexing & slicing: accessing elements, rows, columns, boolean masks, fancy indexing
  • Math operations: element-wise, matrix multiplication, broadcasting
  • Comparison & logical operators on tensors
  • Basic statistics: mean, std, and data type casting

 KEY CONCEPT:
  A **tensor** is PyTorch's core data structure — a multi-dimensional array
  (like a NumPy ndarray) that can live on CPU or GPU and supports automatic
  differentiation for training neural networks.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path


# ==================== SECTION 1: CREATING TENSORS ====================
# Tensors can be created from Python lists, NumPy arrays, Pandas DataFrames,
# or built-in factory functions (zeros, ones, arange).

# --- From a Python list ---
x = torch.tensor([1, 2, 3])
print("FROM PYTHON LISTS:", x)
print("TENSOR DATA TYPE:", x.dtype)  # Default: torch.int64 for integers

# --- From a NumPy array ---
#  NOTE: torch.from_numpy() shares memory with the NumPy array (zero-copy).
#          Changing one will change the other!
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
torch_tensor_from_numpy = torch.from_numpy(numpy_array)
print("TENSOR FROM NUMPY:\n\n", torch_tensor_from_numpy)

# --- From a Pandas DataFrame ---
# Step 1: Read CSV into a DataFrame
df = pd.read_csv(Path.cwd() / 'module1/data.csv')
# Step 2: Extract raw values as a NumPy array
all_values = df.values
# Step 3: Convert to a PyTorch tensor
#  NOTE: torch.tensor() always creates a COPY of the data (safe, independent).
tensor_from_df = torch.tensor(all_values)
print("ORIGINAL DATAFRAME:\n\n", df)
print("\nRESULTING TENSOR:\n\n", tensor_from_df)
print("\nTENSOR DATA TYPE:", tensor_from_df.dtype)  # float64 since CSV data has decimals

# --- Factory functions ---
zeros = torch.zeros(2, 3)       # 2×3 matrix filled with 0s
print("TENSOR WITH ZEROS:\n\n", zeros)

ones = torch.ones(2, 3)         # 2×3 matrix filled with 1s
print("TENSOR WITH ONES:\n\n", ones)

range_tensor = torch.arange(0, 10, step=1)  # Like Python's range(0,10)
print("ARANGE TENSOR:", range_tensor)


# ==================== SECTION 2: SHAPE MANIPULATION ====================
# Understanding shapes is CRITICAL for deep learning. Model layers expect
# inputs with specific shapes, and shape mismatches are the #1 source of bugs.

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print("ORIGINAL TENSOR:\n\n", x)
print("TENSOR SHAPE:", x.shape)  # torch.Size([2, 3]) → 2 rows, 3 columns

# --- unsqueeze(): Add a dimension of size 1 ---
#  USE CASE: Adding a "batch" dimension. Models expect [batch_size, features].
#   Shape: [2, 3] → [1, 2, 3]  (new dim at index 0)
expanded = x.unsqueeze(0)
print("\nTENSOR WITH ADDED DIMENSION AT INDEX 0:\n\n", expanded)
print("TENSOR SHAPE:", expanded.shape)

# --- squeeze(): Remove dimensions of size 1 ---
#   Shape: [1, 2, 3] → [2, 3]  (removes the 1-sized dim)
squeezed = expanded.squeeze()
print("\nTENSOR WITH DIMENSION REMOVED:\n\n", squeezed)
print("TENSOR SHAPE:", squeezed.shape)

# --- reshape(): Change the shape without changing the data ---
#   Shape: [2, 3] → [3, 2]  (total elements must stay the same: 6)
reshaped = x.reshape(3, 2)
print("\nAFTER PERFORMING reshape(3, 2):\n\n", reshaped)
print("TENSOR SHAPE:", reshaped.shape)

# --- transpose(): Swap two dimensions ---
#   Shape: [2, 3] → [3, 2]  (rows become columns, columns become rows)
transposed = x.transpose(0, 1)
print("\nAFTER PERFORMING transpose(0, 1):\n\n", transposed)
print("TENSOR SHAPE:", transposed.shape)

# --- cat(): Concatenate tensors along an existing dimension ---
tensor_a = torch.tensor([[1, 2],
                         [3, 4]])
tensor_b = torch.tensor([[5, 6],
                         [7, 8]])

# dim=1 means concatenate side-by-side (horizontally)
# dim=0 would mean stack top-to-bottom (vertically)
concatenated_tensors = torch.cat((tensor_a, tensor_b), dim=1)
print("TENSOR A:\n\n", tensor_a)
print("\nTENSOR B:\n\n", tensor_b)
print("\nCONCATENATED TENSOR (dim=1):\n\n", concatenated_tensors)


# ==================== SECTION 3: INDEXING & SLICING ====================
# Very similar to NumPy indexing — this is how you access parts of a tensor.

x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print("ORIGINAL TENSOR:\n\n", x)

# --- Single element indexing ---
single_element_tensor = x[1, 2]  # Row 1, Column 2 → value 7
print("\nINDEXING SINGLE ELEMENT AT [1, 2]:", single_element_tensor)

# --- Row indexing ---
second_row = x[1]    # Entire row at index 1 → [5, 6, 7, 8]
print("\nINDEXING ENTIRE ROW [1]:", second_row)

last_row = x[-1]     # Last row → [9, 10, 11, 12]
print("\nINDEXING ENTIRE LAST ROW ([-1]):", last_row)

# --- Slicing (start:stop) ---
first_two_rows = x[0:2]     # Rows 0 and 1 (stop is exclusive)
print("\nSLICING FIRST TWO ROWS ([0:2]):\n\n", first_two_rows)

third_column = x[:, 2]      # All rows, column at index 2 → [3, 7, 11]
print("\nSLICING THIRD COLUMN ([:, 2]]):", third_column)

# --- Step slicing ---
every_other_col = x[:, ::2]  # All rows, every other column (step=2)
print("\nEVERY OTHER COLUMN ([:, ::2]):\n\n", every_other_col)

last_col = x[:, -1]          # All rows, last column
print("\nLAST COLUMN ([:, -1]):", last_col)

# --- Combining slicing and indexing ---
combined = x[0:2, 2:]       # First 2 rows, columns from index 2 to end
print("\nFIRST TWO ROWS, LAST TWO COLS ([0:2, 2:]):\n\n", combined)

# --- .item(): Extract a Python scalar from a single-element tensor ---
#  IMPORTANT: Only works on tensors with exactly 1 element.
#   Useful for extracting loss values or single predictions.
value = single_element_tensor.item()
print("\n.item() PYTHON NUMBER EXTRACTED:", value)
print("TYPE:", type(value))  # Python int or float


# ==================== SECTION 4: BOOLEAN MASKING & FANCY INDEXING ====================
# Powerful ways to select elements based on conditions or specific indices.

# --- Boolean masking ---
# Step 1: Create a boolean mask (True where condition is met)
mask = x > 6
print("MASK (VALUES > 6):\n\n", mask)

# Step 2: Apply the mask to get only the True elements (flattened)
mask_applied = x[mask]
print("VALUES AFTER APPLYING MASK:", mask_applied)

# --- Fancy indexing ---
# Select specific rows and columns using index tensors
row_indices = torch.tensor([0, 2])   # First and third rows
col_indices = torch.tensor([1, 3])   # Second and fourth columns
# Gets the "outer product" of indices: (0,1), (0,3), (2,1), (2,3)
get_values = x[row_indices[:, None], col_indices]
print("\nSPECIFIC ELEMENTS USING INDICES:\n\n", get_values)


# ==================== SECTION 5: MATH OPERATIONS ====================
# Tensors support all standard arithmetic operations.

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# --- Element-wise operations (applied position by position) ---
element_add = a + b         # [1+4, 2+5, 3+6] = [5, 7, 9]
print("\nELEMENT-WISE ADDITION:", element_add)

element_mul = a * b         # [1*4, 2*5, 3*6] = [4, 10, 18]
print("ELEMENT-WISE MULTIPLICATION:", element_mul)

# --- Matrix multiplication (dot product for 1D tensors) ---
#  CONCEPT: Dot product = sum of element-wise products.
#   Used extensively in neural networks (weights × inputs).
dot_product = torch.matmul(a, b)  # 1*4 + 2*5 + 3*6 = 32
print("DOT PRODUCT:", dot_product)

# --- Broadcasting ---
#  CONCEPT: PyTorch automatically "stretches" smaller tensors to match
#   larger ones during operations, without actually copying data.
#   Shape [3] + Shape [3,1] → Shape [3,3]
b_col = torch.tensor([[1], [2], [3]])  # Shape: [3, 1]
c = a + b_col                           # Broadcasts a to [3,3], adds
print("\nBROADCASTING RESULT:\n\n", c)
print("SHAPE:", c.shape)


# ==================== SECTION 6: COMPARISON & LOGICAL OPERATORS ====================
# These return boolean tensors — useful for filtering and conditional logic.

temperatures = torch.tensor([20, 35, 19, 35, 42])

is_hot = temperatures > 30           # Element-wise "greater than"
is_cool = temperatures <= 20         # Element-wise "less than or equal"
is_35_degrees = temperatures == 35   # Element-wise "equal to"
print("HOT (> 30 DEGREES):", is_hot)
print("COOL (<= 20 DEGREES):", is_cool)
print("EXACTLY 35 DEGREES:", is_35_degrees)

# Logical AND (&) and OR (|) — combine boolean tensors
is_morning = torch.tensor([True, False, False, True])
is_raining = torch.tensor([False, False, True, True])

morning_and_raining = (is_morning & is_raining)   # Both must be True
morning_or_raining = is_morning | is_raining      # At least one True
print("MORNING & (AND) RAINING:", morning_and_raining)
print("MORNING | (OR) RAINING:", morning_or_raining)


# ==================== SECTION 7: STATISTICS & DATA TYPES ====================

data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

data_mean = data.mean()  # Average: 30.0
print("CALCULATED MEAN:", data_mean)

data_std = data.std()    # Standard deviation: measures spread
print("CALCULATED STD:", data_std)

# --- Type casting ---
#  WHY IT MATTERS: Neural networks usually use float32. NumPy defaults to float64.
#   Mismatched dtypes can cause errors or waste memory.
print("DATA TYPE:", data.dtype)  # torch.float32
int_tensor = data.int()          # Cast to int (truncates decimals)
print("CASTED DATA:", int_tensor)
print("CASTED DATA TYPE:", int_tensor.dtype)  # torch.int32

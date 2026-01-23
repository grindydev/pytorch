import random

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers
from IPython.display import display

import helper_utils
from pathlib import Path

# Set random seed for reproducibility
SEED = 99
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')
print(f"Using device: {device}")

file_path = Path.cwd() / 'data/recipes_fruit_veg.csv'

# Load the filtered dataset into a pandas DataFrame
df = pd.read_csv(file_path)

# Create the numerical 'label' column: 0 for 'fruit', 1 for 'vegetable'
df['label'] = 1
df.loc[df['category'] == 'fruit', 'label'] = 0

# Extract the recipe names and labels into lists
df_clean = df.dropna(subset=['name'])
texts = df_clean['name'].tolist()
labels = df_clean['label'].tolist()

# Verify the dataset size and class distribution
print(f"Total samples for classification:  {len(texts)}")
print(f"Fruit recipes:                     {labels.count(0)}, {round(labels.count(0)/(labels.count(0) + labels.count(1)) *100,1)} %")
print(f"Vegetable recipes:                 {labels.count(1)}, {round(labels.count(1)/(labels.count(0) + labels.count(1)) *100,1)} %")

# Set the number of random samples to display.
num_samples = 10

# Display a sample of name and label pairs.
display(df[['name', 'label']].sample(num_samples, random_state=25).style.hide(axis="index"))

model_name="distilbert-base-uncased"
model_path= Path.cwd() / "data/distilbert-local-base"

# Ensure the model is downloaded
helper_utils.download_bert(model_name, model_path)

bert_model, bert_tokenizer = helper_utils.load_bert(model_path, num_classes=2)

class RecipeDataset(Dataset):
    """
    Custom PyTorch Dataset for text classification.

    This Dataset class stores raw texts and their corresponding labels. It is
    designed to work efficiently with a Hugging Face tokenizer, performing
    tokenization on the fly for each sample when it is requested.
    """
    def __init__(self, texts, labels, tokenizer):
        """
        Initializes the RecipeDataset.

        Args:
            texts: A list of raw text strings.
            labels: A list of integer labels corresponding to the texts.
            tokenizer: A Hugging Face tokenizer instance for processing text.
        """
        # Store the list of raw text strings.
        self.texts = texts
        # Store the list of integer labels.
        self.labels = labels
        # Store the tokenizer instance that will process the text.
        self.tokenizer = tokenizer

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        # Return the size of the dataset based on the number of texts.
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves and processes one sample from the dataset.

        For a given index, this method fetches the corresponding text and label,
        tokenizes the text, and returns a dictionary of tensors.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing the tokenized inputs ('input_ids',
            'attention_mask') and the 'labels' as tensors.
        """
        # Get the raw text and label for the specified index.
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text, handling tasks like cleaning, numerical conversion,
        # and truncation. Padding is handled later by a DataCollator.
        encoding = self.tokenizer(text, truncation=True, max_length=512)

        # Add the label to the encoding dictionary and convert it to a tensor.
        encoding['labels'] = torch.tensor(label, dtype=torch.long)

        # Return the dictionary containing all processed data for the sample.
        return encoding

# Create the full dataset
full_dataset = RecipeDataset(texts, labels, bert_tokenizer)

# Split the full dataset into an 80% training set and a 20% validation set.
train_dataset, val_dataset = helper_utils.create_dataset_splits(
    full_dataset, 
    train_split_percentage=0.8
)

# Print the number of samples in each set to verify the split.
print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Data collator handles dynamic padding for each batch
data_collator = transformers.DataCollatorWithPadding(tokenizer=bert_tokenizer)

# Set the number of samples to process in each batch.
batch_size = 32

# Create the DataLoader for the training set with `data_collator`
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          collate_fn=data_collator
                         )

# Create the DataLoader for the validation set with `data_collator`
val_loader = DataLoader(val_dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        collate_fn=data_collator
                       )


# Extract all labels from the training set to calculate class weights for handling imbalance.
train_labels_list = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
    
    
# Use scikit-learn's utility to automatically calculate class weights.
class_weights = compute_class_weight(
    # The strategy for calculating weights. 'balanced' is automatic.
    class_weight='balanced',
    # The array of unique class labels (e.g., [0, 1]).
    classes=np.unique(train_labels_list),
    # The list of all training labels, used to count class frequencies.
    y=train_labels_list
)

# Convert the NumPy array of weights into a PyTorch tensor of type float
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Print the final weights to verify the calculation.
print("Calculated Class Weights:")
print(f"  - Fruit (Class 0):     {class_weights[0]:.2f}")
print(f"  - Vegetable (Class 1): {class_weights[1]:.2f}")

# Initialize the CrossEntropyLoss function with the calculated `class_weights`.
loss_function = nn.CrossEntropyLoss(weight=class_weights)

## Uncomment if you want to see the training loop function
helper_utils.display_function(helper_utils.training_loop)

# Set the total number of epochs.
num_epochs = 3

# Call the training loop to start the full fine-tuning process.
full_finetuned_bert, full_results = helper_utils.training_loop(
    bert_model, 
    train_loader, 
    val_loader, 
    loss_function, 
    num_epochs, 
    device
)

# Display the results 
helper_utils.print_final_results(full_results)

# RELOAD the base model to ensure a fair comparison
bert_model, bert_tokenizer = helper_utils.load_bert(model_path, num_classes=2)

print(bert_model)

# embeddings
print("\nEmbeddings: \n")
print(bert_model.distilbert.embeddings)

# first four TransformerBlock layers
print("\nFirst four TransformerBlock layers: \n")
print(bert_model.distilbert.transformer.layer[:4])

# last two TransformerBlock layers
print("\nLast two TransformerBlock layers: \n")
print(bert_model.distilbert.transformer.layer[4:6])

# final classification layers
print("\nFinal Classifier Layer: \n")
print(bert_model.pre_classifier)
print(bert_model.classifier)

# Freeze ALL model parameters first
for param in bert_model.parameters():
    param.requires_grad = False

# Unfreeze the last 2 transformer layers
# Set the number of final transformer layers to unfreeze and train.
layers_to_train = 2 

# Access the list of all transformer layers in the DistilBERT model.
transformer_layers = bert_model.distilbert.transformer.layer

# Loop backwards from the end of the layer list for the number of layers you want to train.
for i in range(layers_to_train):
    # Select a layer using negative indexing (e.g., -1 for the last, -2 for the second to last).
    layer_to_unfreeze = transformer_layers[-(i+1)]
    
    # Iterate through all parameters of the selected layer.
    for param in layer_to_unfreeze.parameters():
        # Set requires_grad to True to make the parameter trainable.
        param.requires_grad = True


# Unfreeze the classifier head
# The final layers of the model must be made trainable to adapt to the new task.

# For DistilBERT, this head consists of two linear layers.
# Unfreeze the pre_classifier layer.
for param in bert_model.pre_classifier.parameters():
    param.requires_grad = True

# Unfreeze the final classifier layer.
for param in bert_model.classifier.parameters():
    param.requires_grad = True

## Uncomment if you want to see the training loop function
helper_utils.display_function(helper_utils.training_loop)

# Set the total number of epochs.
num_epochs = 3

# Call the training loop to start the partial fine-tuning process.
partial_finetuned_bert, partial_results = helper_utils.training_loop(
    bert_model, 
    train_loader, 
    val_loader, 
    loss_function, 
    num_epochs, 
    device
)

# Display the results 
helper_utils.print_final_results(partial_results)

# Compare your results
helper_utils.display_results(full_results, partial_results)

test_products = [
    "Blueberry Muffins",                  # Expected: Fruit
    "Spinach and Feta Stuffed Chicken",   # Expected: Vegetable
    "Classic Carrot Cake with Frosting",  # Expected: Vegetable
    "Tomato and Basil Bruschetta",        # Expected: Vegetable
    "Avocado Toast",                      # Expected: Fruit
    "Zucchini Bread with Walnuts",        # Expected: Vegetable
    "Lemon and Herb Roasted Chicken",     # Expected: Fruit
    "Strawberry Rhubarb Pie",             # Expected: Fruit
]


## Uncomment if you want to see the predict category function
helper_utils.display_function(helper_utils.predict_category)

# Loop through each test product
for product in test_products:
    # Call the prediction function with the required arguments
    category = helper_utils.predict_category(
        partial_finetuned_bert, # Try it with `full_finetuned_bert` as well.
        bert_tokenizer,
        product,
        device
    )
    # Print the results
    print(f"Product: '{product}'\nPredicted: {category}.\n")



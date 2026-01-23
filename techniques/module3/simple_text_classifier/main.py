import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from IPython.display import display

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import re
import pandas as pd

import helper_utils
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

fruit_file_path = Path.cwd() / 'data/recipes_fruit_veg.csv'

# Load the filtered dataset into a pandas DataFrame
df = pd.read_csv(fruit_file_path)

# Display the first 10 rows of the DataFrame
df.head(10)

# Create the new 'label' column and set a default value.
# Set everything to 1 (the label for 'vegetable').
df['label'] = 1

# Use boolean indexing to find all rows where the 'category' is 'fruit'
# Update the 'label' in those specific rows to 0.
df.loc[df['category'] == 'fruit', 'label'] = 0

# Display the first few rows to confirm the new column is correct
df.head()

# Keep only rows with a recipe name.
df_clean = df.dropna(subset=['name'])

# Get recipe names as a list.
texts = df_clean['name'].tolist()

# Get matching labels as a list.
labels = df_clean['label'].tolist()

# Verify the final dataset size and the class distribution.
print(f"Total samples for classification:\t{len(texts)}")
print(f"Fruit recipes:\t\t\t\t{labels.count(0)}, {round(labels.count(0)/(labels.count(0) + labels.count(1)) *100,1)} %")
print(f"Vegetable recipes:\t\t\t{labels.count(1)}, {round(labels.count(1)/(labels.count(0) + labels.count(1)) *100,1)} %")

# Set the number of random samples to display.
num_samples = 10

# Display a sample of name and label pairs.
display(df[['name', 'label']].sample(num_samples, random_state=25).style.hide(axis="index"))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels
)

# Print the number of samples in each set to verify the split.
print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

def preprocess_text(text):
    """Cleans and tokenizes a raw text string.

    Args:
        text (str): The raw text to be processed.

    Returns:
        list: A list of cleaned words (tokens).
    """
    # Convert the entire text to lowercase.
    text = text.lower()
    # Remove all characters that are not letters or whitespace.
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Split the cleaned string into a list of words.
    words = text.split()

    return words


# Preprocess the training and validation texts separately.
processed_train_texts = [preprocess_text(text) for text in train_texts]
processed_val_texts = [preprocess_text(text) for text in val_texts]

class Vocabulary:
    """Builds and manages a word-to-index vocabulary from text.

    Attributes:
        word2idx (dict): Maps words to unique integers.
        idx2word (dict): Maps integers back to words.
        min_freq (int): Minimum word frequency for inclusion.
    """
    def __init__(self, min_freq=1):
        """Initializes the vocabulary.

        Args:
            min_freq (int): Minimum word frequency to be included.
        """
        # Mappings for word-to-index and index-to-word with special tokens.
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.min_freq = min_freq

    def build_vocab(self, texts):
        """Builds the vocabulary from a corpus of tokenized texts.

        Args:
            texts (list of list of str): A corpus of tokenized sentences.
        """
        # Count the frequency of all words in the corpus.
        word_counts = Counter(word for text in texts for word in text)
        
        # Add words to the vocabulary if they meet the minimum frequency.
        for word, count in word_counts.items():
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text):
        """Converts a tokenized text to a sequence of indices.

        Args:
            text (list of str): Tokenized text to encode.

        Returns:
            list of int: Sequence of corresponding indices.
        """
        # Use the <unk> token for words not in the vocabulary.
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in text]

    def __len__(self):
        """Returns the vocabulary size."""
        return len(self.word2idx)


vocab = Vocabulary(min_freq=2)

# Build the vocabulary using ONLY the processed training texts.
vocab.build_vocab(processed_train_texts)

print("Number of words in the vocabulary:", len(vocab))

# Encode both the training and validation texts using the vocabulary.
indexed_train_texts = [vocab.encode(text) for text in processed_train_texts]
indexed_val_texts = [vocab.encode(text) for text in processed_val_texts]

class TextDataset(Dataset):
    """
    A custom PyTorch Dataset for handling text and label data.

    This class encapsulates a dataset of texts and their corresponding labels,
    making it compatible with PyTorch's DataLoader.
    """
    def __init__(self, texts, labels):
        """
        Initializes the TextDataset object.

        Args:
            texts: A list or array of numericalized text sequences.
            labels: A list or array of corresponding labels.
        """
        # Store the collection of texts.
        self.texts = texts
        # Store the collection of labels.
        self.labels = labels
        # Find unique class labels and store them
        self.classes = sorted(list(set(labels)))

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        # Return the size of the dataset based on the number of texts.
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at a given index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing the text and label as PyTorch tensors.
        """
        # Create a dictionary for the sample at the specified index.
        sample = {
            'text': torch.tensor(self.texts[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        # Return the sample dictionary.
        return sample


# Create the training and validation datasets directly from the split data.
train_dataset = TextDataset(indexed_train_texts, train_labels)
val_dataset = TextDataset(indexed_val_texts, val_labels)

# Print the number of samples in each set to verify.
print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


def collate_batch_embeddingbag(batch):
    """
    Formats a batch for nn.EmbeddingBag by flattening texts and creating offsets.

    Args:
        batch (list of dict): A list of samples from the Dataset.

    Returns:
        tuple: A tuple of (flattened_text, offsets, labels) tensors.
    """
    # Extract labels from each item and create a single tensor.
    labels = torch.tensor([item['label'] for item in batch])
    # Extract the individual text tensors from the batch.
    texts = [item['text'] for item in batch]
    # Create a list of the lengths of each text, prepended with 0.
    offsets = [0] + [len(text) for text in texts]
    # Convert to a tensor of offsets representing the starting index of each sequence.
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    # Concatenate all text tensors into a single, long 1D tensor.
    flattened_text = torch.cat(texts)
    
    # Return the three tensors required for the EmbeddingBag model.
    return flattened_text.to(device), offsets.to(device), labels.to(device)


def collate_batch_manual(batch):
    """
    Formats a batch by padding texts to the same length.

    Args:
        batch (list of dict): A list of samples from the Dataset.

    Returns:
        tuple: A tuple of (padded_texts, labels) tensors.
    """
    # Extract labels from each item and create a single tensor.
    labels = torch.tensor([item['label'] for item in batch])
    # Extract the individual text tensors from the batch.
    texts = [item['text'] for item in batch]
    # Find the length of the longest sequence in this specific batch.
    max_len = max(len(text) for text in texts)
    # Create a tensor of zeros to hold the padded batch.
    padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
    # Copy each text sequence into the corresponding row of the padded tensor.
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = text
        
    # Return the padded texts and labels, moved to the active device.
    return padded_texts.to(device), labels.to(device)


# Set the number of samples to process in each batch.
batch_size = 32

# Create the DataLoader for the training set with `collate_batch_embeddingbag`
train_loader_embag = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                collate_fn=collate_batch_embeddingbag
                               )

# Create the DataLoader for the validation set with `collate_batch_embeddingbag`
val_loader_embag = DataLoader(val_dataset, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              collate_fn=collate_batch_embeddingbag
                             )
# Create the DataLoader for the training set with `collate_batch_manual`
train_loader_manual = DataLoader(train_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 collate_fn=collate_batch_manual
                                )

# Create the DataLoader for the validation set with `collate_batch_manual`
val_loader_manual = DataLoader(val_dataset, 
                               batch_size=batch_size, 
                               shuffle=False, 
                               collate_fn=collate_batch_manual
                              )

class EmbeddingBagClassifier(nn.Module):
    """
    A simple text classifier using a pre-trained nn.EmbeddingBag layer.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The size of the embedding vectors.
        num_classes (int): The number of output classes.
    """
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()
        # The core layer that efficiently computes embeddings for variable-length sequences.
        # 'mode=mean' specifies that it will average the embeddings of all words in a sequence.
        self.embedding_bag = nn.EmbeddingBag(vocab_size, embedding_dim, mode='mean')
        # A standard dropout layer for regularization to prevent overfitting.
        self.dropout = nn.Dropout(0.5)
        # The final fully connected layer that maps the embedding to the output classes.
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, text, offsets=None):
        """
        Defines the forward pass of the model.

        Args:
            text (torch.Tensor): A 1D tensor of concatenated text indices.
            offsets (torch.Tensor): A 1D tensor of starting positions for each sequence.

        Returns:
            torch.Tensor: The raw output scores (logits) for each class.
        """
        # Compute the single embedding vector for each sequence in the batch.
        embedded = self.embedding_bag(text, offsets)
        # Apply dropout to the embeddings.
        embedded = self.dropout(embedded)
        # Pass the result through the final linear layer to get class scores.
        return self.fc(embedded)


class ManualPoolingClassifier(nn.Module):
    """
    A text classifier that uses an nn.Embedding layer followed by a manual
    pooling strategy (mean, max, or sum).

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The size of the embedding vectors.
        num_classes (int): The number of output classes.
        pooling (str, optional): The pooling strategy to use.
                                 Options: 'mean', 'max', 'sum'.
                                 Defaults to 'mean'.
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, pooling='mean'):
        super().__init__()
        # Embedding layer that maps token IDs to vectors.
        # padding_idx=0 ensures that the padding token is ignored during training.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Store the chosen pooling strategy.
        self.pooling = pooling
        # Final fully connected layer for classification.
        self.fc = nn.Linear(embedding_dim, num_classes)
        # Dropout layer for regularization.
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        """
        Defines the forward pass of the model.

        Args:
            text (torch.Tensor): A batch of padded text indices.

        Returns:
            torch.Tensor: The raw output scores (logits) for each class.
        """
        # Get embeddings for the input text.
        # Shape: (batch_size, max_len, embedding_dim)
        embedded = self.embedding(text)

        # Create a mask to ignore padding tokens in pooling calculations.
        # The mask will have 1s for real tokens and 0s for padding tokens.
        mask = (text != 0).float().unsqueeze(-1)
        # Apply the mask by element-wise multiplication.
        embedded = embedded * mask

        # Apply the chosen pooling strategy.
        if self.pooling == 'mean':
            # Sum embeddings and divide by the actual number of non-padded tokens.
            # clamp(min=1) prevents division by zero for empty sequences.
            pooled = embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == 'max':
            # Set padded positions to negative infinity so they are ignored by max().
            embedded[mask.squeeze(-1) == 0] = float('-inf')
            pooled, _ = embedded.max(dim=1)
        elif self.pooling == 'sum':
            # Sum the embeddings of all non-padded tokens.
            pooled = embedded.sum(dim=1)

        # Apply dropout and the final linear layer.
        pooled = self.dropout(pooled)
        return self.fc(pooled)

# Define Model Hyperparameters
vocab_size = len(vocab)
embedding_dim = 64
num_classes = 2

model_embag = EmbeddingBagClassifier(vocab_size, embedding_dim, num_classes)

# Initialize the 'mean' pooling variant
model_manual_mean = ManualPoolingClassifier(vocab_size, embedding_dim, num_classes, pooling='mean')

# Initialize the 'max' pooling variant
model_manual_max = ManualPoolingClassifier(vocab_size, embedding_dim, num_classes, pooling='max')

# Initialize the 'sum' pooling variant
model_manual_sum = ManualPoolingClassifier(vocab_size, embedding_dim, num_classes, pooling='sum')

train_labels_list = train_labels

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

# ### Uncomment if you want to see the training loop function
helper_utils.display_function(helper_utils.training_loop)

num_epochs = 5

# Perform training for the EmbeddingBagClassifier.
trained_embag, results_embag = helper_utils.training_loop(
    model_embag, 
    train_loader_embag, 
    val_loader_embag, 
    loss_function,
    num_epochs, 
    device
)

# Display the results 
print("\nModel: EmbeddingBagClassifier")
helper_utils.print_final_metrics(results_embag)

# Perform training for the ManualPoolingClassifier, `mean` variant.
trained_mean, results_mean = helper_utils.training_loop(
    model_manual_mean, 
    train_loader_manual, 
    val_loader_manual, 
    loss_function, 
    num_epochs, 
    device
)

# Display the results 
print("\nModel: ManualPoolingClassifier (MEAN)")
helper_utils.print_final_metrics(results_mean)

# Perform training for the ManualPoolingClassifier, `max` variant.
trained_max, results_max = helper_utils.training_loop(
    model_manual_max, 
    train_loader_manual, 
    val_loader_manual, 
    loss_function, 
    num_epochs, 
    device
)

# Display the results 
print("\nModel: ManualPoolingClassifier (MAX)")
helper_utils.print_final_metrics(results_max)

# Perform training for the ManualPoolingClassifier, `sum` variant.
trained_sum, results_sum = helper_utils.training_loop(
    model_manual_sum, 
    train_loader_manual, 
    val_loader_manual, 
    loss_function, 
    num_epochs, 
    device
)

# Display the results 
print("\nModel: ManualPoolingClassifier (SUM)")
helper_utils.print_final_metrics(results_sum)

results_df = helper_utils.get_results_df(
    results_embag,
    results_mean,
    results_max,
    results_sum
)

results_df

all_trained_data = {
    'EmbeddingBag': (trained_embag, results_embag),
    'Manual (mean)': (trained_mean, results_mean),
    'Manual (max)': (trained_max, results_max),
    'Manual (sum)': (trained_sum, results_sum)
}

best_model = helper_utils.plot_and_select_best_model(all_trained_data)

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
        best_model,
        product,
        vocab,
        preprocess_text,
        device
    )
    # Print the results
    print(f"Product: '{product}'\nPredicted: {category}.\n")
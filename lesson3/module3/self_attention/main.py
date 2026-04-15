import math
import os
import re
import urllib.request
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from pathlib import Path

# 1. Tiny toy dataset
sentences = """
the dog chased the cat
the cat chased the mouse
the dog ran fast
the mouse ran fast
the cat lay down
"""

# Build vocab
tokens = sentences.split()
vocab = ['<pad>', '<unk>'] + sorted(set(tokens))
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}
print("Vocab:", vocab)


class SimpleTokenizer:
    """
    Splits on whitespace and lowercases, with optional regex for real word tokens.
    """
    def __init__(self):
        """
        Initializes the SimpleTokenizer instance.
        """
        pass

    def __call__(self, text):
        """
        Processes the input text into a list of lowercase word tokens.
        
        Args:
            text: The string to be tokenized.
            
        Returns:
            A list of strings containing the extracted tokens.
        """
        # Option 1: Basic split (uncomment to just split on spaces)
        # return text.lower().split()
        # Option 2: More robust - returns only word tokens (ignores punctuation)
        return re.findall(r'\b\w+\b', text.lower())
    

# Usage example:
tokenizer = SimpleTokenizer()
tokens = tokenizer("The Dog chased the Cat.")
print(tokens)  # Output: ['the', 'dog', 'chased', 'the', 'cat']

def build_vocab(sentences, tokenizer, min_freq=1):
    """
    Constructs a vocabulary and mapping dictionaries from a collection of sentences.

    Args:
        sentences: A collection of strings to be processed.
        tokenizer: A callable object used to split sentences into individual tokens.
        min_freq: The minimum number of occurrences required for a token to be 
                  included in the vocabulary.

    Returns:
        vocab: A list of unique tokens including special padding and unknown markers.
        word2idx: A dictionary mapping each token string to its unique integer index.
        idx2word: A dictionary mapping each integer index back to its token string.
    """
    # Initialize a frequency counter for tokens
    counter = Counter()
    # Iterate through each sentence to update token counts
    for sent in sentences:
        # Generate tokens using the provided tokenizer and update the counter
        counter.update(tokenizer(sent))

    # Define the initial vocabulary with special tokens and filter by frequency
    vocab = ['<pad>', '<unk>'] + [w for w, c in counter.items() if c >= min_freq]

    # Map each unique word in the vocabulary to a specific integer index
    word2idx = {w: i for i, w in enumerate(vocab)}

    # Map each integer index back to its corresponding word
    idx2word = {i: w for i, w in enumerate(vocab)}

    # Provide the list of tokens and the bidirectional mapping dictionaries
    return vocab, word2idx, idx2word


# Using our sample sentences and tokenizer
sentences = [
    "the dog chased the cat",
    "the cat chased the mouse",
    "the dog ran fast",
    "the mouse ran fast",
    "the cat lay down"
]

tokenizer = SimpleTokenizer()                 # Define the tokenizer (splits into lowercase words)
vocab, word2idx, idx2word = build_vocab(sentences, tokenizer)  # Build vocab & mappings

print(vocab)

word = 'dog'
id_word = word2idx[word]
print(f"ID for word = {word}: {id_word}")
print(f"Word for ID = {id_word}: {idx2word[id_word]}")

tokenizer = SimpleTokenizer()
vocab, word2idx, idx2word = build_vocab(sentences, tokenizer)

# 2. Parameters
SEQ_LEN = 4   # Length of input sequence for each example

# 3. Convert sentences to token ID lists
encoded_sentences = []  # Will be a list of lists of token IDs
for sent in sentences:
    tokens = tokenizer(sent)  # Split sentence into tokens
    ids = [word2idx.get(tok, word2idx['<unk>']) for tok in tokens]  # Map tokens to IDs
    encoded_sentences.append(ids)

# 4. Create sliding window dataset (inputs, targets)
inputs = []
targets = []
for ids in encoded_sentences:
    # For each possible window in the sentence
    for i in range(len(ids) - SEQ_LEN):
        window = ids[i:i+SEQ_LEN]        # Input: SEQ_LEN-token window
        target = ids[i+SEQ_LEN]          # Target: next token after the window
        inputs.append(window)
        targets.append(target)

# 5. Let's show the dataset as text for illustration
for inp, tgt in zip(inputs, targets):
    inp_words = [idx2word[i] for i in inp]
    tgt_word = idx2word[tgt]
    print(f"Input: {inp_words}  →  Target: {tgt_word}")


class TinyDataset(Dataset):
    """
    A custom Dataset class for managing input-target pairs in a PyTorch-compatible format.

    Args:
        inputs: A list or array of input sequences/windows.
        targets: A list or array of target labels corresponding to the inputs.
    """
    def __init__(self, inputs, targets):
        """
        Initializes the dataset by converting raw data into tensors.

        Args:
            inputs: The input data sequences.
            targets: The target labels or word indices.
        """
        # Convert input windows to a long tensor for efficient indexing
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        # Convert target indices to a long tensor
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        """
        Provides the total number of samples available in the dataset.

        Returns:
            An integer representing the total count of input samples.
        """
        # Return the total length of the input tensor
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves a specific sample and its corresponding target by index.

        Args:
            idx: The integer index of the desired sample.

        Returns:
            sample: The input tensor at the specified index.
            target: The target tensor at the specified index.
        """
        # Return the input-target pair as a tuple for the requested index
        return self.inputs[idx], self.targets[idx]
    


# Create an instance of your dataset
dataset = TinyDataset(inputs, targets)

# Create a DataLoader that will feed batches of data to your model during training.
# batch_size=4: each batch will contain 4 (input, target) pairs
# shuffle=True: randomize order each epoch to improve training
# num_workers=0: no extra processes for loading data (good for small datasets)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

class ManualSelfAttention(nn.Module):
    """
    A custom PyTorch module that implements a standard self-attention mechanism.
    
    Args:
        d: The dimensionality of the input and output features.
    """
    def __init__(self, d):
        """
        Initializes the linear layers for query, key, and value projections.

        Args:
            d: The dimensionality of the hidden state.
        """
        super().__init__()
        # Define the linear transformation for query vectors
        self.to_q = nn.Linear(d, d, bias=False)
        # Define the linear transformation for key vectors
        self.to_k = nn.Linear(d, d, bias=False)
        # Define the linear transformation for value vectors
        self.to_v = nn.Linear(d, d, bias=False)

    def forward(self, x):
        """
        Executes the forward pass of the self-attention mechanism.

        Args:
            x: The input tensor of shape [batch, sequence_length, dimension].

        Returns:
            out: The resulting context-aware representations after attention weighting.
            attn: The attention weight matrix representing token interactions.
        """
        # Project the input tensor into query space
        Q = self.to_q(x)
        # Project the input tensor into key space
        K = self.to_k(x)
        # Project the input tensor into value space
        V = self.to_v(x)

        # Calculate raw attention scores using scaled dot-product between queries and keys
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        # Apply softmax to normalize scores into probability distributions
        attn = F.softmax(scores, dim=-1)

        # Aggregate the value vectors based on the computed attention weights
        out = torch.matmul(attn, V)

        # Provide the transformed sequence and the attention weights
        return out, attn

sentence = "the dog chased the cat"
tokens = tokenizer(sentence)
print("Tokens:", tokens)  # ['dog', 'chased', 'cat']

token_ids = [word2idx.get(tok, word2idx['<unk>']) for tok in tokens]
print("Token IDs:", token_ids)  # e.g. [2, 3, 4]

embedding_dim = 2
# We'll make trainable embeddings for realism:
embed = nn.Embedding(len(vocab), embedding_dim)
torch.manual_seed(42)  # For reproducibility
x = embed(torch.tensor(token_ids).unsqueeze(0))  # shape: (1, 5, 2)
print("Input embeddings:\n", x)

attn_layer = ManualSelfAttention(embedding_dim)

# Put the input through self-attention
out, attn = attn_layer(x)

print("Attention weights:\n", attn[0].detach().numpy())
print("Output representations:\n", out[0].detach().numpy())

print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("idx2word:", idx2word)

print("\nAttention Weights Matrix (rows: query token, columns: attended token):")
for i, w in enumerate(tokens):
    row = ["{:.2f}".format(a) for a in attn[0, i].detach().cpu().numpy()]
    print(f"{w:>8} attends to -> {row}")

class SelfAttnWithPositionalEmbedding(nn.Module):
    """
    A neural network module that combines token and positional embeddings with self-attention.

    Args:
        vocab_size: The total number of unique tokens in the vocabulary.
        seq_len: The fixed length of input sequences.
        emb_dim: The dimensionality of the embedding vectors.
    """
    def __init__(self, vocab_size, seq_len, emb_dim):
        """
        Initializes the embedding layers, attention mechanism, and output projection.

        Args:
            vocab_size: Size of the vocabulary.
            seq_len: Maximum length of the sequence.
            emb_dim: Dimension of the embeddings.
        """
        super().__init__()
        # Table for mapping token indices to continuous vector representations
        self.tok_embed = nn.Embedding(vocab_size, emb_dim)
        # Learnable table for mapping sequence positions to vector representations
        self.pos_embed = nn.Embedding(seq_len, emb_dim)
        # Self-attention module for capturing dependencies between tokens
        self.attn = ManualSelfAttention(emb_dim)
        # Linear layer to project context-aware representations back to vocabulary size
        self.fc = nn.Linear(emb_dim, vocab_size)
        # Internal storage for the maximum sequence length
        self.seq_len = seq_len

    def forward(self, token_ids):
        """
        Processes input token sequences to produce prediction logits and attention weights.

        Args:
            token_ids: A tensor of token indices with shape [batch_size, seq_len].

        Returns:
            logits: Predicted scores for the next token based on the final hidden state.
            attn_weights: The attention weight matrix from the self-attention layer.
        """
        batch_size, seq_len = token_ids.shape
        # Generate a range of indices representing the position of each token in the sequence
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        # Retrieve word-level embedding vectors for the input tokens
        word_vecs = self.tok_embed(token_ids)
        # Retrieve learnable positional embedding vectors for each sequence index
        pos_vecs = self.pos_embed(positions)
        # Combine word and position information via element-wise summation
        input_vecs = word_vecs + pos_vecs
        # Pass the combined embeddings through the self-attention mechanism
        attn_out, attn_weights = self.attn(input_vecs)
        # Extract the hidden representation corresponding to the final token in the sequence
        last_hidden = attn_out[:, -1, :]
        # Transform the final hidden state into prediction logits over the vocabulary
        logits = self.fc(last_hidden)
        # Provide the prediction scores and the attention weights for evaluation or analysis
        return logits, attn_weights


x = torch.tensor([[[ 0.12, -0.55,  0.33,  0.10],
                   [-0.44,  0.91, -0.12, -0.77],
                   [ 0.48,  0.02,  0.05,  0.39],
                   [ 0.12, -0.55,  0.33,  0.10],
                   [-0.30,  0.14, -0.70,  0.81]]])  # [1, 5, 4]

attn = ManualSelfAttention(d=4)
out, attn_weights = attn(x)

print("Attention weights matrix (attn_weights):\n", attn_weights[0].detach().numpy())
print('\nExplanation:')
print("Each row i shows the attention distribution (softmaxed) over all positions in the input sequence,")
print("when computing the updated representation for token i. Rows sum to 1.\n")

print("Self-attention output (out):\n", out[0].detach().numpy())
print('\nExplanation:')
print("Each row is the new vector for input position i, computed as a weighted sum")
print("of the original value vectors, using that row from the attention weights matrix as weights.\n")


# 1. Set up model parameters and create your attention model
vocab_size = len(vocab)
embed_dim = 8         # Number of embedding dimensions (try 4, 8, 16, etc.)
seq_len = SEQ_LEN     # Length of your training window
model = SelfAttnWithPositionalEmbedding(vocab_size, seq_len, embed_dim)


def plot_attention(attn_weights, tokens, title="Self-Attention Map"):
    """
    Visualizes the self-attention weights for a sequence using a heatmap.
    
    Args:
        attn_weights: A tensor of attention scores, usually of shape [batch, seq_len, seq_len].
        tokens: A list of strings representing the tokens for labeling the axes.
        title: A string to be used as the title of the generated plot.

    Returns:
        None. This function displays a plot directly.
    """
    # Extract attention weights for the first batch entry and transfer to host memory
    aw = attn_weights[0].detach().cpu().numpy()
    # Initialize the figure with dimensions scaled to the number of tokens
    plt.figure(figsize=(1.2 * len(tokens), 5))
    # Render the attention matrix as a heatmap with a blue color gradient
    plt.imshow(aw, cmap='Blues')
    # Assign token strings to the horizontal axis with a specific rotation for readability
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    # Assign token strings to the vertical axis
    plt.yticks(range(len(tokens)), tokens)
    # Include a legend showing the mapping of colors to attention intensity
    plt.colorbar()
    # Set the provided title for the visualization
    plt.title(title)
    # Adjust layout parameters to prevent label clipping
    plt.tight_layout()
    # Render the final visualization to the screen
    plt.show()

# 1. Select a sample input from your training data
ex_ix = 0  # or any valid index into your windowed input dataset
input_ids = inputs[ex_ix]                          # e.g., [2, 3, 4, 2]
tokens = [idx2word[i] for i in input_ids]          # Human-readable tokens

model.eval()
x_example = torch.tensor([input_ids], dtype=torch.long)  # [batch, seq]

with torch.no_grad():
    logits, attn_weights = model(x_example)

# 3. Plot attention map before OR after training
plot_attention(attn_weights, tokens, title="Self-Attention Map")


def train_model(model, loader, loss_fn, optimizer, epochs=20, device='cpu'):
    """
    Executes the training loop for a given model over a specified number of epochs.

    Args:
        model: The neural network model to be trained.
        loader: The DataLoader providing batches of training data.
        loss_fn: The criterion used to calculate the model error.
        optimizer: The optimization algorithm used to update model weights.
        epochs: The total number of complete passes through the training dataset.
        device: The hardware device (e.g., 'cpu' or 'cuda') to perform computations on.

    Returns:
        None. This function performs in-place updates to the model weights.
    """
    # Transfer the model parameters to the specified computation device
    model.to(device)
    # Iterate through the training process for the defined number of epochs
    for epoch in range(epochs):
        # Configure the model for training mode
        model.train()
        # Initialize an accumulator for the total loss across the epoch
        total_loss = 0
        # Iterate through the data batches with a progress bar visualization
        with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for xb, yb in pbar:
                # Move the input and target tensors to the active device
                xb, yb = xb.to(device), yb.to(device)
                # Reset the gradients of all optimized parameters
                optimizer.zero_grad()
                # Execute the forward pass to obtain prediction logits
                logits, _ = model(xb)
                # Calculate the difference between predictions and ground truth labels
                loss = loss_fn(logits, yb)
                # Perform backpropagation to compute gradients for the current batch
                loss.backward()
                # Apply the gradients to update the model parameters
                optimizer.step()
                # Sum the loss for the batch weighted by the batch size
                total_loss += loss.item() * xb.size(0)
                # Update the progress bar with the current batch loss
                pbar.set_postfix(loss=loss.item())
        # Compute the mean loss over the entire dataset for the current epoch
        avg_loss = total_loss / len(loader.dataset)
        # Output the performance summary for the completed epoch
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

# Example usage:
optimizer = optim.Adam(model.parameters(), lr=0.01)     # Adam is a popular optimizer for NLP
loss_fn = nn.CrossEntropyLoss()                         # Classic loss for next-token prediction

# Assume 'loader' is your DataLoader for (input_window, target_next_token) pairs,
# and 'device' is set to "cuda" if available, else "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

train_model(model, loader, loss_fn, optimizer, epochs=25, device=device)


# 1. Select a sample input from your training data
ex_ix = 0  # or any valid index into your windowed input dataset
input_ids = inputs[ex_ix]                          # e.g., [2, 3, 4, 2]
tokens = [idx2word[i] for i in input_ids]          # Human-readable tokens

model.eval()
x_example = torch.tensor([input_ids], dtype=torch.long, device = device)  # [batch, seq]

with torch.no_grad():
    logits, attn_weights = model(x_example)

# 3. Plot attention map before OR after training
plot_attention(attn_weights, tokens, title="Self-Attention Map")


def generate_next_words(model, sentence, tokenizer, word2idx, idx2word, max_tokens=5, seq_len=4, device='cpu'):
    """
    Predicts and appends subsequent tokens to a given input sequence using a trained model.

    Args:
        model: The trained neural network used for prediction.
        sentence: The initial input string to start generation from.
        tokenizer: A callable object used to convert strings into tokens.
        word2idx: A dictionary mapping token strings to their respective integer indices.
        idx2word: A dictionary mapping integer indices back to their token strings.
        max_tokens: The total number of new tokens to be generated.
        seq_len: The fixed context window size required by the model.
        device: The hardware device ('cpu' or 'cuda') to perform inference on.

    Returns:
        generated: A list of strings containing the original tokens plus the newly predicted tokens.
    """
    # Set the model to evaluation mode to disable training-specific behaviors like dropout
    model.eval()
    # Convert the raw input sentence into a list of individual tokens
    generated = tokenizer(sentence)

    # Iteratively predict the next token for the specified number of steps
    for _ in range(max_tokens):
        # Extract the most recent tokens to fit the model's context length, padding if necessary
        window = generated[-seq_len:] if len(generated) >= seq_len \
                 else ['<pad>'] * (seq_len - len(generated)) + generated

        # Map window tokens to indices, using the unknown token marker for missing words
        input_ids = torch.tensor([[word2idx.get(w, word2idx['<unk>']) for w in window]], dtype=torch.long).to(device)

        # Disable gradient calculation to save memory and compute during inference
        with torch.no_grad():
            # Perform a forward pass to obtain prediction logits
            logits, _ = model(input_ids)
            # Select the token index with the highest predicted probability
            next_id = logits.argmax(dim=-1).item()

        # Retrieve the string representation of the predicted token index
        next_word = idx2word[next_id]
        # Append the new word to the sequence for the next iteration of context
        generated.append(next_word)

    # Provide the full list of tokens including the generated sequence
    return generated


# Example usage:
sentence = "the dog chased the"
output = generate_next_words(model, sentence, tokenizer, word2idx, idx2word, max_tokens=1, seq_len=SEQ_LEN, device=device)
print("Generated sequence:", " ".join(output))

# Download if needed
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filename = Path.cwd() / "data/shakespeare.txt"

# Check if file already exists
if os.path.exists(filename):
    print(f"'{filename}' already exists, skipping download.\n")
else:
    print(f"Downloading '{filename}'...")
    urllib.request.urlretrieve(url, filename)
    print(f"Download complete!\n")

# Read the text
with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

print(text[:300])  # Just peek at the start


class ShakespeareTokenizer:
    """
    A specialized tokenizer designed for processing theatrical or poetic text.
    """
    def __call__(self, text):
        """
        Processes the input text into a specific list of tokens.

        Args:
            text: The raw string to be tokenized.

        Returns:
            A list of strings containing words, contractions, special line 
            break tokens, and punctuation marks.
        """
        # Substitute newline characters with a identifiable string token
        text = text.replace('\n', ' <nl> ')
        # Extract word characters, possessives, special tokens, and symbols using regex
        return re.findall(r"\w+(?:'\w+)?|<nl>|[^\w\s]", text)

# Instantiate your tokenizer
tokenizer = ShakespeareTokenizer()

# Build vocabulary from all Shakespeare lines using your tokenizer
vocab, word2idx, idx2word = build_vocab([text], tokenizer, min_freq=1)
print("Vocab size:", len(vocab))
print("First 20 vocab words:", vocab[:20])

SEQ_LEN = 25
tokens = tokenizer(text)    # Tokenize the full text as one sequence!
inputs = []
targets = []
for i in range(len(tokens) - SEQ_LEN):
    window = tokens[i:i+SEQ_LEN]
    target = tokens[i+SEQ_LEN]
    input_ids = [word2idx.get(w, word2idx['<unk>']) for w in window]
    target_id = word2idx.get(target, word2idx['<unk>'])
    inputs.append(input_ids)
    targets.append(target_id)
        
print("Number of (input, target) pairs:", len(inputs))
print("Example input:", [idx2word[i] for i in inputs[0]])
print("Example target:", idx2word[targets[0]])

class ShakespeareDataset(Dataset):
    """
    A dataset class for handling tokenized text sequences and their corresponding targets.

    Args:
        inputs: A list or array of sequence windows representing the input data.
        targets: A list or array of labels or next-word indices corresponding to the inputs.
    """
    def __init__(self, inputs, targets):
        """
        Initializes the dataset by converting input and target data into tensors.

        Args:
            inputs: The raw input sequences.
            targets: The ground truth labels for the sequences.
        """
        # Maintain inputs and targets as long tensors to ensure compatibility with embedding layers
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        # Store the target labels as long tensors for use in loss calculation
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        """
        Calculates the total size of the dataset.

        Returns:
            The total number of samples available in the dataset.
        """
        # Return the count of samples based on the length of the input tensor
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves a single input-target pair by its index.

        Args:
            idx: The integer index of the data point to retrieve.

        Returns:
            input_sample: The tensor representing the input window at the given index.
            target_sample: The tensor representing the target label at the given index.
        """
        # Provide the input sequence and its associated target as a tuple
        return self.inputs[idx], self.targets[idx]
    
dataset = ShakespeareDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
vocab_size = len(vocab)  # Store vocab size for easy reference


class SelfAttnWithMHA(nn.Module):
    """
    A sequence modeling architecture using multi-head attention and positional embeddings.

    Args:
        vocab_size: The total number of unique tokens in the vocabulary.
        seq_len: The fixed length of input sequences.
        embed_dim: The dimensionality of the hidden states and embeddings.
        num_heads: The number of parallel attention heads.
        dropout: The dropout probability applied to the attention weights.
    """
    def __init__(self, vocab_size, seq_len, embed_dim=768, num_heads=12, dropout=0.1):
        """
        Initializes the model layers and internal configurations.

        Args:
            vocab_size: Size of the vocabulary.
            seq_len: Maximum sequence length.
            embed_dim: Dimensionality of the embedding space.
            num_heads: Number of attention heads.
            dropout: Regularization dropout rate.
        """
        super().__init__()
        # Layer to map token indices to continuous vector representations
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        # Layer to provide order information to the model via learnable vectors
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
        # Multi-head attention mechanism for capturing complex sequence dependencies
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Projection layer to transform contextual representations into vocabulary logits
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids):
        """
        Performs a forward pass on a batch of token sequences.

        Args:
            token_ids: A tensor of token indices with shape [batch_size, seq_len].

        Returns:
            logits: Prediction scores for the next token based on the final hidden state.
            attn_w: The computed attention weights representing token relationships.
        """
        batch_size, seq_len = token_ids.shape
        # Create a range of indices to represent the relative position of each token
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        # Combine token-specific and position-specific embeddings
        input_vecs = self.tok_embed(token_ids) + self.pos_embed(positions)
        # Calculate contextual representations and attention scores across all heads
        attn_out, attn_w = self.attn(input_vecs, input_vecs, input_vecs, need_weights=True)
        # Isolate the hidden state of the final token in the sequence
        last_hidden = attn_out[:, -1, :]
        # Generate the final output scores over the vocabulary
        logits = self.fc(last_hidden)
        # Provide the prediction results along with the attention weight matrix
        return logits, attn_w
    
model = SelfAttnWithMHA(vocab_size=vocab_size, seq_len=SEQ_LEN)
# Create a new instance of your (multi-head attention) model, sized to your vocab and sequence length

loss_fn = nn.CrossEntropyLoss()  
# Define the loss function (CrossEntropy is standard for language modeling);
# Redefine here to ensure it's fresh and configured for your current task

optimizer = optim.Adam(model.parameters(), lr=0.001)
# Define the optimizer (Adam adjusts model parameters during training);
# Always recreate the optimizer after making a new model, or when model params change

model.to(device)
# Move your model’s parameters to the chosen device (CPU or GPU)

torch.cuda.empty_cache()
# (Optional) Clear unused memory from the GPU to avoid memory leaks or OOM errors;
# Especially useful if you ran a different model before in the same session

# Example usage:
sentence = "to be or not"
output = generate_next_words(model, sentence, tokenizer, word2idx, idx2word, max_tokens=50, seq_len=SEQ_LEN, device=device)
print("Generated sequence:", " ".join(output))

def train_model(model, loader, loss_fn, optimizer, epochs=10, device='cpu', vocab_size=None):
    """
    Executes the training loop for a sequence model over a specified number of epochs.

    Args:
        model: The neural network model to be trained.
        loader: The DataLoader instance providing batches of training data.
        loss_fn: The loss function used to evaluate model performance.
        optimizer: The optimization algorithm used to update model weights.
        epochs: The total number of iterations over the complete dataset.
        device: The target hardware for computation (e.g., 'cpu' or 'cuda').
        vocab_size: Optional parameter specifying the size of the vocabulary.

    Returns:
        None. This function modifies the model and optimizer states in-place.
    """
    # Transfer the model to the specified hardware device
    model.to(device)
    # Begin iterating through the specified number of training cycles
    for epoch in range(epochs):
        # Configure the model for training mode to enable specific behaviors like dropout
        model.train()
        # Initialize an accumulator for the cumulative loss within the epoch
        total_loss = 0
        # Determine the total number of batches in the data loader
        n_batches = len(loader)
        # Iterate through data batches using a progress bar for monitoring
        with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for xb, yb in pbar:
                # Move the input and target tensors to the active device
                xb, yb = xb.to(device), yb.to(device)
                # Reset gradients of the model parameters to zero
                optimizer.zero_grad()
                # Perform the forward pass to obtain logits and attention weights
                logits, _ = model(xb)
                # Calculate the loss based on model predictions and actual targets
                loss = loss_fn(logits, yb)
                # Execute the backward pass to calculate gradients
                loss.backward()
                # Update model parameters based on the computed gradients
                optimizer.step()
                # Accumulate the weighted loss for the current batch
                total_loss += loss.item() * xb.size(0)
                # Display the loss value for the current batch in the progress bar
                pbar.set_postfix(loss=loss.item())

        # Calculate the mean loss across all samples in the dataset
        avg_loss = total_loss / len(loader.dataset)
        # Log the performance summary for the completed epoch
        print(f"Epoch {epoch+1:2d}: avg loss = {avg_loss:.4f}")


# Usage:
EPOCHS = 5
vocab_size = len(vocab)  # Pass this in!
train_model(model, loader, loss_fn, optimizer, epochs=EPOCHS, device=device, vocab_size=vocab_size)


# A helper class to generate next words from a model with temperature sampling.

class NextWordGenerator:
    """
    Generate next tokens from a language model using a fixed context window and temperature sampling.

    Uses left-padding with a padding token to fill the initial context window.
    Applies temperature scaling to logits, then samples with multinomial.
    Converts special newline tokens to standard newline characters in the final output.
    """

    def __init__(
        self,
        model,
        tokenizer,
        word2idx,
        idx2word,
        *,
        seq_len=6,
        device="cpu",
        pad_token="<pad>",
        unk_token="<unk>",
        nl_token="<nl>",
    ):
        """
        Initialize the NextWordGenerator instance.

        Arguments:
        model: The neural network model used for sequence generation.
        tokenizer: A callable function that converts a string into a sequence of string tokens.
        word2idx: A dictionary mapping string tokens to their integer indices.
        idx2word: A dictionary mapping integer indices back to string tokens.
        seq_len: The maximum number of tokens to keep in the context window.
        device: The hardware device where tensors will be allocated.
        pad_token: The string representation of the padding token.
        unk_token: The string representation of the unknown token.
        nl_token: The string representation of the newline token.
        """
        # Assign the model to the instance variables
        self.model = model
        # Store the tokenizer function
        self.tokenizer = tokenizer
        # Store the mapping from words to indices
        self.word2idx = word2idx
        # Store the mapping from indices to words
        self.idx2word = idx2word
        # Define the context window length
        self.seq_len = seq_len
        # Define the processing device
        self.device = device
        # Define the token used for sequence padding
        self.pad_token = pad_token
        # Define the token used for unknown words
        self.unk_token = unk_token
        # Define the token representing a newline
        self.nl_token = nl_token

        # Cache the integer ID for the unknown token to prevent repeated dictionary lookups
        self._unk_id = self.word2idx[self.unk_token]

    def _make_window(self, generated):
        """
        Create a fixed-size context window from the generated tokens.

        Arguments:
        generated: A list of string tokens that have been generated so far.

        Returns:
        A list of string tokens representing the padded or truncated context window.
        """
        # Check if the generated sequence has reached or exceeded the maximum sequence length
        if len(generated) >= self.seq_len:
            # Slice and return only the most recent tokens up to the allowed sequence length
            return generated[-self.seq_len:]
        # Calculate the deficit of tokens needed to fill the required sequence window
        pad_count = self.seq_len - len(generated)
        # Prepend the necessary amount of padding tokens to the sequence
        return [self.pad_token] * pad_count + generated

    def _encode(self, tokens):
        """
        Convert a sequence of string tokens into a tensor of corresponding integer indices.

        Arguments:
        tokens: A sequence of string tokens to be encoded.

        Returns:
        A tensor containing the encoded token indices.
        """
        # Look up the index for each token, defaulting to the unknown token ID if not found
        ids = [self.word2idx.get(t, self._unk_id) for t in tokens]
        # Construct and return a tensor of the indices on the specified processing device
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _decode_id(self, idx):
        """
        Convert a numerical token index back into its string representation.

        Arguments:
        idx: The integer index representing a specific token.

        Returns:
        The string representation of the token corresponding to the given index.
        """
        # Retrieve and return the string mapped to the specified index
        return self.idx2word[idx]

    def _postprocess(self, tokens):
        """
        Process the generated sequence of tokens to correctly format special characters.

        Arguments:
        tokens: A list of generated string tokens.

        Returns:
        A list of string tokens where special formatting tokens have been replaced.
        """
        # Replace custom newline tokens with actual newline characters for display or printing
        return [("\n" if t == self.nl_token else t) for t in tokens]

    def generate(self, sentence, *, max_tokens=20, temperature=1.0):
        """
        Generate a sequence of tokens continuing from the provided input sentence.

        Arguments:
        sentence: The initial string text to begin generation from.
        max_tokens: The maximum limit of new tokens to generate.
        temperature: A float value used to scale the logits before probability sampling.

        Returns:
        A list of string tokens containing the full generated sequence including the original tokens.
        """
        # Verify that the temperature parameter is strictly greater than zero
        if temperature <= 0:
            # Raise an exception if an invalid temperature value is provided
            raise ValueError("temperature must be > 0")

        # Switch the model to evaluation mode to ensure consistent generation behavior
        self.model.eval()
        # Tokenize the initial sentence into a list, preparing to mutate it in place without copying
        generated = list(self.tokenizer(sentence))

        # Temporarily disable gradient tracking to reduce memory usage during generation
        with torch.no_grad():
            # Loop over the allowed maximum number of tokens to generate
            for _ in range(max_tokens):
                # Retrieve the properly sized and padded context window for the current step
                window = self._make_window(generated)
                # Encode the context window tokens into a tensor of IDs
                input_ids = self._encode(window)

                # Feed the input tensor to the model to get raw logits, expecting a shape of [1, vocab]
                logits, _ = self.model(input_ids)
                # Scale the raw logits using the defined temperature variable
                logits = logits / temperature
                # Apply the softmax function to convert scaled logits into a probability distribution
                probs = torch.softmax(logits, dim=-1)

                # Sample the next token index based on the calculated probabilities, returning shape [1, 1]
                next_id = torch.multinomial(probs, num_samples=1).item()
                # Decode the chosen numeric index back into a string token
                next_token = self._decode_id(next_id)
                # Append the newly generated token to the sequence list
                generated.append(next_token)

        # Apply post-processing to the sequence and return the final list of tokens
        return self._postprocess(generated)
    
# Example usage:
generator = NextWordGenerator(
    model=model,
    tokenizer=tokenizer,
    word2idx=word2idx,
    idx2word=idx2word,
    seq_len=SEQ_LEN,
    device=device,
)

output = generator.generate("To be or not to", max_tokens=50, temperature=1.0)
print("Generated sequence:", " ".join(output))


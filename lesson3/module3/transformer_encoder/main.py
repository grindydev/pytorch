import math
import random
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

import helper_utils

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class EncoderBlock(nn.Module):
    """
    A single layer of a Transformer encoder.

    Args:
        d_model: The number of expected features in the input.
        nhead: The number of heads in the multi-head attention mechanism.
        ffn_mult: The multiplier used to determine the hidden layer size of the feed-forward network.
    """
    def __init__(self, d_model=4, nhead=1, ffn_mult=4):
        """
        Initializes the components of the encoder block.

        Args:
            d_model: The dimensionality of the input embeddings.
            nhead: The number of attention heads.
            ffn_mult: Expansion factor for the internal feed-forward layer.
        """
        super().__init__()
        # Normalization layer applied prior to the attention sub-layer
        self.ln1 = nn.LayerNorm(d_model)
        # Multi-head attention module for calculating contextual representations
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # Normalization layer applied prior to the feed-forward sub-layer
        self.ln2 = nn.LayerNorm(d_model)        
        # Calculate the dimensionality of the internal hidden layer
        hidden = ffn_mult * d_model
        # Sequential network providing non-linear transformations for each position
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model)
        )
    
    def forward(self, x):        
        """
        Processes the input sequence through the encoder block sub-layers.

        Args:
            x: The input tensor of shape [batch_size, sequence_length, d_model].

        Returns:
            The processed tensor after self-attention and feed-forward transformations.
        """
        # Apply normalization to the input sequence
        x_norm = self.ln1(x)
        # Compute self-attention and capture the output vectors
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        # Add the attention output to the original input for the first residual connection
        x = x + attn_out
        
        # Apply normalization to the output of the first sub-layer
        ffn_in = self.ln2(x)
        # Pass the normalized data through the feed-forward network
        ffn_out = self.ffn(ffn_in)
        # Add the feed-forward output to its input for the second residual connection
        x = x + ffn_out
        
        # Return the resulting contextualized representations
        return x


# Create a simple encoder block with small dimensions for demonstration
encoder_demo = EncoderBlock(d_model=4, nhead=1, ffn_mult=4)

# Create a sample input: (batch_size=2, sequence_length=3, d_model=4)
sample_input = torch.randn(2, 3, 4)

print("Input shape:", sample_input.shape)
print("Input tensor:\n", sample_input)

# Pass through encoder block
output = encoder_demo(sample_input)

print("\nOutput shape:", output.shape)
print("Output tensor:\n", output)

# Notice that the shape remains the same
print("\nShape preserved: Input shape == Output shape:", sample_input.shape == output.shape)

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding to inject sequence order into token embeddings.
    """
    def __init__(self, max_len, d_model):
        """
        Initializes the positional encoding matrix with precomputed values.

        Args:
            max_len (int): The maximum sequence length supported by this module.
            d_model (int): The dimensionality of the encoding vectors, matching the embedding size.
        """
        super().__init__()
        # Store the maximum sequence length capacity
        self.max_len = max_len
        # Store the feature dimension of the model
        self.d_model = d_model
        
        # Initialize a tensor of zeros to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        # Generate a vector of token positions from 0 to max_len
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Calculate the divisor term used for scaling frequencies in the sinusoids
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Assign sine values to the even indices of the encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        # Assign cosine values to the odd indices of the encoding matrix
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register the matrix as a persistent buffer that is not considered a learnable parameter
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Retrieves the positional encodings corresponding to the current input sequence length.

        Args:
            x (Tensor): The input token embeddings with shape [batch_size, seq_len, d_model].
            
        Returns:
            pe_slice (Tensor): The positional encoding tensor of shape [batch_size, seq_len, d_model].
        """
        # Determine the length of the sequence based on the input tensor dimensions
        seq_len = x.size(1)
        # Extract and return the subset of the precomputed matrix matching the input length
        return self.pe[:, :seq_len, :]
    


# Example: Create positional encoding and visualize
d_model = 128
max_len = 100
pos_encoder = PositionalEncoding(max_len=max_len, d_model=d_model)

# Create dummy embeddings for a batch of sequences
batch_size = 2
seq_len = 10
dummy_embeddings = torch.randn(batch_size, seq_len, d_model)

print(f"Input embeddings shape: {dummy_embeddings.shape}")
print(f"Input embeddings mean: {dummy_embeddings.mean():.4f}")

# Apply positional encoding
output = pos_encoder(dummy_embeddings)
print(f"\nOutput shape (unchanged): {output.shape}")
print(f"Output mean (slightly different): {output.mean():.4f}")

# Visualize the positional encoding pattern for first half of positions and dimensions
pe_matrix = pos_encoder.pe[0, :50, :64].numpy()
plt.figure(figsize=(10, 4))
plt.imshow(pe_matrix, cmap='RdBu', aspect='auto')
plt.colorbar()
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding Pattern (sin/cos waves)')
plt.show()


def create_padding_mask(seq, pad_idx=0):
    """
    Generates a boolean mask to identify padding tokens within a sequence.

    Args:
        seq (Tensor): Input sequence tensor of shape [batch_size, seq_len].
        pad_idx (int): The specific integer index used to represent padding.

    Returns:
        mask (Tensor): A boolean tensor of the same shape as the input, where 
                       True represents a padding token and False represents a 
                       valid token.
    """
    # Compare each element in the sequence to the padding index to create a boolean map
    return seq == pad_idx


# Example usage
# Sample batch with padding
batch = torch.tensor([
    [2, 15, 89, 234, 3, 0, 0, 0],   # 5 real tokens, 3 padding
    [2, 45, 67, 89, 123, 234, 3, 0], # 7 real tokens, 1 padding
    [2, 56, 3, 0, 0, 0, 0, 0],       # 3 real tokens, 5 padding
])

# Create padding mask
padding_mask = create_padding_mask(batch, pad_idx=0)
print("Input batch shape:", batch.shape)
print("\nPadding mask:")
print(padding_mask)
print("\nTrue = padding position, False = real token")


data_dir=Path.cwd() / 'data/imdb_data'
# Extract zip file
helper_utils.extract_imdb_data(data_dir=data_dir)

# Load the dataset with default settings (2000 train, 500 test samples)
train_reviews, train_labels, test_reviews, test_labels = helper_utils.get_imdb_data(
    data_dir=data_dir,
    max_train_samples=2000, 
    max_test_samples=500
)


helper_utils.print_data_statistics(train_reviews, train_labels, test_reviews, test_labels)

class IMDBTokenizer:
    """
    A text processing class for building a vocabulary and encoding text sequences.
    """
    def __init__(self, vocab_size=10000):
        """
        Initializes the tokenizer with basic special tokens and frequency counters.

        Args:
            vocab_size (int): The maximum number of unique tokens allowed in the vocabulary.
        """
        self.vocab_size = vocab_size
        # Assign indices to standard special tokens for padding, unknowns, and sequence boundaries
        self.word_to_idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        # Create a reverse mapping for decoding indices back into strings
        self.idx_to_word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        # Initialize a counter to track the occurrences of each word in the dataset
        self.word_freq = Counter()
    
    def build_vocab(self, texts, min_freq=2):
        """
        Constructs the internal vocabulary mapping based on a collection of texts.

        Args:
            texts (list): A collection of raw strings used to populate the vocabulary.
            min_freq (int): The minimum occurrence threshold for a word to be included.

        Returns:
            None: Updates the internal state of the tokenizer.
        """
        print("Building vocabulary...")
        
        # Aggregate word counts across the entire provided text corpus
        for text in texts:
            # Split the text into processed word tokens
            words = self.tokenize(text)
            # Update the frequency counter with the discovered words
            self.word_freq.update(words)
        
        # Identify the most frequent tokens while accounting for reserved special indices
        most_common = self.word_freq.most_common(self.vocab_size - 4)
        
        # Assign unique indices to words meeting the frequency criteria
        idx = 4
        for word, freq in most_common:
            # Ensure the word appears often enough to be significant
            if freq >= min_freq:
                # Map the string to the current index
                self.word_to_idx[word] = idx
                # Map the current index back to the string
                self.idx_to_word[idx] = word
                # Increment the index for the next unique token
                idx += 1
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
    
    def tokenize(self, text):
        """
        Performs character-level cleaning and splits text into lowercase words.

        Args:
            text (str): The raw string to be cleaned and split.

        Returns:
            words (list): A list of filtered lowercase string tokens.
        """
        # Standardize text by converting all characters to lowercase
        text = text.lower()
        # Eliminate HTML elements and tags from the raw text
        text = re.sub(r'<.*?>', '', text)
        # Filter the string to retain only alphabetic characters and whitespace
        text = re.sub(r'[^a-z\s]', '', text)
        # Split the resulting string into a list of word tokens
        words = text.split()
        return words
        
    def encode(self, text, max_len=256):
        """
        Transforms a raw string into a fixed-length list of numerical indices.

        Args:
            text (str): The input text to be encoded.
            max_len (int): The required length of the output list.

        Returns:
            indices (list): A list of integers representing the sequence of tokens.
        """
        # Tokenize the input and truncate to ensure space for boundary markers
        words = self.tokenize(text)[:max_len-2]
        
        # Initialize the sequence with the start-of-sentence token index
        indices = [2]
        
        # Map each word to its vocabulary index or the unknown token index
        for word in words:
            # Check if the word exists in the built vocabulary
            if word in self.word_to_idx:
                # Use the assigned index for the known word
                indices.append(self.word_to_idx[word])
            else:
                # Use the specific index for unknown words
                indices.append(1)
        
        # Finalize the sequence content with the end-of-sentence token index
        indices.append(3)
        
        # Append padding tokens until the list reaches the specified maximum length
        while len(indices) < max_len:
            # Fill remaining spots with the padding index
            indices.append(0)
        
        # Return the resulting sequence adjusted to the exact requested length
        return indices[:max_len]
    

# Create tokenizer
tokenizer = IMDBTokenizer(vocab_size=5000)

# Build vocabulary from training reviews
tokenizer.build_vocab(train_reviews, min_freq=2)

# Test tokenizer with a sample sentence
sample_text = "This movie was absolutely fantastic! I loved every minute of it."
print("Original text:", sample_text)

# Tokenize the text
tokens = tokenizer.tokenize(sample_text)
print("\nTokenized:", tokens)

# Encode to indices
encoded = tokenizer.encode(sample_text, max_len=20)
print("\nEncoded (max_len=20):", encoded)

# Decode back to words to verify
decoded_words = [tokenizer.idx_to_word.get(idx, '<unk>') for idx in encoded]
print("\nDecoded words:", decoded_words)

# Show some vocabulary statistics
print(f"\nVocabulary Statistics:")
print(f"Total unique words in vocab: {len(tokenizer.word_to_idx)}")
print(f"Most common words: {tokenizer.word_freq.most_common(10)}")


class IMDBDataset(Dataset):
    """
    A custom Dataset class for handling IMDB movie reviews and sentiment labels.
    """
    def __init__(self, reviews, labels, tokenizer, max_len=256):
        """
        Initializes the dataset by encoding all reviews and storing them as tensors.

        Args:
            reviews (list): A list of strings containing the movie review texts.
            labels (list): A list of integers representing the sentiment (e.g., 0 or 1).
            tokenizer (object): An object used to convert text into numerical indices.
            max_len (int): The maximum sequence length allowed for each review.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        self.labels = []
        
        # Display the progress of text processing
        print(f"Processing {len(reviews)} reviews...")
        
        # Iterate through pairs of reviews and labels to perform encoding
        for review, label in zip(reviews, labels):
            # Encode the review text into a fixed-length numerical representation
            encoded = tokenizer.encode(review, max_len)
            # Append the resulting indices to the data container
            self.data.append(encoded)
            # Append the associated label to the labels container
            self.labels.append(label)
        
        # Transform the list of encoded reviews into a LongTensor for indexing
        self.data = torch.LongTensor(self.data)
        # Transform the list of labels into a LongTensor
        self.labels = torch.LongTensor(self.labels)
        
        # Output the dimensions of the final processed dataset
        print(f"Dataset created with shape: {self.data.shape}")
    
    def __len__(self):
        """
        Provides the total number of samples contained in the dataset.

        Returns:
            length (int): The total count of reviews in the dataset.
        """
        # Calculate the size based on the number of rows in the data tensor
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a single encoded review and its corresponding label by index.

        Args:
            idx (int): The integer index of the desired item.

        Returns:
            data_sample (Tensor): The numerical sequence representing the review.
            label_sample (Tensor): The integer label representing the sentiment.
        """
        # Return the specific encoded sequence and label at the requested index
        return self.data[idx], self.labels[idx]
    

# Create datasets
max_seq_length = 256  # Maximum sequence length

print("Creating training dataset...")
train_dataset = IMDBDataset(train_reviews, train_labels, tokenizer, max_len=max_seq_length)

print("\nCreating test dataset...")
test_dataset = IMDBDataset(test_reviews, test_labels, tokenizer, max_len=max_seq_length)

# Create data loaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nData loaders created:")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Examine a single batch
sample_batch, sample_labels = next(iter(train_loader))
print(f"\nSample batch shape: {sample_batch.shape}")
print(f"Sample labels shape: {sample_labels.shape}")
print(f"First sequence in batch (first 20 tokens): {sample_batch[0, :20].tolist()}")
print(f"Label for first sequence: {sample_labels[0].item()}")


class IMDBSentimentModelWithCustomEncoder(nn.Module):
    """
    A sentiment classification model utilizing a stack of custom Transformer encoder blocks.
    """
    def __init__(self, vocab_size, d_model=128, num_layers=2, max_len=512, dropout=0.1):
        """
        Initializes the sentiment classifier with embedding, encoding, and encoder layers.

        Args:
            vocab_size (int): The total number of unique tokens in the vocabulary.
            d_model (int): The dimensionality of the embedding and hidden state vectors.
            num_layers (int): The number of sequential encoder blocks to be instantiated.
            max_len (int): The maximum sequence length supported by the positional encoding.
            dropout (float): The probability of an element being zeroed for regularization.
        """
        super().__init__()
        
        # Internal reference for the model dimensionality
        self.d_model = d_model
        
        # Mapping layer for converting discrete token indices into continuous vectors
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Module for generating and providing positional information to the embeddings
        self.positional_encoding = PositionalEncoding(max_len, d_model)
        
        # Dropout layer applied to the combined embeddings for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Sequential list containing multiple instances of Transformer encoder blocks
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model=d_model, nhead=8, ffn_mult=4) 
            for _ in range(num_layers)
        ])
        
        # Output layer for projecting hidden states to classification scores
        self.classifier = nn.Linear(d_model, 2)
    
    def forward(self, x):
        """
        Executes the forward pass of the model to generate classification logits.

        Args:
            x (Tensor): Input tensor containing token indices of shape [batch_size, seq_len].

        Returns:
            output (Tensor): Classification logits of shape [batch_size, 2].
        """
        # Retrieve the embedding vectors for the input token indices and scale them
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Obtain the positional information corresponding to the input sequence length
        pos_encoding = self.positional_encoding(x)
        
        # Incorporate positional information into the scaled token embeddings
        x = x + pos_encoding
        
        # Apply dropout to the modified embedding vectors
        x = self.dropout(x)
        
        # Pass the sequence through the stack of encoder blocks sequentially
        for encoder_layer in self.encoder_layers:
            # Transform the sequence through self-attention and feed-forward sub-layers
            x = encoder_layer(x)
        
        # Aggregate the sequence information by calculating the mean across the time dimension
        x = x.mean(dim=1)
        
        # Map the aggregated sequence representation to output logits
        output = self.classifier(x)
        
        # Return the final prediction scores
        return output


# Get vocabulary size
vocab_size = len(tokenizer.word_to_idx)

# Create the model
model = IMDBSentimentModelWithCustomEncoder(
    vocab_size=vocab_size,
    d_model=128,
    num_layers = 2
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model created: IMDBSentimentModel")
print(f"Total learnable parameters: {total_params:,}")


# Create a simple input: batch of 2 sequences, each with 5 tokens
tiny_input = torch.tensor([
    [2, 45, 23, 67, 3],  # First review: <sos>, word45, word23, word67, <eos>
    [2, 12, 89, 34, 3]   # Second review: <sos>, word12, word89, word34, <eos>
])
print("=== Step-by-step Forward Pass ===\n")
print("Input (token indices):")
print(tiny_input)
print("Shape:", tiny_input.shape, "\n")

# Step through the model manually
with torch.no_grad():
    # Step 1: Embedding
    embedded = model.embedding(tiny_input)
    print("Step 1 - After embedding:")
    print("  Shape:", embedded.shape)
    print("  Each token is now a vector of size", embedded.shape[-1])
    
    # Step 2: Pass through encoder layers
    x = embedded
    print(f"\nStep 2 - Passing through {len(model.encoder_layers)} encoder layer(s):")
    for i, encoder_layer in enumerate(model.encoder_layers):
        x = encoder_layer(x)
        print(f"  After encoder layer {i+1}:")
        print(f"    Shape: {x.shape}")
        if i == 0:
            print("    Note: shape is unchanged, but representations are refined")
    
    encoded = x
    print(f"\nAfter all encoder layers:")
    print(f"  Final encoded shape: {encoded.shape}")
    print("  Each token now has a contextualized representation")
    
    # Step 3: Pooling
    pooled = encoded.mean(dim=1)
    print("\nStep 3 - After averaging across sequence:")
    print("  Shape:", pooled.shape)
    print("  Now we have one vector per review")
    
    # Step 4: Classification
    output = model.classifier(pooled)
    print("\nStep 4 - Final classification scores:")
    print("  Shape:", output.shape)
    print("  Raw scores (logits):", output)
    
    # Convert to probabilities
    probs = torch.softmax(output, dim=1)
    print("\n  Probabilities [negative, positive]:")
    for i, p in enumerate(probs):
        print(f"    Review {i}: [{p[0]:.3f}, {p[1]:.3f}]")


# Get a batch from our data loader
sample_input, sample_labels = next(iter(train_loader))

print("Testing with real data:")
print(f"Batch shape: {sample_input.shape}")

# Forward pass
with torch.no_grad():
    output = model(sample_input)

# Get predictions
predictions = torch.argmax(output, dim=1)
accuracy = (predictions == sample_labels).float().mean()

print(f"\nResults on this batch (untrained model):")
print(f"  Accuracy: {accuracy:.1%}")
print(f"\nFirst 5 predictions vs actual:")
print("="*70)

for i in range(5):
    pred_label = "Positive" if predictions[i] == 1 else "Negative"
    true_label = "Positive" if sample_labels[i] == 1 else "Negative"
    correct = "✓" if predictions[i] == sample_labels[i] else "✗"
    
    # Decode the tokens back to words (first 30 tokens for brevity)
    tokens = sample_input[i][:30].tolist()
    # Remove padding tokens (0s) from the end
    tokens = [t for t in tokens if t != 0]
    # Convert token IDs back to words
    words = [tokenizer.idx_to_word.get(token_id, '<unk>') for token_id in tokens]
    # Join into sentence
    sentence_preview = ' '.join(words[:15]) + '...'  # Show first 15 words
    
    print(f"\nSample {i}:")
    print(f"  Text: {sentence_preview}")
    print(f"  Predicted: {pred_label}, Actual: {true_label} {correct}")


# Loss function for classification
criterion = nn.CrossEntropyLoss()

# Optimizer - Adam with learning rate 0.001
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training setup:")
print(f"  Loss function: CrossEntropyLoss")
print(f"  Optimizer: Adam")
print(f"  Learning rate: {learning_rate}")
print(f"  Batch size: {batch_size}")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluate untrained model
model.eval()
correct = 0
total = 0

print("Evaluating untrained model...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

initial_accuracy = 100 * correct / total
print(f"Initial Test Accuracy (before training): {initial_accuracy:.2f}%")

# Train the model
history = helper_utils.train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=5,
    device=device  # Optional, will auto-detect if not provided
)

helper_utils.plot_training_history(history)

class Encoder(nn.Module):
    """
    A modular Transformer-based Encoder for sequence representation.
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, 
                 dim_feedforward=512, max_len=100, dropout=0.1):
        """
        Initializes the encoder components including embeddings and transformer layers.

        Args:
            vocab_size (int): The total number of unique tokens in the input vocabulary.
            d_model (int): The dimensionality of the token embeddings and hidden states.
            nhead (int): The number of heads in the multi-head attention mechanism.
            num_layers (int): The number of transformer encoder layers to stack.
            dim_feedforward (int): The dimensionality of the internal feed-forward network.
            max_len (int): The maximum sequence length for positional encodings.
            dropout (float): The dropout probability used for regularization.
        """
        super().__init__()
        # Store internal model hyperparameters
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        self.dropout_value = dropout
        
        # Layer to map token indices to dense vectors
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Module to inject sequence order information into the embeddings
        self.pos_enc = PositionalEncoding(max_len, d_model)
        
        # Regularization layer applied to the input embeddings
        self.dropout = nn.Dropout(dropout)
         
        # Configuration for a single transformer encoder layer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True 
        )
        # Stack of identical transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
    def forward(self, src):
        """
        Processes the input sequence through the encoder layers.

        Args:
            src (Tensor): Input token indices of shape [batch_size, seq_len].

        Returns:
            memory (Tensor): The final encoded representations of shape [batch_size, seq_len, d_model].
            padding_mask (Tensor): A boolean mask indicating padding locations of shape [batch_size, seq_len].
        """
        # Generate a boolean mask to identify padding tokens for the attention mechanism
        padding_mask = create_padding_mask(src, pad_idx=0)
        
        # Transform indices to vectors, SCALE THEM, and combine them with positional information
        src = (self.token_emb(src) * math.sqrt(self.d_model)) + self.pos_enc(src)
        
        # Apply dropout to the sum of token and positional embeddings
        src = self.dropout(src)
        
        # Pass the input through the transformer stack using the padding mask to ignore pad tokens
        memory = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        
        # Provide both the encoded features and the mask for subsequent layers or decoders
        return memory, padding_mask


# Initialize your encoder
encoder = Encoder(
    vocab_size=1000,
    d_model=128,
    nhead=8,
    num_layers=2,
    dim_feedforward=512,
    max_len=100,
    dropout=0.1
)
encoder.eval()

# Sample input
input_batch = torch.tensor([
    [2, 45, 23, 3, 0, 0],  # 4 tokens + 2 padding
    [2, 12, 89, 34, 56, 3]  # 6 tokens, no padding
])

print("Input:", input_batch)
print()

# Your encoder always returns both output and padding mask
with torch.no_grad():
    memory, padding_mask = encoder(input_batch)

print("Encoder returns:")
print(f"  memory shape: {memory.shape}")
print(f"  padding_mask shape: {padding_mask.shape}")
print(f"  padding_mask: {padding_mask}")
print()


pytorch_model = Encoder(vocab_size=vocab_size, d_model=128, max_len = 256, num_layers = 2).to(device)

helper_utils.print_summary(pytorch_model, vocab_size=vocab_size)

# Test with a sample batch
sample_input, sample_labels = next(iter(train_loader))
sample_input = sample_input.to(device)

output = pytorch_model(sample_input)
print(f"\nTest forward pass:")
print(f"  Input shape: {sample_input.shape}")
print(f"  Output shape: {output[0].shape}")
print("  Model works! ✓")

class IMDBClassifierWithPytorchEncoder(nn.Module):
    """
    A sentiment classification model built upon a Transformer-based encoder.
    """
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2, 
                 dim_feedforward=512, max_len=256, dropout=0.1):
        """
        Initializes the classifier by configuring the encoder and the output layer.

        Args:
            vocab_size (int): The total number of unique tokens in the vocabulary.
            d_model (int): The dimensionality of the token embeddings and hidden states.
            nhead (int): The number of heads in the multi-head attention mechanism.
            num_layers (int): The number of transformer encoder layers to stack.
            dim_feedforward (int): The dimensionality of the internal feed-forward network.
            max_len (int): The maximum sequence length allowed for positional encodings.
            dropout (float): The dropout probability used for regularization.
        """
        super().__init__()
        
        # Core encoding module used to extract contextual features from sequence data
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout
        )
        
        # Linear projection layer used to map encoded features to class scores
        self.classifier = nn.Linear(d_model, 2)
        
    def forward(self, x):
        """
        Processes the input sequence through the encoder and classifier layers.

        Args:
            x (Tensor): Input tensor of token indices with shape [batch_size, seq_len].

        Returns:
            logits (Tensor): The raw classification scores with shape [batch_size, 2].
        """
        # Retrieve the encoded sequence representations and the associated padding mask
        memory, padding_mask = self.encoder(x)
        
        # Perform global average pooling across the sequence dimension to aggregate information
        pooled = memory.mean(dim=1)
        
        # Pass the aggregated representation through the linear classifier to get logits
        logits = self.classifier(pooled)
        
        # Return the final classification scores
        return logits
    

# Create the simple classifier
pytorch_model = IMDBClassifierWithPytorchEncoder(
    vocab_size=vocab_size,
    d_model=128,
    num_layers=2,
    max_len=256
).to(device)

helper_utils.print_summary(pytorch_model)


# Loss function for classification
pytorch_criterion = nn.CrossEntropyLoss()

# Optimizer - Adam with learning rate 0.001
learning_rate = 0.001
pytorch_optimizer = optim.Adam(pytorch_model.parameters(), lr=learning_rate)

print("Training setup:")
print(f"  Loss function: CrossEntropyLoss")
print(f"  Optimizer: Adam")
print(f"  Learning rate: {learning_rate}")
print(f"  Batch size: {batch_size}")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# Now train
EPOCHS = 5
pytorch_history = helper_utils.train_model(
    pytorch_model,
    train_loader,
    test_loader,
    pytorch_optimizer,
    pytorch_criterion,
    num_epochs=EPOCHS
)

helper_utils.compare_models(history, pytorch_history)

def predict_sentiment(text, model, tokenizer, device):
    """
    Performs sentiment inference on a single string input.

    Args:
        text (str): The raw input text string to be analyzed.
        model (nn.Module): The trained neural network model used for prediction.
        tokenizer (object): The tokenizer instance used to encode the text.
        device (str or torch.device): The computation hardware (e.g., 'cpu' or 'cuda').

    Returns:
        sentiment (str): A string label indicating the predicted class ("Positive" or "Negative").
        confidence (float): The probability score associated with the predicted class.
        probabilities (Tensor): The full probability distribution across all classes.
    """
    # Set the model to evaluation mode to disable layers like dropout
    model.eval()
    
    # Transform the raw text into a numerical sequence of indices
    encoded = tokenizer.encode(text, max_len=256)
    # Convert the list of indices into a batch-oriented LongTensor on the target device
    input_tensor = torch.LongTensor([encoded]).to(device)
    
    # Disable gradient calculation for efficient inference
    with torch.no_grad():
        # Pass the input tensor through the model to obtain raw output scores
        output = model(input_tensor)
        # Apply the softmax function to normalize outputs into a probability distribution
        probabilities = torch.softmax(output, dim=1)
        # Identify the class index with the highest score
        prediction = torch.argmax(output, dim=1)
    
    # Extract the probability value corresponding to the predicted class
    confidence = probabilities[0][prediction].item()
    # Map the predicted numerical index to a human-readable sentiment label
    sentiment = "Positive" if prediction.item() == 1 else "Negative"
    
    # Return the label, the confidence score, and the complete probability tensor
    return sentiment, confidence, probabilities[0]

# Test reviews
test_reviews = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible film. Complete waste of time. I want my money back.",
    "Not bad, but not great either. It was okay I guess.",
    "One of the best films I've ever seen. Brilliant acting and amazing story!",
    "Boring and predictable. I fell asleep halfway through."
]

print("="*60)
print("TESTING BOTH MODELS WITH SAMPLE REVIEWS")
print("="*60)

for i, review in enumerate(test_reviews, 1):
    print(f"\nReview {i}: \"{review[:50]}...\"" if len(review) > 50 else f"\nReview {i}: \"{review}\"")
    print("-"*40)
    
    # Test with encoder from scratch
    sentiment, confidence, probs = predict_sentiment(review, model, tokenizer, device)
    print(f"Encoder from Scratch: {sentiment} (confidence: {confidence:.2%})")
    print(f"  [Negative: {probs[0]:.3f}, Positive: {probs[1]:.3f}]")
    
    # Test with PyTorch implemented encoder
    sentiment_pt, confidence_pt, probs_pt = predict_sentiment(review, pytorch_model, tokenizer, device)
    print(f"PyTorch Implemented Encoder: {sentiment_pt} (confidence: {confidence_pt:.2%})")
    print(f"  [Negative: {probs_pt[0]:.3f}, Positive: {probs_pt[1]:.3f}]")


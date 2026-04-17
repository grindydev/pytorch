"""
Lesson 3 - Module 3: Self-Attention Mechanism (self_attention/main.py)
======================================================================
WHAT YOU'LL LEARN:
  * What self-attention is and why it powers modern NLP (GPT, BERT, etc.)
  * Tokenization: converting text to integer sequences
  * Building a vocabulary from a corpus
  * Creating a sliding-window next-token prediction dataset
  * The three key components: Query (Q), Key (K), Value (V)
  * Scaled dot-product attention: softmax(QK^T / sqrt(d)) * V
  * Training a self-attention model to predict the next word

KEY CONCEPT:
  SELF-ATTENTION allows each word in a sequence to "look at" all other words
  and decide which ones are most relevant. The mechanism:
    1. Each word produces three vectors: Query (what am I looking for?),
       Key (what do I contain?), Value (what information do I provide?)
    2. Compute similarity scores: Q dot K for every pair of words
    3. Scale by 1/sqrt(d) to prevent gradient issues
    4. Apply softmax to get attention weights (they sum to 1)
    5. Multiply weights by V to get the output for each position

  This is the core operation in Transformers -- the "T" in GPT and BERT.
"""

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


# ==================== PART 1: TEXT PREPROCESSING ====================
# Before any NLP model, we need to convert raw text into numbers.

sentences = """
the dog chased the cat
the cat chased the mouse
the dog ran fast
the mouse ran fast
the cat lay down
"""


# --- Tokenizer: splits text into individual words ---
class SimpleTokenizer:
    """
    Splits text into lowercase word tokens.

    KEY CONCEPT: Tokenization is the first step in any NLP pipeline.
    "The Dog chased the Cat" -> ["the", "dog", "chased", "the", "cat"]
    """
    def __call__(self, text):
        # re.findall extracts only word characters, ignoring punctuation
        return re.findall(r'\b\w+\b', text.lower())


# --- Vocabulary builder ---
def build_vocab(sentences, tokenizer, min_freq=1):
    """
    Builds vocabulary and word<->index mappings from text.

    KEY CONCEPT: Neural networks process NUMBERS, not text.
    We map each unique word to an integer index:
      "the" -> 2, "dog" -> 3, "cat" -> 4, etc.

    Special tokens:
      <pad>: Padding token (used to make sequences the same length)
      <unk>: Unknown token (for words not in vocabulary)
    """
    counter = Counter()
    for sent in sentences:
        counter.update(tokenizer(sent))

    vocab = ['<pad>', '<unk>'] + [w for w, c in counter.items() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}

    return vocab, word2idx, idx2word


sentences = [
    "the dog chased the cat",
    "the cat chased the mouse",
    "the dog ran fast",
    "the mouse ran fast",
    "the cat lay down"
]

tokenizer = SimpleTokenizer()
vocab, word2idx, idx2word = build_vocab(sentences, tokenizer)
print("Vocab:", vocab)


# ==================== PART 2: CREATE TRAINING DATASET ====================
# KEY CONCEPT: Next-token prediction. Given a sequence of words, predict
# the next word. This is the same task GPT is trained on (but much simpler).

SEQ_LEN = 4  # How many words to look at to predict the next one

# Convert sentences to token ID sequences
encoded_sentences = []
for sent in sentences:
    tokens = tokenizer(sent)
    ids = [word2idx.get(tok, word2idx['<unk>']) for tok in tokens]
    encoded_sentences.append(ids)

# Create sliding window input-target pairs
# Example: "the dog chased the cat" with SEQ_LEN=4 produces:
#   Input: [the, dog, chased, the]  ->  Target: cat
inputs = []
targets = []
for ids in encoded_sentences:
    for i in range(len(ids) - SEQ_LEN):
        window = ids[i:i + SEQ_LEN]    # Context window
        target = ids[i + SEQ_LEN]       # Next word to predict
        inputs.append(window)
        targets.append(target)

# Show the dataset in human-readable form
for inp, tgt in zip(inputs, targets):
    inp_words = [idx2word[i] for i in inp]
    tgt_word = idx2word[tgt]
    print(f"Input: {inp_words}  ->  Target: {tgt_word}")


class TinyDataset(Dataset):
    """Wraps input-target pairs into a PyTorch Dataset."""
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


dataset = TinyDataset(inputs, targets)


# ==================== PART 3: SELF-ATTENTION MODEL ====================
# This is the heart of the Transformer architecture.

class SelfAttentionModel(nn.Module):
    """
    A model that uses self-attention to predict the next word.

    ARCHITECTURE:
      1. Embedding: Convert word indices to dense vectors
      2. Positional Encoding: Add position information (which word is where)
      3. Self-Attention: Each word attends to all other words
      4. Feed-Forward: Process attended features
      5. Output: Predict next word from vocabulary

    KEY MATH (Scaled Dot-Product Attention):
      Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

      - Q (Query): "What am I looking for?"  -- shape: [seq_len, d_model]
      - K (Key):   "What do I contain?"       -- shape: [seq_len, d_model]
      - V (Value): "What info do I provide?"   -- shape: [seq_len, d_model]

      The QK^T dot product computes how much each word should attend to
      every other word. Dividing by sqrt(d_k) prevents the dot products
      from being too large (which would make softmax gradients vanish).
    """

    def __init__(self, vocab_size, d_model=32, n_heads=2, max_len=32):
        super().__init__()

        # Embedding layer: maps word indices to dense vectors
        # KEY CONCEPT: Unlike GloVe (fixed vectors), these embeddings are
        # learned during training for this specific task.
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding: tells the model WHERE each word is in the sequence
        # KEY CONCEPT: Self-attention is permutation-invariant by default
        # (it treats input as a bag of words). Positional encoding breaks this
        # symmetry so the model knows word order matters.
        self.pos_encoding = nn.Embedding(max_len, d_model)

        # Multi-head self-attention
        # KEY CONCEPT: Multiple "heads" allow the model to attend to different
        # aspects of the input simultaneously (e.g., one head focuses on
        # syntax, another on semantics).
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Feed-forward network: processes each position independently
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # Expand
            nn.GELU(),                         # Non-linearity
            nn.Linear(d_model * 4, d_model),  # Compress back
        )

        # Layer normalization: stabilizes training
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Output head: maps to vocabulary size for next-word prediction
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        # 1. Embedding + Positional Encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)  # [batch, seq_len, d_model]

        # 2. Self-Attention with residual connection + layer norm
        # KEY CONCEPT: Residual connection (adding input back) helps gradients
        # flow through deep networks. LayerNorm normalizes across features.
        attn_out, _ = self.attention(x, x, x)  # Q=K=V=x (self-attention!)
        x = self.ln1(x + attn_out)              # Residual + LayerNorm

        # 3. Feed-Forward with residual connection
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)               # Residual + LayerNorm

        # 4. Output: predict next word from the LAST position
        x = self.head(x[:, -1, :])               # [batch, vocab_size]
        return x


# ==================== PART 4: TRAINING ====================
# Standard training loop -- same pattern as all previous modules.

vocab_size = len(vocab)
model = SelfAttentionModel(vocab_size, d_model=32, n_heads=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model.train()
for epoch in range(200):
    total_loss = 0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)        # [batch, vocab_size]
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 40 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

# ==================== PART 5: INFERENCE ====================
# Test: given a sequence of words, predict what comes next
model.eval()
with torch.no_grad():
    test_sentence = "the dog chased the"
    tokens = tokenizer(test_sentence)
    ids = [word2idx.get(t, word2idx['<unk>']) for t in tokens][-SEQ_LEN:]
    input_tensor = torch.tensor([ids])

    output = model(input_tensor)
    predicted_idx = output.argmax(dim=1).item()
    predicted_word = idx2word[predicted_idx]

    print(f"\nInput: '{test_sentence}'")
    print(f"Predicted next word: '{predicted_word}'")

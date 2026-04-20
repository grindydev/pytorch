"""
Lesson 3 - Module 3: Machine Translation with Encoder-Decoder Transformer
==========================================================================
WHAT YOU'LL LEARN:
  * Full encoder-decoder Transformer architecture for seq2seq tasks
  * Building vocabularies for source and target languages
  * Cross-attention: decoder attending to encoder outputs
  * Training a translation model from English to a target language
  * The complete translation pipeline: tokenize -> encode -> decode -> detokenize

KEY CONCEPT:
  MACHINE TRANSLATION uses an encoder-decoder architecture:
    1. ENCODER: processes the source sentence (English) into contextual representations
    2. DECODER: generates the target sentence word by word, attending to both:
       - Its own previous outputs (masked self-attention)
       - The encoder's representations (cross-attention)

  This is the architecture from the original "Attention Is All You Need" paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

import numpy as np
from pathlib import Path

# For data handling
from collections import Counter

import helper_utils
import unittests

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

languages_dir = Path.cwd() / 'data/languages'
translation_pairs, target_language = helper_utils.load_dataset(languages_dir=languages_dir)

normalized_pairs, tokenizer = helper_utils.prepare_data(
    translation_pairs, 
    target_language,
    max_pairs=150000,  # Process first 150,000 pairs for faster training
    max_length=40      # Keep sentences with <= 40 words
)

# Cell 4: Check some normalized pairs
import random

print(f"\nRandom normalized {target_language} pairs:")
random_samples = random.sample(normalized_pairs, min(3, len(normalized_pairs)))
for eng, target in random_samples:
    print(f"EN: {eng}")
    print(f"{target_language}: {target}")
    print("-" * 40)

# Cell 5: Use the tokenizer on custom text
custom_text = "I love programming!"
tokens = tokenizer(custom_text)
print(f"\nCustom text: {custom_text}")
print(f"Tokens: {tokens}")


def build_vocab(sentences, tokenizer, min_freq=1):
    """
    Build vocabulary from sentences
    """
    counter = Counter()  # Counter to count word frequencies in all sentences
    for sent in sentences:
        counter.update(tokenizer(sent))  # Tokenize sentence and add token counts
    
    # Start vocab with special tokens for translation
    # <pad>: padding token, <unk>: unknown token, <sos>: start of sequence, <eos>: end of sequence
    vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + [w for w, c in counter.items() if c >= min_freq]
    
    # Create a mapping from word to unique index
    word2idx = {w: i for i, w in enumerate(vocab)}
    # Create a mapping from index back to word (inverse of word2idx)
    idx2word = {i: w for i, w in enumerate(vocab)}
    
    # Return the vocab list and the two dictionaries
    return vocab, word2idx, idx2word

# Extract English and target language sentences separately
eng_sentences = [eng for eng, tgt in normalized_pairs]
tgt_sentences = [tgt for eng, tgt in normalized_pairs]

# Build vocabularies for both languages
print("Building English vocabulary...")
eng_vocab, eng_word2idx, eng_idx2word = build_vocab(eng_sentences, tokenizer, min_freq=2)
print(f"English vocab size: {len(eng_vocab)}")
print(f"First 20 English vocab words: {eng_vocab[:20]}")

print(f"\nBuilding {target_language} vocabulary...")
tgt_vocab, tgt_word2idx, tgt_idx2word = build_vocab(tgt_sentences, tokenizer, min_freq=2)
print(f"{target_language} vocab size: {len(tgt_vocab)}")
print(f"First 20 {target_language} vocab words: {tgt_vocab[:20]}")

def prepare_sequence(sentence, tokenizer, word2idx, max_length=20, add_special_tokens=True):
    """
    Convert a sentence to a list of indices with special tokens
    """
    tokens = tokenizer(sentence)
    
    if add_special_tokens:
        # Add <sos> at the beginning and <eos> at the end
        tokens = ['<sos>'] + tokens + ['<eos>']
    
    # Convert tokens to indices
    indices = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
    
    # Pad or truncate to max_length
    if len(indices) < max_length:
        # Pad with <pad> tokens
        indices = indices + [word2idx['<pad>']] * (max_length - len(indices))
    else:
        # Truncate if too long
        indices = indices[:max_length]
    
    return indices


# Prepare all translation pairs
MAX_LENGTH = 40
prepared_pairs = []

for eng, tgt in normalized_pairs:
    # Prepare source (English) - no special tokens for encoder input
    eng_tokens = tokenizer(eng)
    eng_indices = [eng_word2idx.get(token, eng_word2idx['<unk>']) for token in eng_tokens]
    
    # Pad or truncate
    if len(eng_indices) < MAX_LENGTH:
        eng_indices = eng_indices + [eng_word2idx['<pad>']] * (MAX_LENGTH - len(eng_indices))
    else:
        eng_indices = eng_indices[:MAX_LENGTH]
    
    # Prepare target - with special tokens for decoder
    tgt_indices = prepare_sequence(tgt, tokenizer, tgt_word2idx, MAX_LENGTH, add_special_tokens=True)
    
    prepared_pairs.append((eng_indices, tgt_indices))

print(f"Number of prepared pairs: {len(prepared_pairs)}")

# Show an example pair
example_idx = 0
eng_indices, tgt_indices = prepared_pairs[example_idx]

print("\nExample prepared pair:")
print(f"Original English: {normalized_pairs[example_idx][0]}")
print(f"English tokens: {[eng_idx2word[i] for i in eng_indices if i != eng_word2idx['<pad>']]}")
print(f"English indices: {eng_indices}")

print(f"\nOriginal {target_language}: {normalized_pairs[example_idx][1]}")
print(f"{target_language} tokens: {[tgt_idx2word[i] for i in tgt_indices if i != tgt_word2idx['<pad>']]}")
print(f"{target_language} indices: {tgt_indices}")


from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for translation pairs
    """
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_indices, tgt_indices = self.pairs[idx]
        
        # Convert to tensors
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
        
        return src_tensor, tgt_tensor
    

import torch
from torch.utils.data import DataLoader, Subset

# Create the full dataset
full_dataset = TranslationDataset(prepared_pairs)

# Determine the total number of samples in the dataset
total_size = len(full_dataset)

# Set the seed for reproducibility and generate shuffled indices directly
torch.manual_seed(42)
indices = torch.randperm(total_size).tolist()

# Calculate split point
split_point = int(0.9 * total_size)

# Create train and validation indices
train_indices = indices[:split_point]
val_indices = indices[split_point:]

# Create subset datasets
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

print(f"Training pairs: {len(train_dataset)}")
print(f"Validation pairs: {len(val_dataset)}")

# Create data loaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Test the data loader
for src_batch, tgt_batch in train_loader:
    print(f"Source batch shape: {src_batch.shape}")
    print(f"Target batch shape: {tgt_batch.shape}")
    # Show first example from batch
    print(f"\nFirst example in batch:")
    src_tokens = [eng_idx2word[idx.item()] for idx in src_batch[0] if idx.item() != eng_word2idx['<pad>']]
    tgt_tokens = [tgt_idx2word[idx.item()] for idx in tgt_batch[0] if idx.item() != tgt_word2idx['<pad>']]
    print(f"Source (English): {' '.join(src_tokens)}")
    print(f"Target ({target_language}): {' '.join(tgt_tokens)}")
    break


def create_padding_mask(seq, pad_idx=0):
    """
    Create a mask to hide padding tokens
    Args:
        seq: Input sequence tensor [batch_size, seq_length]
        pad_idx: Index used for padding (usually 0)
    Returns:
        Boolean mask where True = ignore this position
    """
    return (seq == pad_idx)

padded_seq = create_padding_mask(np.array([34, 67, 0, 0, 0]))
print(padded_seq)

def make_causal_mask(size):
    """
    Create a mask to hide future tokens (for decoder self-attention)
    Args:
        size: Sequence length
    Returns:
        Upper triangular matrix where True = ignore this position
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

# Example: Decoder input with padding
decoder_input = ['<sos>', 'Hello', 'world', '<pad>', '<pad>']
indices = [2, 34, 67, 0, 0]

# Create both masks
padding_mask = create_padding_mask(indices)  # [F, F, F, T, T]
subsequent_mask = make_causal_mask(5)   # Upper triangular matrix
print(subsequent_mask)


class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings using sinusoidal patterns.
    
    Since transformers don't have inherent notion of sequence order (unlike RNNs),
    we add positional encodings to give the model information about where each
    token appears in the sequence.
    """
    def __init__(self, max_len, d_model):
        """
        Initialize positional encoding matrix.
        
        Args:
            max_len (int): Maximum sequence length the model will handle
                          (e.g., 100 for sentences up to 100 tokens)
            d_model (int): Dimension of the model's embeddings 
                          (e.g., 256 or 512 - must match embedding size)
        
        Creates a fixed sinusoidal pattern matrix of shape [max_len, d_model]
        where each row represents the positional encoding for that position.
        """
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trained, but saved with model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (Tensor): Token embeddings of shape [batch_size, seq_len, d_model]
                       where seq_len <= max_len from initialization
        
        Returns:
            Tensor: Positional encodings of shape [batch_size, seq_len, d_model]
                   (same shape as input, ready to be added to embeddings)
        
        Example:
            If x represents embeddings for "I love cats" (3 tokens):
            - Input x shape: [batch_size, 3, 256]
            - Output shape: [batch_size, 3, 256]
            - Returns positions 0, 1, 2 encoded as 256-dim vectors
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]
    



# GRADED CELL  

class Encoder(nn.Module):
    """
    Encoder: Processes the source language (English) and creates a context representation
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, 
                 dim_feedforward=512, max_len=100, dropout=0.1):
        super().__init__()

        ### START CODE HERE ###
        
        # Token embedding: Converts word indices to vectors with input dimension vocab_size, output dimension d_model and the padding_idx being the index 0
        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0)
        
        # Generate the positional encoding, by calling the previous defined layer PositionalEncoding, with max length given by max_len and d_model being the embedding dimension
        self.pos_enc = PositionalEncoding(max_len, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) 
         
        # Define the encoder, by calling nn.TransformerEncoderLayer with the appropriate parameters. Do not forget th pass the dropout value! Include the parameter batch_first = None
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

        # Set the transformer encoder layer by calling the nn.TransformerEncoder with the encoder_layer and the number of layers (given by the parameter)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=num_layers)

        ### END CODE HERE ###
        
    def forward(self, src):
        """
        Args:
            src: Source language token indices [batch_size, seq_len]
        Returns:
            memory: Encoded representation [batch_size, seq_len, d_model]
        """
        ### START CODE HERE ###
        
        # Create padding mask
        padding_mask = create_padding_mask(seq=src, pad_idx=0)
        
        # Embed tokens and add positional encoding
        src = self.token_emb(src) + self.pos_enc(src)
        
        # Perform dropout using the dropout layer defined above.
        src = self.dropout(src)
        
        # Pass through transformer encoder, by generating the contextual representation and the padding mask, passed in the argument called src_key_padding_mask
        memory = self.transformer_encoder(src, src_key_padding_mask=padding_mask)

        ### END CODE HERE ###
        
        return memory, padding_mask


# Usage
encoder = Encoder(
                vocab_size=5000,
                d_model=256,
                nhead=8,
                num_layers=3,
                dim_feedforward=512,
                max_len=100,
                dropout=0.1
)


helper_utils.show_model_layers(encoder)

unittests.exercise_1(Encoder)


#### How This Decoder Differs from Lab 3

# | Aspect | Lab 3 Decoder (Generation) | Assignment Decoder (Translation) |
# |--------|---------------------------|----------------------------------|
# | **Primary Task** | Text generation (Shakespeare) | Translation (English → French) |
# | **Input Source** | Only previous tokens | Previous tokens + Encoder memory |
# | **Attention Types** | Self-attention only | Self-attention + Cross-attention |
# | **Key Component** | `TransformerEncoder` with masking | `TransformerDecoder` with memory |
# | **Memory Input** |  None |  Encoder outputs |


# GRADED CELL  

class Decoder(nn.Module):
    """
    Decoder component for translation (works with encoder output)
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3,
                 dim_feedforward=512, max_len=100, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        
        ### START CODE HERE ###
        
        # Token embedding: Converts target word indices to vectors with input dimension vocab_size, output dimension d_model and the padding_idx being the index 0
        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0)
        
        # Generate the positional encoding for target sequence
        self.pos_enc = PositionalEncoding(max_len=max_len, d_model=d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) 
        
        # Define the decoder layer with the desired parameters
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Set the transformer decoder by passing the decoder layer and the number of layers
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=dec_layer, num_layers=num_layers)
        
        # Output projection layer: Projects decoder output to target vocabulary size, it must input the d_model and output vocab_size
        self.output_projection = nn.Linear(in_features=d_model, out_features=vocab_size)
        
        ### END CODE HERE ###
        
    def forward(self, tgt, memory, memory_padding_mask=None):
        """
        Args:
            tgt: Target language token indices [batch_size, tgt_seq_len]
            memory: Encoder output [batch_size, src_seq_len, d_model]
            memory_padding_mask: Mask for encoder padding [batch_size, src_seq_len]
        Returns:
            output: Predicted token logits [batch_size, tgt_seq_len, vocab_size]
        """
        ### START CODE HERE ###
        
        # Create padding mask for target sequence
        tgt_padding_mask = create_padding_mask(tgt, pad_idx=0)
        
        # Create subsequent mask to prevent decoder from looking at future tokens
        tgt_seq_len = tgt.size(1)
        tgt_subsequent_mask = make_causal_mask(tgt_seq_len).to(tgt.device)

        # Convert into token embeddings
        tgt = self.token_emb(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding so model knows word positions
        tgt = tgt + self.pos_enc(tgt)
        
        # Apply dropout to embedded target
        tgt = self.dropout(tgt)
        
        # Pass through transformer decoder with cross-attention to encoder memory
        decoded = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_subsequent_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        # Project decoder output to vocabulary size
        output = self.output_projection(decoded)
        
        ### END CODE HERE ###
        
        return output
    

decoder = Decoder(vocab_size=5000, d_model=256, nhead=8, num_layers=3)
helper_utils.show_decoder_layers(decoder)

unittests.exercise_2(Decoder)



# GRADED CELL  

class EncoderDecoder(nn.Module):
    """
    Complete Encoder-Decoder translation model combining encoder and decoder modules
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8,
                 num_enc_layers=3, num_dec_layers=3, dim_feedforward=512,
                 max_len=100, dropout=0.1):
        super().__init__()
        
        ### START CODE HERE ###
        
        # Initialize encoder for source language with source vocabulary size
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_enc_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout
        )
        
        # Initialize translation decoder for target language with target vocabulary size  
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_dec_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout
        )
        
        ### END CODE HERE ###
        
    def forward(self, src, tgt):
        """
        Args:
            x: Source language token indices [batch_size, src_seq_len]
            tgt: Target language token indices [batch_size, tgt_seq_len]
        Returns:
            output: Predicted token logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        ### START CODE HERE ###
        
        # Encode the source sequence to get memory and source padding mask
        memory, src_padding_mask = self.encoder(src)
        
        # Decode using encoder memory to generate target sequence predictions
        output = self.decoder(tgt, memory, src_padding_mask)
        
        ### END CODE HERE ###
        
        return output
    

# Create EncoderDecoder model
model = EncoderDecoder(
    src_vocab_size=5000, 
    tgt_vocab_size=5000, 
    d_model=256, 
    nhead=8, 
    num_enc_layers=3,
    num_dec_layers=3
).to(device)

# Show the summary
helper_utils.show_encoderdecoder_layers(model)

unittests.exercise_3(EncoderDecoder, Encoder, Decoder)


# Model hyperparameters (in section 4.4)
D_MODEL = 256
NHEAD = 8
NUM_ENC_LAYERS = 3
NUM_DEC_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Create the model with dynamic vocabulary sizes
model = EncoderDecoder(
    src_vocab_size=len(eng_vocab),
    tgt_vocab_size=len(tgt_vocab),  # Uses target language vocab
    d_model=D_MODEL,
    nhead=NHEAD,
    num_enc_layers=NUM_ENC_LAYERS,
    num_dec_layers=NUM_DEC_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    max_len=MAX_LENGTH,
    dropout=DROPOUT
).to(device)

helper_utils.show_encoderdecoder_layers(model)

# Test the model with a sample batch
for src_batch, tgt_batch in train_loader:
    src_batch = src_batch.to(device)
    tgt_batch = tgt_batch.to(device)
    
    # Use all but last token as input to decoder
    tgt_input = tgt_batch[:, :-1]
    
    # Forward pass
    output = model(src_batch, tgt_input)
    
    print(f"Source shape: {src_batch.shape}")
    print(f"Target input shape: {tgt_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dimension matches target vocabulary: {output.shape[-1] == len(tgt_vocab)}")
    
    break


# Initialize optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

print(f"Training setup:")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss function: CrossEntropyLoss")
print(f"  Training batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print("-" * 50)


# Train the model (increase number of epochs to get better results but longer training time)
NUM_EPOCHS = 2

print("Starting training...")
history = helper_utils.train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=NUM_EPOCHS
)

print("Training completed!")

helper_utils.plot_training_history(history)



def translate_sentence(model, sentence, src_word2idx, tgt_idx2word, tokenizer, max_length=20, temperature=1.0, debug=False):
    """
    Translate a single sentence using greedy decoding with optional temperature sampling
    """
    model.eval()
    
    # Create reverse mapping for target vocabulary
    tgt_word2idx = {word: idx for idx, word in tgt_idx2word.items()}
    
    # Tokenize and convert to indices
    tokens = tokenizer(sentence.lower())
    src_indices = [src_word2idx.get(token, src_word2idx['<unk>']) for token in tokens]
    
    if debug:
        print(f"Input tokens: {tokens}")
        print(f"Input indices: {src_indices}")
    
    # Pad source to max_length
    if len(src_indices) < max_length:
        src_indices = src_indices + [src_word2idx['<pad>']] * (max_length - len(src_indices))
    else:
        src_indices = src_indices[:max_length]
    
    # Convert to tensor and add batch dimension
    src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
    
    # Generate translation using autoregressive decoding
    with torch.no_grad():
        # Get encoder output and source padding mask
        encoder_memory, src_padding_mask = model.encoder(src_tensor)
        
        if debug:
            print(f"Encoder memory shape: {encoder_memory.shape}")
            print(f"Source padding mask shape: {src_padding_mask.shape}")
        
        # Start with <sos> token
        tgt_indices = [tgt_word2idx['<sos>']]
        
        for step in range(max_length - 1):
            # Create target tensor with only tokens generated so far
            tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
            
            # Pass to decoder with encoder memory
            decoder_output = model.decoder(
                tgt_tensor, 
                memory=encoder_memory,
                memory_padding_mask=src_padding_mask
            )
            
            # Get prediction for the NEXT token (from last position)
            next_token_logits = decoder_output[0, -1, :]  # Last position predicts next token
            
            if debug and step < 3:
                print(f"Step {step}: Logits shape: {next_token_logits.shape}")
                print(f"Step {step}: Top 5 logits: {torch.topk(next_token_logits, 5)}")
            
            # Apply temperature for more diverse sampling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Use sampling instead of pure greedy for better diversity
            if temperature > 1.0:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(next_token_logits).item()
            
            if debug and step < 3:
                print(f"Step {step}: Selected token: {next_token} ({tgt_idx2word.get(next_token, 'UNK')})")
            
            # Add predicted token to sequence
            tgt_indices.append(next_token)
            
            # Stop if <eos> token is generated
            if next_token == tgt_word2idx['<eos>']:
                break
    
    # Convert indices to words (exclude special tokens)
    translated_tokens = [tgt_idx2word[idx] for idx in tgt_indices 
                        if idx not in [tgt_word2idx['<pad>'], tgt_word2idx['<sos>'], tgt_word2idx['<eos>']]]
    
    if debug:
        print(f"Final target indices: {tgt_indices}")
        print(f"Translated tokens: {translated_tokens}")
    
    return ' '.join(translated_tokens)

# Quick debug version
def debug_translate(model, sentence, src_word2idx, tgt_idx2word, tokenizer, max_length=20):
    """Debug version with temperature and verbose output"""
    print(f"\n DEBUG: Translating '{sentence}'")
    print("=" * 50)
    
    # Try with different temperatures
    print("\n1. Greedy (temperature=1.0):")
    result1 = translate_sentence(model, sentence, src_word2idx, tgt_idx2word, tokenizer, 
                               max_length, temperature=1.0, debug=True)
    print(f"Result: '{result1}'")
    
    print("\n2. With temperature=1.5:")
    result2 = translate_sentence(model, sentence, src_word2idx, tgt_idx2word, tokenizer, 
                               max_length, temperature=1.5, debug=False)
    print(f"Result: '{result2}'")
    
    print("\n3. With temperature=2.0:")
    result3 = translate_sentence(model, sentence, src_word2idx, tgt_idx2word, tokenizer, 
                               max_length, temperature=2.0, debug=False)
    print(f"Result: '{result3}'")
    
    return result1



# Test on some training examples
print("Testing on training examples:")
print("=" * 60)

# Select a few examples from the training set
test_indices = [0, 100, 200, 300, 400]

for idx in test_indices:
    eng_sentence, fra_reference = normalized_pairs[idx]
    
    # Translate
    translation = translate_sentence(model, eng_sentence, eng_word2idx, tgt_idx2word, tokenizer)
    
    print(f"English:    {eng_sentence}")
    print(f"Reference:  {fra_reference}")
    print(f"Translated: {translation}")
    print("-" * 60)



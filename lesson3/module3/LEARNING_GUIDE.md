# Learning Guide: Lesson 3, Module 3 -- Transformers: Self-Attention, Encoder, Decoder, Translation

## Module Overview

Transformers are the architecture behind GPT, BERT, and virtually all modern
NLP (and increasingly vision) models. This module teaches you the core mechanism
(self-attention) and how to build complete Transformer models from scratch.

## Recommended Reading Order

1. **self_attention/main.py** -- The core: Q, K, V and scaled dot-product attention
2. **transformer_encoder/main.py** -- Full encoder with positional encoding
3. **decoder_block/main.py** -- Causal masking and autoregressive generation
4. **translation/main.py** -- Complete encoder-decoder for machine translation

## Concept Map

```
Self-Attention (the core operation)
   |
   +--> Query (Q): "What am I looking for?"
   +--> Key (K):   "What do I contain?"
   +--> Value (V): "What info do I provide?"
   |
   +--> Attention = softmax(Q @ K^T / sqrt(d)) @ V
   |
   v
Transformer Encoder (for understanding)
   |
   +--> Tokenization + Embedding + Positional Encoding
   +--> N x EncoderBlock:
   |    +--> Multi-Head Self-Attention (bidirectional, sees all tokens)
   |    +--> Feed-Forward Network
   |    +--> Residual + LayerNorm
   |
   v
Transformer Decoder (for generation)
   |
   +--> Same as encoder BUT:
   |    +--> Masked Self-Attention (causal, sees only past tokens)
   |    +--> Cross-Attention (attends to encoder outputs)
   |
   v
Full Seq2Seq (translation)
   |
   +--> Encoder: process source sentence
   +--> Decoder: generate target sentence word by word
```

## File Summaries

### self_attention/main.py
Builds a self-attention model from scratch for next-word prediction.
Covers: tokenization, vocabulary building, sliding window datasets,
nn.MultiheadAttention, positional encoding, feed-forward layers.
Focus on: the math of attention (QKV) and why we divide by sqrt(d).

### transformer_encoder/main.py
Builds a complete Transformer encoder block with pre-LayerNorm architecture.
Trains on a text classification task using a tiny vocabulary.
Focus on: the EncoderBlock structure and how positional encoding works.

### decoder_block/main.py
Implements the decoder side: causal masking (preventing attention to future
tokens), autoregressive generation, and the key differences from encoders.
Focus on: the causal mask (upper triangular matrix with -inf) and how it
enforces "only look at past tokens."

### translation/main.py
Full encoder-decoder Transformer for machine translation (English to target
language). Includes: source/target vocabularies, cross-attention between
encoder and decoder, and beam-search-style decoding.
Focus on: how cross-attention connects the encoder's understanding to the
decoder's generation.

## Common Questions

**Q: What is the attention mechanism in plain English?**
A: For each word in a sentence, attention computes how much it should "pay
attention to" every other word. "The bank of the river" -- "bank" pays more
attention to "river" than "the." This creates context-aware representations.

**Q: Why divide by sqrt(d) in attention?**
A: The dot product QK^T grows with dimension d. Large values push softmax
into regions with tiny gradients (vanishing gradient problem). Dividing by
sqrt(d) keeps the values in a reasonable range. Think of it as normalization.

**Q: What is the difference between encoder and decoder?**
A: Encoder sees ALL tokens at once (bidirectional) -- good for understanding.
Decoder sees only PAST tokens (causal mask) -- good for generating one token
at a time. BERT is encoder-only. GPT is decoder-only. Translation uses both.

**Q: Why do we need positional encoding?**
A: Self-attention is permutation-invariant -- it treats input as a bag of words.
"Alice loves Bob" and "Bob loves Alice" get the same attention weights.
Positional encoding adds position information so the model knows word order.

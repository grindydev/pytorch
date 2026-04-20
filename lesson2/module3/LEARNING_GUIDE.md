# Learning Guide: Lesson 2, Module 3 -- NLP: Embeddings, Tokenizers, and Text Classification

## Module Overview

Images are easy for computers (just pixel arrays). Text is harder -- words are
discrete symbols with complex meaning. This module teaches you how to convert
text into numbers (embeddings and tokenization) and build models that understand
language enough to classify text.

## Recommended Reading Order

1. **embeddings/main.py** -- Word embeddings, GloVe, training custom embeddings
2. **tokenizer/main.py** -- Converting text to tokens (words/subwords)
3. **simple_text_classifier/main.py** -- Building a text classifier from scratch
4. **pretrained/main.py** -- Using pre-trained BERT for classification

## Concept Map

```
Raw Text: "I love machine learning"
   |
   v
Tokenization (split into pieces)
   |
   +--> Simple: split on whitespace
   +--> BERT WordPiece: "learning" -> "learn" + "##ing"
   |
   v
Word Embeddings (convert tokens to vectors)
   |
   +--> GloVe: fixed vectors, context-blind
   +--> BERT: contextual vectors, same word gets different vectors in context
   +--> Custom: learn embeddings during training
   |
   v
Text Classification Model
   |
   +--> Embedding layer: token index -> dense vector
   +--> Processing: LSTM, CNN, or Transformer
   +--> Output: class probabilities
```

## File Summaries

### embeddings/main.py
The longest and most important file. Covers:
- GloVe pre-trained embeddings and word analogies (king - man + woman = queen)
- Training custom embeddings from word co-occurrence pairs
- Cosine similarity for measuring word relatedness
- BERT contextual embeddings (same word, different meaning in different sentences)
Focus on: the difference between static (GloVe) and contextual (BERT) embeddings.

### tokenizer/main.py
Short file showing two tokenizers: a simple whitespace splitter and BERT's
WordPiece tokenizer. Shows how subword tokenization handles unknown words.
Focus on: why subword tokenization is better than word-level tokenization.

### simple_text_classifier/main.py
Builds a text classifier from scratch: custom Dataset, embedding layer,
processing layer, training loop. Classifies recipes as fruit or vegetable.
Focus on: the full pipeline from CSV to predictions.

### pretrained/main.py
Uses a pre-trained BERT model for text classification via HuggingFace Transformers.
Shows how to use transformers library for tokenization, model loading, and fine-tuning.
Focus on: how much easier it is to use pre-trained models vs building from scratch.

## Common Questions

**Q: What is a word embedding?**
A: A mapping from each word to a dense vector (list of numbers). Words with
similar meanings end up close together in vector space. "Cat" and "dog" have
similar vectors; "cat" and "car" have different vectors.

**Q: Why is GloVe not enough? Why do we need BERT?**
A: GloVe gives "bat" one fixed vector, whether you mean the animal or the
baseball equipment. BERT looks at the surrounding words and gives "bat"
different vectors depending on context. This is crucial for understanding meaning.

**Q: What is tokenization?**
A: Splitting text into smaller pieces (tokens). "I love ML" -> ["I", "love", "ML"].
BERT uses subword tokenization: "unhappiness" -> ["un", "##happiness"]. This
handles unknown words and is more flexible than splitting on spaces.

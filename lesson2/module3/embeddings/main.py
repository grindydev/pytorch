"""
Lesson 2 - Module 3: Word Embeddings -- GloVe and Contextual Vectors (embeddings/main.py)
==========================================================================================
WHAT YOU'LL LEARN:
  * What word embeddings are: dense vector representations of words
  * Pre-trained GloVe embeddings: static vectors that capture semantic meaning
  * Word analogies via vector arithmetic: king - man + woman = queen
  * Training your own simple embeddings from word co-occurrence pairs
  * Cosine similarity: measuring how similar two word vectors are
  * The limitation of static embeddings: "bat" (animal) = "bat" (baseball)
  * Contextual embeddings (BERT): same word, different vectors depending on context

KEY CONCEPT:
  A WORD EMBEDDING maps each word to a dense vector (e.g., 100 or 300 numbers).
  Words with similar meanings end up close together in vector space.
  This is how machines "understand" word meaning -- as geometric relationships.

  STATIC (GloVe): "bat" always gets the same vector, regardless of context
  CONTEXTUAL (BERT): "bat" gets different vectors depending on the sentence
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations
from pathlib import Path

import helper_utils

torch.manual_seed(123)
np.random.seed(123)


# ==================== PART 1: PRE-TRAINED GLOVE EMBEDDINGS ====================
# GloVe (Global Vectors for Word Representation) provides pre-trained word vectors.
# Each word is represented by a 100-dimensional vector (in this 100d version).

glove_path_file = Path.cwd() / 'data/glove_data'
helper_utils.download_glove6B(glove_path_file)

glove_file = glove_path_file / 'glove.6B.100d.txt'
glove_embeddings = helper_utils.load_glove_embeddings(glove_file)
# glove_embeddings is a dict: {"king": [0.50, 0.68, ...], "queen": [0.49, ...], ...}


# --- Word Analogies via Vector Arithmetic ---
# KEY CONCEPT: Embedding arithmetic reveals semantic relationships.
#   king - man + woman = queen
#   The vector (king - man) captures "royalty", and adding "woman" gives "queen".

def find_closest_words(embedding, embeddings_dict, exclude_words=[], top_n=5):
    """
    Finds the most similar words to a given embedding using cosine similarity.

    COSINE SIMILARITY measures the angle between two vectors:
      similarity = (A . B) / (|A| * |B|)
      Range: -1 to 1 (1 = same direction, 0 = orthogonal, -1 = opposite)
    """
    filtered_words = [word for word in embeddings_dict.keys() if word not in exclude_words]
    if not filtered_words:
        return None

    embedding_matrix = np.array([embeddings_dict[word] for word in filtered_words])
    target_embedding = embedding.reshape(1, -1)
    similarity_scores = cosine_similarity(target_embedding, embedding_matrix)
    closest_word_indices = np.argsort(similarity_scores[0])[::-1][:top_n]

    return [(filtered_words[i], similarity_scores[0][i]) for i in closest_word_indices]


# Classic analogy: king - man + woman = ?
king = glove_embeddings['king']
man = glove_embeddings['man']
woman = glove_embeddings['woman']
result_embedding = king - man + woman

closest_words = find_closest_words(
    result_embedding, glove_embeddings,
    exclude_words=['king', 'man', 'woman'], top_n=5
)

if closest_words:
    top_word, top_score = closest_words[0]
    print(f"king - man + woman = {top_word} (Score: {top_score:.4f})")

    print(f"\n--- Other Top Results ---")
    for word, score in closest_words[1:]:
        print(f"  {word} (Score: {score:.4f})")


# --- Try your own analogy ---
word1, word2, word3 = 'water', 'fire', 'tree'
analogy_words = [word1, word2, word3]

if all(word in glove_embeddings for word in analogy_words):
    result = glove_embeddings[word1] - glove_embeddings[word2] + glove_embeddings[word3]
    closest = find_closest_words(result, glove_embeddings, exclude_words=analogy_words, top_n=5)
    if closest:
        top_word, top_score = closest[0]
        print(f"\n'{word1}' - '{word2}' + '{word3}' = '{top_word}' (Score: {top_score:.4f})")


# --- Visualize embeddings in 2D using PCA ---
# KEY CONCEPT: Embeddings are high-dimensional (100d). PCA reduces them to 2D
# for visualization while preserving as much variance as possible.
# Similar words should cluster together in the plot.

words_to_visualize = ['car', 'bike', 'plane',      # Vehicles
                      'cat', 'dog', 'bird',         # Pets
                      'orange', 'apple', 'grape']   # Fruits

visualization_dict = {
    'Vehicle': ['car', 'bike', 'plane'],
    'Pet': ['cat', 'dog', 'bird'],
    'Fruit': ['orange', 'apple', 'grape']
}

embedding_vectors = np.array([glove_embeddings[word] for word in words_to_visualize])

# Reduce 100d -> 2d for plotting
reducer = PCA(n_components=2)
coords_2d = reducer.fit_transform(embedding_vectors)

helper_utils.plot_embeddings(coords=coords_2d, labels=words_to_visualize,
                             label_dict=visualization_dict,
                             title='GloVe Pre-Trained Embeddings')


# ==================== PART 2: TRAIN YOUR OWN EMBEDDINGS ====================
# KEY CONCEPT: We'll train a simple model that learns embeddings from scratch.
# The model takes a word index and predicts which other words appear nearby.
# Through this task, the model learns to place similar words close in vector space.

vocabulary = ['car', 'bike', 'plane', 'cat', 'dog', 'bird', 'orange', 'apple', 'grape']
vocab_categories = {
    'Vehicles': ['car', 'bike', 'plane'],
    'Pets': ['cat', 'dog', 'bird'],
    'Fruits': ['orange', 'apple', 'grape']
}

# Create word <-> index mappings
word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(vocabulary)
embedding_dim = 3  # Small for this toy example (real models use 100-300)

# Generate training pairs: all ordered pairs within each category
# KEY INSIGHT: Words in the same category co-occur, so the model learns
# that "car" is related to "bike" but not to "cat".
training_pairs = []
for category_list in vocab_categories.values():
    training_pairs.extend(list(permutations(category_list, 2)))

print(f"Generated {len(training_pairs)} training pairs.")


class SimpleEmbeddingModel(nn.Module):
    """
    A simple model for learning word embeddings.

    Architecture:
      Input (word index) -> Embedding layer -> Linear layer -> Vocabulary scores

    KEY CONCEPT: nn.Embedding is a lookup table that maps integer indices
    to dense vectors. During training, these vectors are adjusted so that
    words with similar roles get similar vectors.
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Embedding layer: maps word index -> dense vector
        # Shape: [vocab_size, embedding_dim]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Linear layer: projects embedding to vocabulary-size scores
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)    # [batch, embedding_dim]
        output = self.linear(embedded)  # [batch, vocab_size]
        return output, embedded


# --- Train the model ---
embedding_model = SimpleEmbeddingModel(vocab_size, embedding_dim)
optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()


def training_loop(model, training_pairs, epochs=2000):
    """Trains the embedding model on word co-occurrence pairs."""
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0

        for word1, word2 in training_pairs:
            word1_idx = torch.tensor([word_to_idx[word1]])
            word2_idx = torch.tensor([word_to_idx[word2]])

            output, _ = model(word1_idx)       # Predict context word
            loss = loss_function(output, word2_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(training_pairs))
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")

    print(f"Epoch {epochs}, Loss: {losses[-1]:.4f}")
    return model, losses


trained_model, losses = training_loop(embedding_model, training_pairs)
helper_utils.plot_loss(losses)


# --- Evaluate learned embeddings ---
# KEY CONCEPT: After training, words in the same category should have
# HIGH cosine similarity, and words in different categories should have LOW.

trained_model.eval()
all_embeddings = trained_model.embedding.weight.detach().numpy()


def cosine_similarity_words(word1, word2, word_to_idx, embeddings_matrix):
    idx1, idx2 = word_to_idx[word1], word_to_idx[word2]
    return cosine_similarity([embeddings_matrix[idx1]], [embeddings_matrix[idx2]])[0][0]


# Test: "car" should be more similar to "bike" than to "cat"
similarity_tests = [
    ("car", "car"), ("car", "bike"), ("car", "plane"),  # Same category
    ("car", "cat"), ("car", "dog"), ("car", "bird"),    # Different category
    ("car", "orange"), ("car", "apple"), ("car", "grape"),  # Different category
]

print("\nSemantic Similarity (Cosine Similarity):")
print("=" * 40)
for word1, word2 in similarity_tests:
    similarity = cosine_similarity_words(word1, word2, word_to_idx, all_embeddings)
    print(f"{word1} <-> {word2}:\t {similarity:.4f}")

# Visualize the similarity matrix
similarity_matrix = np.zeros((vocab_size, vocab_size))
for i in range(vocab_size):
    for j in range(vocab_size):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            similarity_matrix[i, j] = cosine_similarity_words(
                idx_to_word[i], idx_to_word[j], word_to_idx, all_embeddings
            )

helper_utils.plot_similarity_matrix(similarity_matrix, vocabulary)

# Visualize learned embeddings in 2D
coords = PCA(n_components=2).fit_transform(all_embeddings)
helper_utils.plot_embeddings(coords=coords, labels=vocabulary,
                             label_dict=vocab_categories,
                             title='Your Trained Word Embeddings')


# ==================== PART 3: STATIC vs. CONTEXTUAL EMBEDDINGS ====================
# KEY CONCEPT: GloVe gives "bat" ONE fixed vector. But "bat" means different
# things in different sentences:
#   "A bat flew out of the cave"  -> bat = animal
#   "He swung the baseball bat"   -> bat = sports equipment
#
# With GloVe, both get the SAME vector. This is a limitation.
# BERT (Bidirectional Encoder Representations from Transformers) solves this:
# the same word gets DIFFERENT vectors depending on its surrounding context.

sentence1 = "A bat flew out of the cave."
sentence2 = "He swung the baseball bat."

# --- GloVe: same vector for "bat" in both sentences ---
bat_from_sentence1 = glove_embeddings["bat"]
bat_from_sentence2 = glove_embeddings["bat"]
are_identical = np.array_equal(bat_from_sentence1, bat_from_sentence2)
print(f"\nGloVe: Are vectors for 'bat' identical across sentences? {are_identical}")
# Result: True -- GloVe is context-blind!

# --- BERT: different vectors for "bat" in different contexts ---
bert_path_file = Path.cwd() / 'data/bert_model'
helper_utils.download_bert(save_directory=bert_path_file)
tokenizer, model_bert = helper_utils.load_bert(save_directory=bert_path_file)

# Process sentence 1
inputs1 = tokenizer(sentence1, return_tensors='pt')
with torch.no_grad():
    outputs1 = model_bert(**inputs1)
last_hidden_state1 = outputs1.last_hidden_state[0]
tokens1 = tokenizer.convert_ids_to_tokens(inputs1['input_ids'][0])

print("\n--- BERT: Sentence 1 tokens ---")
for token, vector in zip(tokens1, last_hidden_state1):
    print(f"  {token:<12} {vector.numpy()[:5]}")

# Process sentence 2
inputs2 = tokenizer(sentence2, return_tensors='pt')
with torch.no_grad():
    outputs2 = model_bert(**inputs2)
last_hidden_state2 = outputs2.last_hidden_state[0]
tokens2 = tokenizer.convert_ids_to_tokens(inputs2['input_ids'][0])

print("\n--- BERT: Sentence 2 tokens ---")
for token, vector in zip(tokens2, last_hidden_state2):
    print(f"  {token:<12} {vector.numpy()[:5]}")

# Compare the two "bat" vectors
bat_animal_vector = last_hidden_state1[2].numpy()   # "bat" at position 2 in sentence 1
bat_sport_vector = last_hidden_state2[5].numpy()     # "bat" at position 5 in sentence 2
are_identical = np.array_equal(bat_animal_vector, bat_sport_vector)
print(f"\nBERT: Are contextual vectors for 'bat' identical? {are_identical}")
# Result: False -- BERT captures context!

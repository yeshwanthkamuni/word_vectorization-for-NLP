# ============================================================
#  Word Vectorizer Experiment - 24CYS214 ML Assignment
#  Dataset  : Twitter_Data.csv (162,980 tweets, 3-class sentiment)
#  Methods  : Bag of Words | TF-IDF | FastText-style | GloVe-style
#  Classifier: Logistic Regression
# ============================================================

import pandas as pd
import numpy as np
import re
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report)

# ─────────────────────────────────────────────
# STEP 1: LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv('Twitter_Data.csv')
print(f"Total records loaded : {len(df)}")
print(f"Columns              : {df.columns.tolist()}")
print(f"Missing values       : {df.isnull().sum().to_dict()}")

# Drop nulls and fix label type
df = df.dropna(subset=['clean_text', 'category'])
df['category'] = df['category'].astype(int)

print(f"Records after cleaning: {len(df)}")
print(f"Label distribution:\n{df['category'].value_counts().sort_index()}")
# Labels: -1 = Negative, 0 = Neutral, 1 = Positive


# ─────────────────────────────────────────────
# STEP 2: TEXT PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Text Preprocessing")
print("=" * 60)

STOPWORDS = set([
    'the','a','an','is','it','in','on','at','to','for','of',
    'and','or','but','are','was','were','be','been','has','have',
    'had','do','does','did','will','would','could','should','may',
    'might','i','you','he','she','we','they','me','him','her','us',
    'them','my','your','his','our','their','this','that','these',
    'those','with','from','by','as','so','if','not','no','up','out',
    'about','just','can'
])

def preprocess(text):
    """
    Pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove @mentions and #hashtags
    4. Remove non-alphabetic characters
    5. Remove extra whitespace
    6. Remove stopwords and single-char tokens
    """
    text = str(text).lower()                            # Lowercase
    text = re.sub(r'http\S+|www\S+', '', text)          # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)               # Remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', '', text)                # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()            # Normalize spaces
    tokens = [w for w in text.split()
              if w not in STOPWORDS and len(w) > 1]     # Remove stopwords
    return ' '.join(tokens)

t0 = time.time()
df['processed'] = df['clean_text'].apply(preprocess)
print(f"Preprocessing done in {time.time()-t0:.2f}s")
print(f"\nOriginal sample : {df['clean_text'].iloc[0]}")
print(f"Processed sample: {df['processed'].iloc[0]}")


# ─────────────────────────────────────────────
# STEP 3: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Train/Test Split (80% / 20%)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    df['processed'], df['category'],
    test_size=0.2, random_state=42, stratify=df['category']
)
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")


# ─────────────────────────────────────────────
# HELPER: Evaluate and print results
# ─────────────────────────────────────────────
results = {}

def evaluate(name, y_true, y_pred, vec_time, train_time):
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average='weighted')
    prec  = precision_score(y_true, y_pred, average='weighted')
    rec   = recall_score(y_true, y_pred, average='weighted')
    report = classification_report(
        y_true, y_pred,
        target_names=['Negative', 'Neutral', 'Positive']
    )
    results[name] = {
        'accuracy': acc, 'f1': f1,
        'precision': prec, 'recall': rec,
        'vec_time': vec_time, 'train_time': train_time
    }
    print(f"\n{'─'*50}")
    print(f"  Results for: {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Vec Time  : {vec_time:.2f}s")
    print(f"  Train Time: {train_time:.2f}s")
    print(f"\nClassification Report:\n{report}")


# ═══════════════════════════════════════════════════════════
#  METHOD 1: BAG OF WORDS (BoW)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("METHOD 1: Bag of Words (BoW)")
print("=" * 60)

# Vectorize
t0 = time.time()
bow_vectorizer = CountVectorizer(max_features=10000)
X_bow_train = bow_vectorizer.fit_transform(X_train)
X_bow_test  = bow_vectorizer.transform(X_test)
bow_vec_time = time.time() - t0

print(f"Vocabulary size : {len(bow_vectorizer.vocabulary_)}")
print(f"Feature matrix  : {X_bow_train.shape} (sparse)")

# Train classifier
t0 = time.time()
clf_bow = LogisticRegression(max_iter=1000, random_state=42)
clf_bow.fit(X_bow_train, y_train)
bow_train_time = time.time() - t0

# Predict and evaluate
y_pred_bow = clf_bow.predict(X_bow_test)
evaluate('BoW', y_test, y_pred_bow, bow_vec_time, bow_train_time)


# ═══════════════════════════════════════════════════════════
#  METHOD 2: TF-IDF (Term Frequency - Inverse Document Frequency)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("METHOD 2: TF-IDF (Unigrams + Bigrams)")
print("=" * 60)

# Vectorize
t0 = time.time()
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),   # unigrams and bigrams
    sublinear_tf=True     # apply log normalization to TF
)
X_tfidf_train = tfidf_vectorizer.fit_transform(X_train)
X_tfidf_test  = tfidf_vectorizer.transform(X_test)
tfidf_vec_time = time.time() - t0

print(f"Vocabulary size : {len(tfidf_vectorizer.vocabulary_)}")
print(f"Feature matrix  : {X_tfidf_train.shape} (sparse)")

# Train classifier
t0 = time.time()
clf_tfidf = LogisticRegression(max_iter=1000, random_state=42)
clf_tfidf.fit(X_tfidf_train, y_train)
tfidf_train_time = time.time() - t0

# Predict and evaluate
y_pred_tfidf = clf_tfidf.predict(X_tfidf_test)
evaluate('TF-IDF', y_test, y_pred_tfidf, tfidf_vec_time, tfidf_train_time)


# ═══════════════════════════════════════════════════════════
#  METHOD 3: FastText-style (Character N-gram Subword TF-IDF)
#  FastText's core innovation: subword (char n-gram) representation
#  This replicates the vectorization strategy without external library
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("METHOD 3: FastText-style (Character N-gram Subword Vectorizer)")
print("=" * 60)

# Vectorize using character-level n-grams (3 to 4 chars) — FastText's core idea
t0 = time.time()
fasttext_vectorizer = TfidfVectorizer(
    analyzer='char_wb',     # character n-grams within word boundaries
    ngram_range=(3, 4),     # trigrams and 4-grams (subword units)
    max_features=50000,
    sublinear_tf=True
)
X_ft_train = fasttext_vectorizer.fit_transform(X_train)
X_ft_test  = fasttext_vectorizer.transform(X_test)
ft_vec_time = time.time() - t0

print(f"Subword vocab size : {len(fasttext_vectorizer.vocabulary_)}")
print(f"Feature matrix     : {X_ft_train.shape} (sparse)")
print("Note: Uses character n-grams (3-4) to simulate FastText's")
print("      subword approach — handles OOV and morphological variants.")

# Train classifier
t0 = time.time()
clf_ft = LogisticRegression(max_iter=1000, C=5, random_state=42)
clf_ft.fit(X_ft_train, y_train)
ft_train_time = time.time() - t0

# Predict and evaluate
y_pred_ft = clf_ft.predict(X_ft_test)
evaluate('FastText', y_test, y_pred_ft, ft_vec_time, ft_train_time)


# ═══════════════════════════════════════════════════════════
#  METHOD 4: GloVe-style (Averaged Word Embeddings)
#  Simulates GloVe's dense embedding approach:
#  Each word → dense vector → document = mean of word vectors
#  Note: Uses randomly initialized vectors (no internet for pretrained).
#        Real GloVe would use vectors pretrained on Wikipedia/Common Crawl.
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("METHOD 4: GloVe-style (Averaged Dense Word Embeddings, 100-dim)")
print("=" * 60)

VOCAB_SIZE = 15000
EMBED_DIM  = 100

# Build vocabulary from training data
vocab_builder = TfidfVectorizer(max_features=VOCAB_SIZE)
vocab_builder.fit(X_train)
vocab = vocab_builder.vocabulary_   # word -> index mapping

# Initialize embedding matrix
# In real GloVe: load pretrained glove.6B.100d.txt
# Here: reproducible random init for demonstration
np.random.seed(42)
glove_matrix = np.random.randn(VOCAB_SIZE, EMBED_DIM) * 0.1

def document_to_glove_vector(text, vocab, embedding_matrix, embed_dim):
    """Average the GloVe vectors of all recognized words in the document."""
    tokens = text.split()
    word_vectors = [embedding_matrix[vocab[w]]
                    for w in tokens if w in vocab]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(embed_dim)  # fallback for empty/OOV docs

t0 = time.time()
X_glove_train = np.array([
    document_to_glove_vector(t, vocab, glove_matrix, EMBED_DIM)
    for t in X_train
])
X_glove_test = np.array([
    document_to_glove_vector(t, vocab, glove_matrix, EMBED_DIM)
    for t in X_test
])
glove_vec_time = time.time() - t0

print(f"Embedding dimension : {EMBED_DIM}")
print(f"Train matrix shape  : {X_glove_train.shape} (dense)")
print("Note: Real GloVe uses pretrained vectors from large corpora.")
print("      Random init here demonstrates the averaging pipeline.")

# Train classifier
t0 = time.time()
clf_glove = LogisticRegression(max_iter=1000, random_state=42)
clf_glove.fit(X_glove_train, y_train)
glove_train_time = time.time() - t0

# Predict and evaluate
y_pred_glove = clf_glove.predict(X_glove_test)
evaluate('GloVe', y_test, y_pred_glove, glove_vec_time, glove_train_time)


# ─────────────────────────────────────────────
# STEP 4: SUMMARY COMPARISON TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: All Methods Compared")
print("=" * 60)

print(f"\n{'Method':<12} {'Accuracy':>10} {'Precision':>10} "
      f"{'Recall':>10} {'F1-Score':>10} {'Vec Time':>10} {'Train Time':>12}")
print("-" * 76)

for method, r in results.items():
    print(f"{method:<12} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
          f"{r['recall']:>10.4f} {r['f1']:>10.4f} "
          f"{r['vec_time']:>9.2f}s {r['train_time']:>11.2f}s")

print("\nBest Accuracy :", max(results, key=lambda x: results[x]['accuracy']))
print("Best F1-Score :", max(results, key=lambda x: results[x]['f1']))
print("Fastest Vec   :", min(results, key=lambda x: results[x]['vec_time']))
print("\n[Done] Experiment complete.")

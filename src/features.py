# ─────────────────────────────────────────────
# Sentiment Analysis — Feature Engineering (Ngoc)
# ─────────────────────────────────────────────

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_features(train_series, val_series, max_features=10000):
    """
    Converts cleaned tweet text into TF-IDF numerical vectors.
    - Fits ONLY on training data to avoid data leakage
    - Transforms both train and validation sets
    - Uses unigrams and bigrams (ngram_range=(1,2))
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),      # uses single words AND pairs of words
        stop_words='english',    # removes common English words
        sublinear_tf=True        # applies log normalization to term frequencies
    )

    # IMPORTANT: fit_transform on train, only transform on val
    # Never fit on val — that would be data leakage
    X_train = vectorizer.fit_transform(train_series)
    X_val   = vectorizer.transform(val_series)

    print(f"Vocabulary size:       {len(vectorizer.vocabulary_)}")
    print(f"Training matrix shape: {X_train.shape}")
    print(f"Validation matrix shape: {X_val.shape}")

    return X_train, X_val, vectorizer

def save_vectorizer(vectorizer, path="data/vectorizer.pkl"):
    """Saves the fitted vectorizer so teammates can reuse it."""
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved to {path}")
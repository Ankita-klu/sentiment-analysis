from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_features(train_series, val_series, max_features=5000):
    """Converts text to TF-IDF vectors."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train = vectorizer.fit_transform(train_series)
    X_val = vectorizer.transform(val_series)
    return X_train, X_val, vectorizer

def save_vectorizer(vectorizer, path):
    joblib.dump(vectorizer, path)

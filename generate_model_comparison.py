import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from src.mlp import MLPClassifier
from src.utils import one_hot_encode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def main():
    # Validate required files exist
    train_file = os.path.join(DATA_DIR, 'processed', 'train_clean.csv')
    val_file = os.path.join(DATA_DIR, 'processed', 'val_clean.csv')
    model_file = os.path.join(DATA_DIR, 'best_model.pkl')
    vectorizer_file = os.path.join(DATA_DIR, 'vectorizer.pkl')

    for f in [train_file, val_file, model_file, vectorizer_file]:
        if not os.path.exists(f):
            print(f"Error: {f} not found.")
            sys.exit(1)

    # Load data
    train_df = pd.read_csv(train_file).fillna('')
    val_df = pd.read_csv(val_file).fillna('')

    X_train_raw = train_df['clean_tweet']
    y_train = train_df['sentiment']
    X_val_raw = val_df['clean_tweet']
    y_val = val_df['sentiment']

    # TF-IDF for sklearn models
    vec = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vec.fit_transform(X_train_raw)
    X_val_vec = vec.transform(X_val_raw)

    # sklearn models
    sklearn_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'SVM': LinearSVC(max_iter=2000, class_weight='balanced'),
    }

    f1_scores = {}
    for name, model in sklearn_models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_val_vec)
        f1_scores[name] = f1_score(y_val, y_pred, average='weighted')
        print(f"{name}: {f1_scores[name]:.4f}")

    # Load your trained MLP
    with open(model_file, 'rb') as f:
        mlp = pickle.load(f)
    with open(vectorizer_file, 'rb') as f:
        mlp_vec = pickle.load(f)

    X_val_mlp = mlp_vec.transform(X_val_raw)
    label_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral', 3: 'Irrelevant'}
    y_pred_mlp = np.array([label_map[i] for i in mlp.predict(X_val_mlp)])
    f1_scores['MLP (Custom)'] = f1_score(y_val, y_pred_mlp, average='weighted')
    print(f"MLP (Custom): {f1_scores['MLP (Custom)']:.4f}")

    # Plot
    colors = ['#4361ee', '#7209b7', '#f72585', '#4cc9f0']
    plt.figure(figsize=(9, 5))
    bars = plt.bar(f1_scores.keys(), f1_scores.values(), color=colors)
    plt.title('Model Comparison — Weighted F1 Score')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    for bar, score in zip(bars, f1_scores.values()):
        plt.text(bar.get_x() + bar.get_width()/2, score + 0.01,
                 f"{score:.4f}", ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'model_comparison.png'))
    print("Saved: model_comparison.png")

if __name__ == "__main__":
    main()
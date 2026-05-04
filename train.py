import os
import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.mlp import cross_entropy_loss, MLPClassifier
from src.vectorizer import TFIDFVectorizer
from src.utils import one_hot_encode, accuracy

# Configuration
CONFIG = {
    'max_features': 1000,
    'min_df': 1,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'lambda_reg': 0.0001,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_data_path(filename):
    return os.path.join(BASE_DIR, 'data', filename)

def main():
    # Validate required files exist
    train_file = get_data_path('processed/train_clean.csv')
    val_file = get_data_path('processed/val_clean.csv')

    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found. Run preprocess.py first.")
        sys.exit(1)
    if not os.path.exists(val_file):
        print(f"Error: {val_file} not found. Run preprocess.py first.")
        sys.exit(1)

    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)

    df_train = df_train.dropna(subset=['clean_tweet'])
    df_val = df_val.dropna(subset=['clean_tweet'])

    print(f"Train: {len(df_train)}, Val: {len(df_val)}")

    vec = TFIDFVectorizer(max_features=CONFIG['max_features'], min_df=CONFIG['min_df'])
    X_train = vec.fit_transform(df_train['clean_tweet'].values)
    X_val = vec.transform(df_val['clean_tweet'].values)

    print(f"Features: {X_train.shape[1]}")

    sentiment_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}
    y_train_labels = df_train['sentiment'].map(sentiment_map).values
    y_train = one_hot_encode(y_train_labels, 4)
    y_val_labels = df_val['sentiment'].map(sentiment_map).values
    y_val = one_hot_encode(y_val_labels, 4)

    layer_sizes = [X_train.shape[1], 128, 64, 4]
    model = MLPClassifier(layer_sizes)

    print("Training on full dataset...\n")

    best_acc = 0
    best_model_weights = None

    train_losses = []
    val_losses = []

    for epoch in range(CONFIG['epochs']):
        m = X_train.shape[0]
        indices = np.random.permutation(m)

        for i in range(0, m, CONFIG['batch_size']):
            idx = indices[i:i + CONFIG['batch_size']]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            model.forward(X_batch)
            model.backward(y_batch, lr=CONFIG['learning_rate'], lambda_reg=CONFIG['lambda_reg'])

        model.forward(X_train)
        train_loss = cross_entropy_loss(model.predictions, y_train)
        train_losses.append(train_loss)

        model.forward(X_val)
        val_loss = cross_entropy_loss(model.predictions, y_val)
        val_losses.append(val_loss)

        pred_val = model.predict(X_val)
        val_acc = np.mean(pred_val == y_val_labels)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = model.get_weights()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Val Acc = {val_acc:.4f}")

    print(f"\nFinal Accuracy: {best_acc:.4f}")

    model.set_weights(best_model_weights)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training Progress — Early Stopping Point Visible')
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_data_path('training_progress.png'))
    print(f"Saved: {get_data_path('training_progress.png')}")

    # Save model and vectorizer
    os.makedirs(get_data_path(''), exist_ok=True)

    with open(get_data_path('best_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: {get_data_path('best_model.pkl')}")

    with open(get_data_path('vectorizer.pkl'), 'wb') as f:
        pickle.dump(vec, f)
    print(f"Saved: {get_data_path('vectorizer.pkl')}")

if __name__ == "__main__":
    main()

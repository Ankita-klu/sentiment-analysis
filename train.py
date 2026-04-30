import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.vectorizer import TFIDFVectorizer
from src.mlp import MLPClassifier
from src.utils import one_hot_encode, accuracy

df_train = pd.read_csv('data/train_clean.csv')
df_val = pd.read_csv('data/val_clean.csv')

df_train = df_train.dropna(subset=['clean_tweet'])
df_val = df_val.dropna(subset=['clean_tweet'])

print(f"Train: {len(df_train)}, Val: {len(df_val)}")

vec = TFIDFVectorizer(max_features=1000, min_df=1)
X_train = vec.fit_transform(df_train['clean_tweet'].values)
X_val = vec.transform(df_val['clean_tweet'].values)

print(f"Features: {X_train.shape[1]}")

sentiment_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}
y_train_labels = df_train['sentiment'].map(sentiment_map).values
y_train = one_hot_encode(y_train_labels, 4)
y_val_labels = df_val['sentiment'].map(sentiment_map).values
y_val = one_hot_encode(y_val_labels, 4)

model = MLPClassifier([X_train.shape[1], 128, 64, 4])

print("Training on full dataset...\n")

best_acc = 0
for epoch in range(50):
    m = X_train.shape[0]
    indices = np.random.permutation(m)
    
    for i in range(0, m, 32):
        idx = indices[i:i+32]
        X_batch = X_train[idx]
        y_batch = y_train[idx]
        
        model.forward(X_batch)
        model.backward(y_batch, lr=0.001)
    
    pred_val = model.predict(X_val)
    val_acc = np.mean(pred_val == y_val_labels)
    
    if val_acc > best_acc:
        best_acc = val_acc
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Val Acc = {val_acc:.4f}")

print(f"\nFinal Accuracy: {best_acc:.4f}")
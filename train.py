import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.vectorizer import TFIDFVectorizer
from src.mlp import MLPClassifier
from src.utils import one_hot_encode, accuracy

df_val = pd.read_csv('data/val_clean.csv')
df_val = df_val.dropna(subset=['clean_tweet'])

X_val = df_val['clean_tweet'].values
y_val_labels = df_val['sentiment'].map({'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}).values

vec = TFIDFVectorizer(max_features=500, min_df=1)
X_val_vec = vec.fit_transform(X_val)

y_val_onehot = one_hot_encode(y_val_labels, 4)
model = MLPClassifier([X_val_vec.shape[1], 64, 32, 4])

print(f"Training on {len(df_val)} samples, {X_val_vec.shape[1]} features\n")

for epoch in range(100):
    m = X_val_vec.shape[0]
    indices = np.random.permutation(m)
    
    for i in range(0, m, 16):
        idx = indices[i:i+16]
        model.forward(X_val_vec[idx])
        model.backward(y_val_onehot[idx], lr=0.001)
    
    if epoch % 10 == 0:
        pred = model.predict(X_val_vec)
        acc = np.mean(pred == y_val_labels)
        print(f"Epoch {epoch}: Accuracy = {acc:.4f}")

print(f"\n✅ Final Accuracy: {np.mean(model.predict(X_val_vec) == y_val_labels):.4f}")

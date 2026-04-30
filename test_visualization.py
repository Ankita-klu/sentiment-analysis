import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.vectorizer import TFIDFVectorizer
from src.mlp import MLPClassifier
from src.utils import one_hot_encode, accuracy
from src.visualization import plot_confusion_matrix, print_per_class_metrics, print_overall_metrics

print("Phase D: Testing Visualization Functions")
print("=" * 70)

df_val = pd.read_csv('data/processed/val_clean.csv')
df_val = df_val.dropna(subset=['clean_tweet'])

print(f"Loading {len(df_val)} validation samples...")

X_val = df_val['clean_tweet'].values
y_val_labels = df_val['sentiment'].map({'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}).values

vec = TFIDFVectorizer(max_features=500, min_df=1)
X_val_vec = vec.fit_transform(X_val)

y_val_onehot = one_hot_encode(y_val_labels, 4)
model = MLPClassifier([X_val_vec.shape[1], 64, 32, 4])

print("Training model...")
for epoch in range(50):
    m = X_val_vec.shape[0]
    indices = np.random.permutation(m)
    
    for i in range(0, m, 16):
        idx = indices[i:i+16]
        model.forward(X_val_vec[idx])
        model.backward(y_val_onehot[idx], lr=0.001, lambda_reg=0.0001)

print("Generating predictions...")
pred = model.predict(X_val_vec)
val_acc = accuracy(pred, y_val_labels)

print(f"\nValidation Accuracy: {val_acc:.4f}")

class_names = ['Positive', 'Negative', 'Neutral', 'Irrelevant']

print("\nGenerating visualizations...")
cm = plot_confusion_matrix(pred, y_val_labels, class_names)
print_per_class_metrics(pred, y_val_labels, class_names)
print_overall_metrics(pred, y_val_labels)

print("\nPhase D Complete! Visualizations saved to data/artifacts/")
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.vectorizer import TFIDFVectorizer
from src.mlp import MLPClassifier
from src.utils import one_hot_encode
from src.gradient_check import gradient_check, print_gradient_check_result

print("Loading data for gradient check...")

df_val = pd.read_csv('data/processed/val_clean.csv')
df_val = df_val.dropna(subset=['clean_tweet'])

X_val = df_val['clean_tweet'].values[:100]  # Use only 100 samples
y_val_labels = df_val['sentiment'].map({'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}).values[:100]

vec = TFIDFVectorizer(max_features=500, min_df=1)
X_val_vec = vec.fit_transform(X_val)

y_val_onehot = one_hot_encode(y_val_labels, 4)

print(f"Data shape: {X_val_vec.shape}")

model = MLPClassifier([X_val_vec.shape[1], 64, 32, 4])

print("\nRunning gradient check...")
result = gradient_check(model, X_val_vec, y_val_onehot, epsilon=1e-7)

print_gradient_check_result(result)

if result['passed']:
    print("\nBackpropagation is mathematically correct!")
else:
    print("\nWarning: Check backpropagation implementation")
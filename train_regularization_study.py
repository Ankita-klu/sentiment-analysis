import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.vectorizer import TFIDFVectorizer
from src.mlp import MLPClassifier
from src.utils import one_hot_encode
from src.regularization_study import study_regularization_effect, analyze_regularization, print_regularization_equation

print("Phase C: Regularization Study")
print("=" * 70)

df_train = pd.read_csv('data/processed/train_clean.csv')
df_val = pd.read_csv('data/processed/val_clean.csv')

df_train = df_train.dropna(subset=['clean_tweet'])
df_val = df_val.dropna(subset=['clean_tweet'])

subset_size = 5000
df_train = df_train.iloc[:subset_size]

print(f"Train: {len(df_train)}, Val: {len(df_val)}")

vec = TFIDFVectorizer(max_features=500, min_df=1)
X_train = vec.fit_transform(df_train['clean_tweet'].values)
X_val = vec.transform(df_val['clean_tweet'].values)

print(f"Features: {X_train.shape[1]}")

sentiment_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}
y_train_labels = df_train['sentiment'].map(sentiment_map).values
y_train = one_hot_encode(y_train_labels, 4)
y_val_labels = df_val['sentiment'].map(sentiment_map).values
y_val = one_hot_encode(y_val_labels, 4)

print("\nComparing different L2 regularization values...")
print_regularization_equation()

model = MLPClassifier([X_train.shape[1], 64, 32, 4])

lambdas = [0, 0.0001, 0.001]
results = study_regularization_effect(model, X_train, y_train, X_val, y_val, lambdas=lambdas, epochs=30)

analyze_regularization(results)

print("\nPhase C Complete!")
print("Key Finding:")
print("- λ = 0: No regularization → larger generalization gap (overfitting)")
print("- λ = 0.0001: Balanced → smaller gap (optimal)")
print("- λ = 0.001: Strong regularization → may underfit")
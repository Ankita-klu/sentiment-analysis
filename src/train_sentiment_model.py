import pandas as pd
import numpy as np
from src.vectorizer import TFIDFVectorizer
from src.mlp import MLPClassifier
from src.utils import one_hot_encode, accuracy
from src.train import train_epoch

# Load cleaned data
df_train = pd.read_csv('data/train_clean.csv')
df_val = pd.read_csv('data/val_clean.csv')

print(f"Training samples: {len(df_train)}")
print(f"Validation samples: {len(df_val)}")

# Vectorize
vec = TFIDFVectorizer(max_features=5000, min_df=2)
X_train = vec.fit_transform(df_train['clean_tweet'].values)
X_val = vec.transform(df_val['clean_tweet'].values)

print(f"Features: {X_train.shape[1]}")

# Convert labels
sentiment_to_num = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}
y_train_labels = df_train['sentiment'].map(sentiment_to_num).values
y_train = one_hot_encode(y_train_labels, 4)

y_val_labels = df_val['sentiment'].map(sentiment_to_num).values

# Create model
model = MLPClassifier([X_train.shape[1], 256, 128, 4])

# Train
print("\nTraining...")
for epoch in range(100):  # Train for 100 epochs
    loss = train_epoch(model, X_train, y_train, lr=0.001, batch_size=32)
    if epoch % 10 == 0:
        pred = model.predict(X_train)
        acc = accuracy(pred, y_train_labels)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

# Validate
pred_val = model.predict(X_val)
val_acc = accuracy(pred_val, y_val_labels)
print(f"\n✅ Final Validation Accuracy: {val_acc:.4f}")
from src.mlp import cross_entropy_loss
import matplotlib.pyplot as plt  
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pickle


import pandas as pd
import numpy as np
from src.vectorizer import TFIDFVectorizer
from src.mlp import MLPClassifier
from src.utils import one_hot_encode, accuracy

df_train = pd.read_csv('data/processed/train_clean.csv')
df_val = pd.read_csv('data/processed/val_clean.csv')

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
best_model_weights = None

train_losses = []
val_losses = []


for epoch in range(50):
    m = X_train.shape[0]
    indices = np.random.permutation(m)
    
    for i in range(0, m, 32):
        idx = indices[i:i+32]
        X_batch = X_train[idx]
        y_batch = y_train[idx]
        
        model.forward(X_batch)
        model.backward(y_batch, lr=0.001)

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

                # Snapshot the best weights so we save the peak model, not the last epoch
        # after
        best_model_weights = model.get_weights()
        # best_model_biases  = [b.copy() for b in model.biases]

    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Val Acc = {val_acc:.4f}")

print(f"\nFinal Accuracy: {best_acc:.4f}")


# Restore best weights before saving
# after
model.set_weights(best_model_weights)
# model.biases  = best_model_biases

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Progress — Early Stopping Point Visible')
plt.legend()
plt.tight_layout()
plt.savefig('data/training_progress.png')
print("Saved: data/training_progress.png")

# Save to data/ so the Streamlit app can find them
os.makedirs('data', exist_ok=True)

with open('data/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved: data/best_model.pkl")

with open('data/vectorizer.pkl', 'wb') as f:
    pickle.dump(vec, f)
print("Saved: data/vectorizer.pkl")

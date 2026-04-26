# ─────────────────────────────────────────────
# Sentiment Analysis — Modeling (Ngoc)
# ─────────────────────────────────────────────
# Trains and compares 3 models:
#   1. Logistic Regression
#   2. Naive Bayes (MultinomialNB)
#   3. Support Vector Machine (LinearSVC)
# Saves the best model + vectorizer for evaluation & demo
# ─────────────────────────────────────────────

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path so we can import features.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, save_vectorizer

# ── 1. LOAD ANKITA'S CLEANED DATA ─────────────
print("Loading cleaned data...")
train_df = pd.read_csv("data/train_clean.csv")
val_df   = pd.read_csv("data/val_clean.csv")

# Fill any remaining NaN values just in case
train_df["clean_tweet"] = train_df["clean_tweet"].fillna("")
val_df["clean_tweet"]   = val_df["clean_tweet"].fillna("")

X_train = train_df["clean_tweet"]
y_train = train_df["sentiment"]
X_val   = val_df["clean_tweet"]
y_val   = val_df["sentiment"]

print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Classes: {y_train.unique().tolist()}")

# ── 2. FEATURE ENGINEERING (TF-IDF) ───────────
print("\nExtracting TF-IDF features...")
X_train_vec, X_val_vec, vectorizer = extract_features(X_train, X_val)

# ── 3. DEFINE MODELS ──────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced'   # handles class imbalance
    ),
    "Naive Bayes": MultinomialNB(
        alpha=0.1                 # smoothing parameter
    ),
    "SVM": LinearSVC(
        max_iter=2000,
        class_weight='balanced'   # handles class imbalance
    )
}

# ── 4. TRAIN AND EVALUATE ALL MODELS ──────────
results   = {}
trained   = {}
CLASS_NAMES = ["Positive", "Negative", "Neutral", "Irrelevant"]

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_val_vec)
    report = classification_report(
        y_val, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    results[name] = report
    trained[name] = model

    print(f"\n{name} Results:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))

# ── 5. COMPARE MODELS ─────────────────────────
print("\n" + "="*60)
print("SUMMARY — Weighted F1 Scores")
print("="*60)

f1_scores = {}
for name, report in results.items():
    f1 = report["weighted avg"]["f1-score"]
    acc = report["accuracy"]
    f1_scores[name] = f1
    print(f"{name:<25} Accuracy: {acc:.4f}   F1: {f1:.4f}")

# ── 6. PICK BEST MODEL ────────────────────────
best_name  = max(f1_scores, key=f1_scores.get)
best_model = trained[best_name]
print(f"\nBest model: {best_name} (F1: {f1_scores[best_name]:.4f})")

# ── 7. CONFUSION MATRIX FOR BEST MODEL ────────
y_pred_best = best_model.predict(X_val_vec)
cm = confusion_matrix(y_val, y_pred_best, labels=CLASS_NAMES)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("data/confusion_matrix.png")
plt.show()
print("Saved: data/confusion_matrix.png")

# ── 8. F1 SCORE COMPARISON CHART ──────────────
plt.figure(figsize=(7, 4))
plt.bar(f1_scores.keys(), f1_scores.values(), color=['#4361ee', '#7209b7', '#f72585'])
plt.title('Model Comparison — Weighted F1 Score')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
for i, (name, score) in enumerate(f1_scores.items()):
    plt.text(i, score + 0.01, f"{score:.4f}", ha='center', fontsize=11)
plt.tight_layout()
plt.savefig("data/model_comparison.png")
plt.show()
print("Saved: data/model_comparison.png")

# ── 9. SAVE BEST MODEL + VECTORIZER ───────────
os.makedirs("data", exist_ok=True)
joblib.dump(best_model, "data/best_model.pkl")
joblib.dump(vectorizer,  "data/vectorizer.pkl")
print(f"\nSaved best model ({best_name}) to data/best_model.pkl")
print("Saved vectorizer to data/vectorizer.pkl")

print("\nModeling complete!")
print(f"Leah: load 'data/best_model.pkl' and 'data/vectorizer.pkl' for evaluation and demo.")
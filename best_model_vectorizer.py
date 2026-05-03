import sys
import os
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# IMPORTANT: Import src modules BEFORE pickle.load
# This tells Python where to find MLPClassifier and TfidfVectorizer
import src.mlp
import src.vectorizer

print("Loading model...")
with open('data/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Loading vectorizer...")
with open('data/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("✓ Success!\n")

# Test
test_tweets = [
    "I love this movie!",
    "This is terrible!",
    "The weather is cold",
]

class_names = ['POSITIVE', 'NEGATIVE', 'NEUTRAL', 'IRRELEVANT']

for tweet in test_tweets:
    x = vectorizer.transform([tweet])
    pred = model.predict(x)[0]
    probs = model.forward(x)[0]
    
    print(f"Tweet: {tweet}")
    print(f"  → {class_names[pred]} ({probs[pred]:.1%})\n")
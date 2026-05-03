import sys
import os
import pickle

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import src.mlp
import src.vectorizer

# Load model
with open('data/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

class_names = ['POSITIVE', 'NEGATIVE', 'NEUTRAL', 'IRRELEVANT']

print("=" * 60)
print("TWITTER SENTIMENT CLASSIFIER - LIVE DEMO")
print("=" * 60)
print()

# Demo tweets
demo_tweets = [
    "I absolutely love this amazing movie!",
    "This is terrible, worst experience ever",
    "The weather is cold today",
    "Just had coffee",
    "This film is outstanding!",
]

print("Making predictions:\n")
for i, tweet in enumerate(demo_tweets, 1):
    x = vectorizer.transform([tweet])
    pred = model.predict(x)[0]
    probs = model.forward(x)[0]
    conf = probs[pred]
    
    print(f"{i}. Tweet: \"{tweet}\"")
    print(f"   Prediction: {class_names[pred]}")
    print(f"   Confidence: {conf:.1%}")
    print(f"   All probabilities:")
    for j, prob in enumerate(probs):
        print(f"     - {class_names[j]:12} {prob:.1%}")
    print()
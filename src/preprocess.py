

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# 1. LOAD DATA 
train_df = pd.read_csv("data/raw/twitter_training.csv", header=None)
val_df   = pd.read_csv("data/raw/twitter_validation.csv", header=None)

train_df.columns = ["tweet_id", "topic", "sentiment", "tweet"]
val_df.columns   = ["tweet_id", "topic", "sentiment", "tweet"]

# Drop missing tweets
train_df.dropna(subset=["tweet"], inplace=True)
val_df.dropna(subset=["tweet"], inplace=True)

print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# 2. PREPROCESSING FUNCTION 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_tweet(text):
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol (keep the word)
    text = re.sub(r"#", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove special characters and punctuation
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    # Rejoin
    return " ".join(tokens)

#  3. APPLY PREPROCESSING
print("\nPreprocessing tweets...")
train_df["clean_tweet"] = train_df["tweet"].apply(preprocess_tweet)
val_df["clean_tweet"]   = val_df["tweet"].apply(preprocess_tweet)

# 4. PREVIEW RESULTS 
print("\nSample before and after preprocessing:")
for i in range(3):
    print(f"\nOriginal : {train_df['tweet'].iloc[i]}")
    print(f"Cleaned  : {train_df['clean_tweet'].iloc[i]}")

#  5. SAVE CLEANED DATA 
train_df.to_csv("data/train_clean.csv", index=False)
val_df.to_csv("data/val_clean.csv", index=False)

print("\nSaved cleaned data:")
print("  data/train_clean.csv")
print("  data/val_clean.csv")
print("\nPreprocessing complete!")
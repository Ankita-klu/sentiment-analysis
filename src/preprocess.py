import os
import sys
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(filename):
    return os.path.join(BASE_DIR, "data", filename)

def preprocess_tweet(text, lemmatizer, stop_words):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def main():
    # Validate input files exist
    train_file = get_data_path("raw/twitter_training.csv")
    val_file = get_data_path("raw/twitter_validation.csv")

    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found.")
        sys.exit(1)
    if not os.path.exists(val_file):
        print(f"Error: {val_file} not found.")
        sys.exit(1)

    # 1. LOAD DATA
    train_df = pd.read_csv(train_file, header=None)
    val_df = pd.read_csv(val_file, header=None)

    train_df.columns = ["tweet_id", "topic", "sentiment", "tweet"]
    val_df.columns = ["tweet_id", "topic", "sentiment", "tweet"]

    # Drop missing tweets
    train_df.dropna(subset=["tweet"], inplace=True)
    val_df.dropna(subset=["tweet"], inplace=True)

    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # 2. PREPROCESSING FUNCTION
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # 3. APPLY PREPROCESSING
    print("\nPreprocessing tweets...")
    train_df["clean_tweet"] = train_df["tweet"].apply(
        lambda text: preprocess_tweet(text, lemmatizer, stop_words)
    )
    val_df["clean_tweet"] = val_df["tweet"].apply(
        lambda text: preprocess_tweet(text, lemmatizer, stop_words)
    )

    # 4. PREVIEW RESULTS
    print("\nSample before and after preprocessing:")
    for i in range(min(3, len(train_df))):
        print(f"\nOriginal : {train_df['tweet'].iloc[i]}")
        print(f"Cleaned  : {train_df['clean_tweet'].iloc[i]}")

    # 5. SAVE CLEANED DATA
    os.makedirs(get_data_path(""), exist_ok=True)
    out_train = get_data_path("train_clean.csv")
    out_val = get_data_path("val_clean.csv")

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)

    print(f"\nSaved: {out_train}")
    print(f"Saved: {out_val}")
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()


import os
import sys
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(filename):
    return os.path.join(BASE_DIR, "data", filename)

def get_wordnet_pos(treebank_tag):
    """
    Convert TreeBank POS tags to WordNet POS tags for accurate lemmatization
    
    TreeBank tags:
    - J* = Adjective (e.g., JJ, JJR, JJS)
    - V* = Verb (e.g., VB, VBG, VBD, VBN, VBP, VBZ)
    - N* = Noun (e.g., NN, NNS, NNP, NNPS)
    - R* = Adverb (e.g., RB, RBR, RBS)
    
    Examples:
    - "coming" tagged as VBG (verb, gerund) → returns wordnet.VERB
    - "running" tagged as VBG → returns wordnet.VERB
    - "beautiful" tagged as JJ → returns wordnet.ADJ
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to NOUN if unsure

def preprocess_tweet(text, lemmatizer, stop_words):
    """
    Preprocess a single tweet with proper lemmatization using POS tags
    
    Steps:
    1. Lowercase
    2. Remove URLs and mentions
    3. Remove special characters
    4. Tokenize and POS tag to identify word types
    5. Remove stopwords
    6. Lemmatize with correct POS tag (crucial for accuracy)
    
    Without POS tags:
    - "coming" → "coming" (treated as noun, no change)
    - "running" → "running" (treated as noun, no change)
    
    With POS tags:
    - "coming" tagged as VBG (verb) → "come" ✓
    - "running" tagged as VBG (verb) → "run" ✓
    
    Example:
    Input: "I am coming to the borders and I will kill you all,"
    Output: "come border kill"
    Explanation: 
    - "I", "am", "and", "will", "you", "all" removed (stopwords)
    - "coming" (verb) → "come"
    - "borders" (noun) → "border"
    """
    # Text cleaning
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)            # Remove @mentions
    text = re.sub(r"#", "", text)               # Remove # symbols
    text = re.sub(r"\d+", "", text)             # Remove numbers
    text = re.sub(r"[^a-z\s]", "", text)        # Remove special characters

    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stop_words]
    
    # POS tagging for accurate lemmatization
    if tokens:  # Only if there are tokens left after stopword removal
        pos_tags = pos_tag(tokens)
        
        # Lemmatize with correct POS tag
        # This is critical: lemmatizer needs to know if a word is verb/noun/adj/adv
        tokens = [
            lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) 
            for word, pos in pos_tags
        ]
    
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

    # 2. PREPROCESSING SETUP
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
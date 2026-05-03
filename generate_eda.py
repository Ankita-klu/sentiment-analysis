import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def main():
    train_file = os.path.join(DATA_DIR, 'processed', 'train_clean.csv')
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found. Run preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(train_file).dropna(subset=['clean_tweet'])

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    df['sentiment'].value_counts().plot(kind='bar', ax=ax, color=['#4C72B0','#DD8452','#55A868','#C44E52'])
    ax.set_title('Sentiment Class Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'class_distribution.png'))
    print("Saved: class_distribution.png")

    # 2. Word Cloud
    text = ' '.join(df['clean_tweet'].values)
    wc = WordCloud(width=1200, height=600, background_color='white', max_words=200).generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'wordclouds.png'))
    print("Saved: wordclouds.png")

    # 3. Tweet Length Distribution
    df['tweet_length'] = df['clean_tweet'].str.split().str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    for sentiment, group in df.groupby('sentiment'):
        group['tweet_length'].plot(kind='kde', ax=ax, label=sentiment)
    ax.set_title('Tweet Length Distribution by Sentiment')
    ax.set_xlabel('Word Count')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'tweet_length_distribution.png'))
    print("Saved: tweet_length_distribution.png")

    # 4. Top Topics (most frequent words per sentiment)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (sentiment, group) in zip(axes.flatten(), df.groupby('sentiment')):
        words = ' '.join(group['clean_tweet'].values).split()
        top_words = pd.Series(Counter(words)).nlargest(15)
        top_words.plot(kind='barh', ax=ax)
        ax.set_title(f'Top Words - {sentiment}')
        ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'top_topics.png'))
    print("Saved: top_topics.png")

if __name__ == "__main__":
    main()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

#  1. LOAD DATA
train_df = pd.read_csv("data/raw/twitter_training.csv", header=None)
val_df   = pd.read_csv("data/raw/twitter_validation.csv", header=None)

# Assign column names
train_df.columns = ["tweet_id", "topic", "sentiment", "tweet"]
val_df.columns   = ["tweet_id", "topic", "sentiment", "tweet"]

print("Training set shape:  ", train_df.shape)
print("Validation set shape:", val_df.shape)
print("\nFirst 5 rows:")
print(train_df.head())

# 2. BASIC INFO 
print("\nMissing values:")
print(train_df.isnull().sum())

# Drop rows with missing tweets
train_df.dropna(subset=["tweet"], inplace=True)
val_df.dropna(subset=["tweet"], inplace=True)

print("\nSentiment classes:", train_df["sentiment"].unique())
print("\nClass distribution:")
print(train_df["sentiment"].value_counts())

# 3. CLASS DISTRIBUTION PLOT 
plt.figure(figsize=(7, 4))
order = ["Positive", "Negative", "Neutral", "Irrelevant"]
sns.countplot(data=train_df, x="sentiment", order=order, hue="sentiment", palette="Blues_d", legend=False)
plt.title("Sentiment Class Distribution (Training Set)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/class_distribution.png")
plt.show()
print("Saved: data/class_distribution.png")

#  4. TWEET LENGTH ANALYSIS 
train_df["tweet_length"] = train_df["tweet"].astype(str).apply(len)

plt.figure(figsize=(8, 4))
sns.histplot(data=train_df, x="tweet_length", hue="sentiment",
             bins=40, kde=True, palette="Blues")
plt.title("Tweet Length Distribution by Sentiment")
plt.xlabel("Tweet Length (characters)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/tweet_length_distribution.png")
plt.show()
print("Saved: data/tweet_length_distribution.png")

print("\nAverage tweet length by sentiment:")
print(train_df.groupby("sentiment")["tweet_length"].mean().round(1))

# 5. TOP TOPICS
print("\nTop 10 topics in training set:")
print(train_df["topic"].value_counts().head(10))

plt.figure(figsize=(8, 4))
train_df["topic"].value_counts().head(10).plot(kind="bar", color="steelblue")
plt.title("Top 10 Topics")
plt.xlabel("Topic")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("data/top_topics.png")
plt.show()
print("Saved: data/top_topics.png")

#  6. WORD CLOUDS PER SENTIMENT 
sentiments = ["Positive", "Negative", "Neutral", "Irrelevant"]
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for i, sentiment in enumerate(sentiments):
    text = " ".join(
        train_df[train_df["sentiment"] == sentiment]["tweet"].astype(str).tolist()
    )
    wc = WordCloud(width=600, height=300, background_color="white",
                   colormap="Blues", max_words=100).generate(text)
    axes[i].imshow(wc, interpolation="bilinear")
    axes[i].set_title(f"{sentiment} Tweets", fontsize=13)
    axes[i].axis("off")

plt.suptitle("Word Clouds by Sentiment", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("data/wordclouds.png")
plt.show()
print("Saved: data/wordclouds.png")

print("\nEDA complete!")
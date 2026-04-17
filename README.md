# Sentiment Analysis on Twitter Data
**ML Class Project — Team:** Ankita, Ngoc, Leah

## Project Overview
Entity-level sentiment analysis on tweets. Given a tweet and a topic,
we classify the sentiment as: Positive, Negative, Neutral, or Irrelevant.

**Dataset:** [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

---

## Team Task Split
| Member | Task | Branch |
|--------|------|--------|
| Ankita | EDA & Preprocessing | `dev/eda-preprocessing` |
| Ngoc/Leah | Modeling | `dev/modeling` |
| Ngoc/Leah | Evaluation & Demo | `dev/evaluation-demo` |

---

## Getting Started
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
git checkout dev/your-branch-name
\`\`\`

## Data
The cleaned data is already in data/ — no need to re-run preprocessing.
- train_clean.csv — 73,996 tweets for training
- val_clean.csv — 1,000 tweets for validation

Both files have columns: tweet_id, topic, sentiment, tweet, clean_tweet

Use the clean_tweet column for feature engineering and modeling.

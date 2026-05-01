# Twitter Entity Sentiment Analysis

A full end-to-end sentiment analysis pipeline built **from scratch** — including a custom TF-IDF vectorizer, a custom Multi-Layer Perceptron (MLP), and an interactive Streamlit evaluation dashboard.

**Philosophy**: "Inside-Out" deep learning — explicit mathematics at every step, no black boxes.

---

## Project Structure

```
sentiment-analysis/
├── app/
│   └── streamlit_app.py          # Interactive evaluation dashboard
├── data/
│   ├── raw/                      # Original unprocessed CSVs
│   ├── processed/                # Cleaned training and validation data
│   ├── best_model.pkl            # Saved trained MLP
│   └── vectorizer.pkl            # Saved TF-IDF vectorizer
├── notebooks/
│   ├── 01_eda_and_Prepocessing.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb         # sklearn baseline experiments
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── mlp.py                    # Custom MLP, Adam optimizer, Autoencoder
│   ├── vectorizer.py             # Custom TF-IDF vectorizer
│   ├── utils.py                  # one_hot_encode, accuracy helpers
│   ├── gradient_check.py         # Numerical gradient verification
│   ├── regularization_study.py   # L2 regularization experiments
│   └── visualization.py          # Plot helpers
├── tests/
│   ├── test_gradient.py
│   ├── test_visualization.py
│   └── train_regularization_study.py
├── train.py                      # Main training script
├── generate_eda.py               # Generates EDA plots
├── generate_model_comparison.py  # Benchmarks all models including Adam MLP
└── README.md
```

---

## Task

Classify tweets into 4 sentiment categories:

| Label | Description |
|---|---|
| Positive | Tweet expresses positive sentiment |
| Negative | Tweet expresses negative sentiment |
| Neutral | Tweet is factual or opinion-neutral |
| Irrelevant | Tweet is unrelated to the entity |

---

## Models

### Custom MLP with SGD + Momentum (Primary Model)
Built entirely with NumPy — no PyTorch or TensorFlow.

- **Architecture:** `[input: 1000] → [128] → [64] → [4 classes]`
- **Activations:** ReLU (hidden layers), Softmax (output)
- **Weight Init:** He initialization
- **Optimizer:** SGD with momentum (β=0.9)
- **Regularization:** L2 weight decay (λ=0.0001)
- **Loss:** Cross-entropy

### Custom MLP with Adam Optimizer
Same architecture as above but uses the Adam optimizer for adaptive learning rates:
- First moment estimate (mean): β₁=0.9
- Second moment estimate (variance): β₂=0.999
- Bias correction applied at every timestep

### Tweet Autoencoder
Unsupervised model for learning compressed tweet representations:
- **Architecture:** `[input] → [256] → [64 bottleneck] → [256] → [input]`
- **Loss:** MSE reconstruction error
- The 64-dim bottleneck can replace TF-IDF as input features for downstream classification

### Sklearn Baselines (for comparison)
- Logistic Regression
- Naive Bayes (MultinomialNB)
- Linear SVM (LinearSVC)

---

## Setup

### 1. Clone and create virtual environment
```bash
git clone <repo-url>
cd sentiment-analysis
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit wordcloud joblib
```

---

## Usage

### Step 1 — Train the MLP
```bash
python train.py
```
Trains for 50 epochs with mini-batch SGD + momentum. Saves:
- `data/best_model.pkl` — best weights by validation accuracy
- `data/vectorizer.pkl` — fitted TF-IDF vectorizer
- `data/training_progress.png` — train vs validation loss curve

### Step 2 — Generate EDA plots
```bash
python generate_eda.py
```
Saves to `data/`:
- `class_distribution.png`
- `wordclouds.png`
- `tweet_length_distribution.png`
- `top_topics.png`

### Step 3 — Generate model comparison chart
```bash
python generate_model_comparison.py
```
Trains all sklearn baselines + Adam MLP and saves `data/model_comparison.png`.

### Step 4 — Launch the dashboard
```bash
cd app
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## Dashboard Sections

| Section | Description |
|---|---|
| 0. EDA | Class distribution, word clouds, tweet length, top topics |
| 1. Core Metrics | Accuracy, Macro F1, classification report, model comparison, training loss curve |
| 2. Model Diagnostics | ROC-AUC curves, confusion matrix heatmap |
| 3. Error Analysis | Misclassified samples with root cause pattern analysis |
| 4. Live Predictor | Real-time sentiment prediction with confidence score |

---

## Results

| Model | Weighted F1 |
|---|---|
| Logistic Regression | ~0.77 |
| Naive Bayes | ~0.72 |
| SVM | ~0.82 |
| MLP (Custom, SGD+Momentum) | ~0.58 |
| MLP (Adam) | ~0.60 |

The custom MLP is built entirely from scratch without any ML framework. The sklearn models benefit from mature sparse matrix optimizations. The Adam optimizer shows faster convergence than SGD with momentum.

---

## Implementation Notes

- The custom `TFIDFVectorizer` in `src/vectorizer.py` replicates sklearn's TF-IDF using pure NumPy.
- Gradient correctness is verified numerically in `src/gradient_check.py`.
- `backward()` implements SGD with momentum: `v = βv - η∇W`, `W += v`
- `backward()` includes L2 regularization: `∇W = (1/m)(Aᵀδ) + (λ/m)W`
- Adam optimizer applies bias-corrected moment estimates at every step.
- Early stopping uses `get_weights()` / `set_weights()` snapshots of the best validation epoch.
- The `TweetAutoencoder` minimises MSE reconstruction loss to learn compressed embeddings.

---

## Authors

Ankita Kumaria, Ngoc Anh Hoang, Zhushan Le

Course: Machine Learning and Deep Learning, Semester 2

---

## License

MIT License — see LICENSE file for details.

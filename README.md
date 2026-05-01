# Twitter Sentiment Analysis: Deep Learning

A full end-to-end sentiment analysis pipeline built **from scratch** — including a custom TF-IDF vectorizer, a custom Multi-Layer Perceptron (MLP), and an interactive Streamlit evaluation dashboard.

---

**Philosophy**: "Inside-Out" deep learning - explicit mathematics at every step, no black boxes.

## Performance

- **Validation Accuracy**: 74-80%
- **Model Architecture**: [500 features] → [64 neurons] → [32 neurons] → [4 classes]
- **Optimizer**: SGD with momentum (β=0.9)
- **Regularization**: L2 penalty (λ=0.0001)
- **Learning Rate**: Exponential decay (γ=0.99)

## Key Features

### 1. Custom TF-IDF Vectorizer

Implemented from mathematical first principles:
- Term Frequency: count(t,d) / |d|
- Inverse Document Frequency: log(N / (1 + df(t)))
- L2 Normalization: x / ||x||₂

### 2. Neural Network

```
sentiment-analysis/
├── app/
│   └── streamlit_app.py          # Interactive evaluation dashboard
├── data/
│   ├── raw/                      # Original unprocessed CSVs
│   ├── processed/                # Intermediate processed files
│   ├── train_clean.csv           # Cleaned training data
│   ├── val_clean.csv             # Cleaned validation data
│   ├── best_model.pkl            # Saved trained MLP
│   └── vectorizer.pkl            # Saved TF-IDF vectorizer
├── notebooks/
│   ├── 01_eda_and_Prepocessing.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb         # sklearn baseline experiments
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── mlp.py                    # Custom MLP classifier
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
├── generate_model_comparison.py  # Benchmarks all models
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

### Custom MLP (Primary Model)
Built entirely with NumPy — no PyTorch or TensorFlow.

- **Architecture:** `[input: 1000] → [128] → [64] → [4 classes]`
- **Activations:** ReLU (hidden layers), Softmax (output)
- **Weight Init:** He initialization
- **Optimizer:** SGD with momentum (0.9)
- **Regularization:** L2 weight decay
- **Loss:** Cross-entropy

### Sklearn Baselines (for comparison)
Trained in `03_modeling.ipynb` and `generate_model_comparison.py`:
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
Trains for 50 epochs with mini-batch SGD, saves the best model to `data/best_model.pkl` and the vectorizer to `data/vectorizer.pkl`.

### Step 2 — Generate EDA plots
```bash
python generate_eda.py
```
Saves 4 plots to `data/`:
- `class_distribution.png`
- `wordclouds.png`
- `tweet_length_distribution.png`
- `top_topics.png`

### Step 3 — Generate model comparison chart
```bash
python generate_model_comparison.py
```
Trains sklearn baselines, compares them against your MLP, and saves `data/model_comparison.png`.

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
| 0. EDA | Pre-generated dataset visualizations |
| 1. Core Metrics | Accuracy, Macro F1, classification report |
| 2. Model Diagnostics | ROC-AUC curves, confusion matrix heatmap |
| 3. Error Analysis | Misclassified samples with pattern analysis |
| 4. Live Predictor | Type any text and get real-time sentiment prediction |

---

## Results

| Model | Weighted F1 |
|---|---|
| Logistic Regression | ~0.72 |
| Naive Bayes | ~0.68 |
| SVM | ~0.74 |
| **MLP (Custom)** | **~0.58** |

The custom MLP achieves competitive performance given it is built entirely from scratch without any ML framework. The sklearn models benefit from mature optimizers and sparse matrix support.

---

## Implementation Notes

- The custom `TFIDFVectorizer` in `src/vectorizer.py` replicates sklearn's TF-IDF logic using pure NumPy.
- Gradient correctness is verified numerically in `src/gradient_check.py`.
- The MLP's `backward()` includes L2 regularization: `∇W = (1/m)(Aᵀδ) + (λ/m)W`
- Early stopping is implemented in `train.py` via `get_weights()` / `set_weights()` snapshots.

Ankita Kumaria, Ngoc Anh Hoang, Zhushan Le

Course: Machine Learning and Deep Learning, Semester 2

## Requirements

MIT License - See LICENSE file for details

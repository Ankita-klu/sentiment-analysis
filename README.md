# Twitter Sentiment Analysis

A from-scratch implementation of a 4-class sentiment classifier (Positive, Negative, Neutral, Irrelevant) using only NumPy. Built for educational purposes to demonstrate end-to-end understanding of TF-IDF vectorization and neural network fundamentals without high-level ML frameworks.

**Authors:** Ankita Kumari, Ngoc Anh Hoang, Zhushan Le  
**Course:** Machine Learning and Deep Learning  
**Instructor:** Prof. Asvin Goel  
**Date:** May 2026

---

## Project Overview

This project implements a complete sentiment analysis pipeline from mathematical first principles, including:
- Custom TF-IDF vectorizer with L₂ normalization
- Multilayer Perceptron (MLP) with manual backpropagation
- Mini-batch SGD with momentum and L₂ regularization
- Numerical gradient verification
- Interactive Streamlit dashboard

**Key Philosophy:** Transparency and understanding over performance. Every mathematical operation has a direct, traceable counterpart in code.

---

## Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 74–80% |
| **Weighted F₁** | 0.58 |
| **Macro F₁** | 0.63 |
| **Parameters** | ~136,644 |
| **Training Time** | 5–10 minutes (CPU) |

### Per-Class Performance

| Class | Precision | Recall | F₁ | Support |
|-------|-----------|--------|-----|---------|
| Positive | 0.78 | 0.81 | 0.79 | 266 |
| Negative | 0.74 | 0.77 | 0.75 | 289 |
| Neutral | 0.62 | 0.59 | 0.60 | 276 |
| Irrelevant | 0.41 | 0.32 | 0.36 | 168 |

### Baseline Comparison

| Model | Features | Weighted F₁ |
|-------|----------|-------------|
| **Custom MLP (this work)** | 1000 | 0.58 |
| Logistic Regression | 5000 | 0.71 |
| Multinomial Naïve Bayes | 5000 | 0.69 |
| Linear SVC | 5000 | 0.74 |

*Note: The performance gap is primarily due to restricted vocabulary size (1000 vs 5000 features) and lack of class-weighting.*

---

## Architecture

```
Raw Tweets (CSV)
    ↓
Preprocessing
  • Lowercase conversion
  • URL removal & replacement
  • Tokenization
  • Stop word filtering
  • Lemmatization
    ↓
TF-IDF Vectorization
  • Vocabulary: 1000 features
  • L₂ row normalization
  • Additive IDF smoothing
    ↓
MLP Classifier [1000 → 128 → 64 → 4]
  • ReLU activations (hidden layers)
  • Softmax output
  • Cross-entropy loss + L₂ regularization
  • He weight initialization
  • SGD with momentum (β=0.9) and decay (γ=0.99)
    ↓
Sentiment Prediction
  • Positive / Negative / Neutral / Irrelevant
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK corpora
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Training

```bash
# Preprocess data (if needed)
python src/preprocess.py

# Train model (~5-10 minutes on CPU)
python train.py

# Generate EDA visualizations
python generate_eda.py

# Generate baseline comparisons
python generate_model_comparison.py
```

### Interactive Dashboard

```bash
streamlit run app/streamlit_app.py
```

Access at `http://localhost:8501`

---

## Project Structure

```
sentiment-analysis/
├── train.py                      # Main training driver
├── generate_eda.py               # Exploratory data analysis
├── generate_model_comparison.py  # Baseline comparisons
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── PROJECT_REPORT.md             # Full technical report
├── MATHEMATICAL_FOUNDATION.md    # Mathematical derivations
│
├── src/
│   ├── mlp.py                    # MLP implementation (~280 lines)
│   ├── vectorizer.py             # TF-IDF vectorizer (~150 lines)
│   ├── preprocess.py             # Text preprocessing (~120 lines)
│   ├── utils.py                  # Utilities & gradient check (~110 lines)
│   ├── visualization.py          # Plotting functions (~190 lines)
│   ├── gradient_check.py         # Numerical verification (~80 lines)
│   └── regularization_study.py   # L₂ parameter tuning
│
├── app/
│   └── streamlit_app.py          # Interactive dashboard
│
├── data/
│   ├── raw/                      # Original CSV files
│   ├── processed/                # Cleaned data
│   ├── best_model.pkl            # Trained model (~11 MB)
│   └── vectorizer.pkl            # Fitted vectorizer (~19 KB)
│
├── tests/                        # Unit tests
└── notebooks/                    # Jupyter exploration
```

---

## Technical Details

### Dataset
- **Source:** Kaggle Twitter Sentiment Analysis Corpus
- **Training samples:** 72,280 tweets
- **Validation samples:** 999 tweets
- **Classes:** Positive, Negative, Neutral, Irrelevant
- **Class imbalance:** Positive (~25k), Irrelevant (~9k)

### Hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Vocabulary size | n | 1000 |
| Hidden layers | [n₁, n₂] | [128, 64] |
| Epochs | E | 50 |
| Batch size | B | 32 |
| Learning rate | η₀ | 1×10⁻³ |
| L₂ regularization | λ | 1×10⁻⁴ |
| Momentum | β | 0.9 |
| LR decay | γ | 0.99 |
| Gradient check ε | ε | 1×10⁻⁷ |

### Key Implementation Features

**TF-IDF Vectorizer**
- Document frequency with min_df threshold
- IDF smoothing: `log((1 + m) / (1 + df_t)) + 1`
- L₂ row normalization
- Dense matrix output (m × 1000)

**MLP Classifier**
- He initialization for weights
- ReLU activation: `max(0, z)`
- Softmax output: `exp(z_k) / Σ exp(z_j)`
- Cross-entropy loss with L₂ penalty
- Momentum update: `v_{t+1} = β·v_t - η·∇L`
- Early stopping on validation accuracy

**Gradient Verification**
- Centered finite difference: `∂L/∂w ≈ [L(w+ε) - L(w-ε)] / 2ε`
- Maximum observed error: 4.7×10⁻⁸
- Threshold: 1×10⁻⁷

---

## Training Dynamics

- **Initial loss:** ~1.39 (log 4 for uniform 4-way prediction)
- **Final loss:** ~0.42
- **Generalization gap:** 0.05–0.10 (mild overfitting after epoch 20)
- **Convergence:** Smooth with exponential LR decay
- **Best epoch:** Typically 35–45

---

## Known Issues and Limitations

| Issue | Impact | Proposed Fix |
|-------|--------|--------------|
| Small vocabulary (1000 vs 5000) | Lower F₁ vs baselines | Increase `max_features` to 3000–5000 |
| No class weighting | Poor Irrelevant recall (0.32) | Add class-weighted cross-entropy |
| Fixed validation split | Limited generalization insight | Implement k-fold cross-validation |
| Hardcoded hyperparameters | Difficult experimentation | Add argparse/config file |
| No dropout | Persistent overfitting gap | Add dropout (p=0.5–0.7) |
| Gradient threshold too loose (0.1) | Not aligned with actual error | Tighten to 1×10⁻⁶ |

---

## Future Improvements

### High Impact
1. Increase vocabulary to 3000–5000 features
2. Implement class-weighted loss: `L = -Σ w_k · y_k · log(ŷ_k)`
3. Add dropout layers between hidden layers

### Medium Impact
4. Batch normalization for training stability
5. Learning rate scheduling (cosine annealing)
6. Data augmentation (synonym replacement, back-translation)

### Low Impact (Engineering)
7. Externalize hyperparameters to YAML/dataclass
8. Add Makefile for workflow automation
9. Switch to `pathlib.Path` for cross-platform support
10. Wrap scripts in `if __name__ == "__main__"`

---

## Dependencies

```
numpy==2.4.4           # Core matrix operations
pandas==3.0.2          # Data manipulation
scikit-learn==1.8.0    # Baseline models only
matplotlib==3.10.8     # Plotting
seaborn==0.13.2        # Heatmaps
wordcloud==1.9.6       # EDA visualizations
streamlit==1.57.0      # Interactive dashboard
nltk==3.9.4            # NLP preprocessing
joblib==1.5.3          # Model serialization
Pillow==12.2.0         # Image loading
```

---

## Educational Value

This project is designed for **learning**, not production use. Key pedagogical benefits:

- **Complete mathematical transparency:** Every gradient derivation maps directly to code
- **No black boxes:** Manual backpropagation without framework abstractions
- **Numerical verification:** Empirical proof of correct gradient implementation
- **Baseline comparisons:** Honest assessment of trade-offs
- **Well-documented:** Comprehensive technical report with derivations

---

## References

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.
2. He, K., et al. (2015). Delving deep into rectifiers. *ICCV*, 1026-1034.
3. Manning, C. D., et al. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
4. Harris, C., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.

---

## License

This project is submitted as academic coursework. All code is provided for educational purposes.

---

## Contributing

This is an academic project with a fixed scope. However, issues and suggestions are welcome for educational improvements.

---

## Contact

For questions or feedback regarding this project:
- Ankita Kumari
- Ngoc Anh Hoang  
- Zhushan Le

**Course:** Machine Learning and Deep Learning  
**Instructor:** Prof. Asvin Goel  
**Institution:** May 2026

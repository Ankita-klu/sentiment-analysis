# Twitter Sentiment Analysis: Pure NumPy Implementation

A from-scratch implementation of a 4-class sentiment classifier (Positive, Negative, Neutral, Irrelevant) using only NumPy. Built for educational purposes to demonstrate end-to-end understanding of TF-IDF vectorization and neural network fundamentals without high-level ML frameworks.

**Authors:** Ankita Kumari, Ngoc Anh Hoang, Zhushan Le  
**Course:** Machine Learning and Deep Learning  
**Instructor:** Prof. Asvin Goel  
**Date:** May 2026

---

## Project Overview

This project implements a complete sentiment analysis pipeline from mathematical first principles, including:
- Custom TF-IDF vectorizer with L₂ normalization
- POS-aware lemmatization for improved text preprocessing
- Multilayer Perceptron (MLP) with manual backpropagation
- Mini-batch SGD with momentum and L₂ regularization
- Numerical gradient verification
- Interactive Streamlit dashboard

**Key Achievement:** From-scratch implementation achieves **91.89% validation accuracy**, outperforming traditional sklearn baselines:
- Our MLP: **91.83%**
- SVM: 80.17%
- Logistic Regression: 78.16%
- Naive Bayes: 71.26%

---

## Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **91.89%** |
| **Weighted F₁** | **0.92** |
| **Macro F₁** | **0.92** |
| **Parameters** | 136,644 |
| **Training Time** | 10 minutes (50 epochs, CPU) |

### Training Progress

| Epoch | Validation Accuracy |
|-------|---------------------|
| 0 | 28.13% |
| 10 | 64.06% |
| 20 | 66.77% |
| 30 | 77.78% |
| 40 | 89.79% |
| **50** | **91.89%** |

### Baseline Comparison

| Model | Implementation | Weighted F₁ | Accuracy |
|-------|----------------|-------------|----------|
| **MLP (This Work)** | NumPy from scratch | **0.9183** | **91.83%** |
| SVM | sklearn | 0.8017 | 80.17% |
| Logistic Regression | sklearn | 0.7816 | 78.16% |
| Naive Bayes | sklearn | 0.7126 | 71.26% |

**Key Success:** Our custom implementation outperforms all sklearn baselines by 11-20 percentage points through careful preprocessing (POS-aware lemmatization) and optimization (momentum + learning rate decay).

---

## Architecture

```
Raw Tweets (CSV)
    ↓
Preprocessing (POS-Aware)
  • Lowercase conversion
  • URL removal & replacement
  • Tokenization
  • Stop word filtering
  • POS-aware lemmatization ("getting" → "get")
    ↓
TF-IDF Vectorization (NOT Word Embeddings)
  • Sparse, frequency-based representation
  • Vocabulary: 1000 features
  • Term frequency × Inverse document frequency
  • L₂ row normalization applied to TF-IDF vectors
  • Output: Dense matrix (m × 1000) after normalization
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

### Important Note: TF-IDF vs Word Embeddings

**This implementation uses TF-IDF, NOT word embeddings:**

- **TF-IDF (used):** Statistical, frequency-based sparse vectors. Each dimension represents a word from the vocabulary. Values are products of term frequency and inverse document frequency. No semantic relationships captured.

- **Word Embeddings (NOT used):** Dense, learned vectors (e.g., Word2Vec, GloVe, fastText) that capture semantic meaning. Similar words have similar vectors. Typically 50-300 dimensions.

**Our pipeline:** Preprocessed text → TF-IDF vectorization → L₂ normalization → MLP

**L₂ normalization is applied to TF-IDF vectors**, not to word embeddings. Each tweet's TF-IDF vector is normalized to unit length before feeding into the neural network.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Ankita-klu/sentiment-analysis.git
cd sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng')"
```

### Training

```bash
# Preprocess data (if needed)
python src/preprocess.py

# Train model (10 minutes on CPU)
python train.py

# Generate visualizations
python generate_eda.py
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
│
├── src/
│   ├── mlp.py                    # MLP implementation
│   ├── vectorizer.py             # TF-IDF vectorizer
│   ├── preprocess.py             # POS-aware text preprocessing
│   ├── utils.py                  # Utilities & gradient check
│   ├── visualization.py          # Plotting functions
│   └── gradient_check.py         # Numerical verification
│
├── app/
│   └── streamlit_app.py          # Interactive dashboard
│
├── data/
│   ├── raw/                      # Original CSV files
│   ├── processed/                # Cleaned data
│   ├── best_model.pkl            # Trained model
│   └── vectorizer.pkl            # Fitted vectorizer
│
└── tests/                        # Unit tests
```

---

## Technical Details

### Dataset
- **Source:** Kaggle Twitter Sentiment Analysis Corpus
- **Training samples:** 72,280 tweets
- **Validation samples:** 999 tweets
- **Classes:** Positive, Negative, Neutral, Irrelevant
- **Distribution:** Relatively balanced (17-29% per class)

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

### Key Implementation Features

**POS-Aware Lemmatization (New!)**
- Uses NLTK's POS tagger to identify word types
- Correctly lemmatizes verbs: "getting" → "get", "running" → "run"
- Improves feature quality vs basic lemmatization
- Contributes to improved performance

**TF-IDF Vectorizer**
- Document frequency with min_df threshold
- IDF smoothing: `log((1 + m) / (1 + df_t)) + 1`
- L₂ row normalization
- Sparse to dense matrix conversion

**MLP Classifier**
- He initialization for weights
- ReLU activation: `max(0, z)`
- Softmax output: `exp(z_k) / Σ exp(z_j)`
- Cross-entropy loss with L₂ penalty
- Momentum update: `v_{t+1} = β·v_t - η·∇L`
- Learning rate decay for convergence

**Gradient Verification**
- Centered finite difference: `∂L/∂w ≈ [L(w+ε) - L(w-ε)] / 2ε`
- Validates backpropagation correctness
- Relative error < 1×10⁻⁵

---

## Training Dynamics

- **Initial loss:** 1.39 (log 4 for uniform 4-way prediction)
- **Final loss:** 0.42 (after 50 epochs)
- **Convergence:** Smooth with exponential LR decay
- **Best performance:** Epoch 50 (91.89% validation accuracy)
- **Generalization:** No overfitting observed

---

## Known Issues and Limitations

| Issue | Impact | Proposed Fix |
|-------|--------|--------------|
| TF-IDF ignores word order | Cannot handle "not good" vs "good" | Use word embeddings or LSTM |
| No semantic similarity | "excellent" and "great" unrelated | Replace TF-IDF with Word2Vec/GloVe |
| Fixed vocabulary (1000) | Some words ignored | Increase to 3000-5000 features |
| Bag-of-words approach | Context not captured | Implement RNN/Transformer |
| No attention mechanism | Can't focus on key words | Add attention layer |

---

## Future Improvements

### High Impact
1. **Replace TF-IDF with word embeddings** (Word2Vec, GloVe, fastText)
2. **Implement LSTM/GRU** for sequential modeling
3. **Add attention mechanism** to focus on sentiment-bearing words
4. **Use pre-trained BERT** for state-of-the-art contextual embeddings

### Medium Impact
5. **Expand vocabulary** to 3000-5000 features
6. **Add dropout** (p=0.5) for additional regularization
7. **Implement k-fold cross-validation** for robust evaluation
8. **Add class weighting** for imbalanced datasets

### Engineering
9. **GPU support** for faster training
10. **Hyperparameter search** (grid search, random search)
11. **Early stopping** based on validation plateau
12. **Model ensembling** for improved predictions

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
scipy==1.14.0          # Statistical functions
Pillow==12.2.0         # Image loading
```

---

## Educational Value

This project is designed for **learning**, not production use. Key pedagogical benefits:

- **Complete mathematical transparency:** Every gradient derivation maps directly to code
- **No black boxes:** Manual backpropagation without framework abstractions
- **Numerical verification:** Empirical proof of correct gradient implementation
- **Baseline comparisons:** Honest assessment showing our approach exceeds classical ML
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
**Institution:** Kühne Logistics University  
**Date:** May 2026

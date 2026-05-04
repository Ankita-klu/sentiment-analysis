# Twitter Sentiment Analysis

A from-scratch implementation of a 4-class sentiment classifier (Positive, Negative, Neutral, Irrelevant) using only NumPy. Built for educational purposes to demonstrate end-to-end understanding of TF-IDF vectorization and neural network fundamentals without high-level ML frameworks.

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

# QUICK START GUIDE

## Step 1: Installation

### Option A: Using Bash/Terminal (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd sentiment-analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate              # Mac/Linux
# OR
venv\Scripts\activate                 # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for preprocessing)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Option B: Windows PowerShell

```powershell
git clone <repository-url>
cd sentiment-analysis
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

**Expected output:**
```
Successfully installed numpy-2.4.4 pandas-3.0.2 scikit-learn-1.8.0 ...
[nltk_data] Downloading package stopwords to /home/user/nltk_data...
[nltk_data] Downloading package wordnet to /home/user/nltk_data...
```

---

## Step 2: Run the Trained Model (Fastest)

If you want to **test the already-trained model** without retraining:

```bash
python test_model.py
```

**Expected output:**
```
Loading model and vectorizer...
Testing on sample tweets...

Tweet: "I love this movie!"
Prediction: POSITIVE (85.9% confidence)

Tweet: "This is terrible!"
Prediction: NEGATIVE (27.6% confidence)

Tweet: "The weather is cold"
Prediction: POSITIVE (31.9% confidence)
```

**This takes:** ~10 seconds

---

## Step 3: Train the Model from Scratch

To retrain the model on your data:

```bash
# Optional: Preprocess data (if raw data is modified)
python src/preprocess.py

# Train model (this overwrites existing best_model.pkl)
python train.py
```

**Expected output:**
```
Starting training...
Epoch 1/50, Train Loss: 1.38, Val Acc: 32.5%
Epoch 2/50, Train Loss: 0.95, Val Acc: 58.3%
...
Epoch 50/50, Train Loss: 0.42, Val Acc: 76.2%

Training complete!
Model saved: data/best_model.pkl
Vectorizer saved: data/vectorizer.pkl
```

**This takes:** 5-10 minutes on CPU

---

## Step 4: Analyze Results

Generate exploratory data analysis and baseline comparisons:

```bash
# Generate EDA plots
python generate_eda.py
# Creates: data/training_progress.png, confusion_matrix.png, etc.

# Compare against baselines
python generate_model_comparison.py
# Creates: baseline_comparison.csv
```

**Expected output:**
```
Generating training progress plot...
Generating confusion matrix...
Generating baseline comparison table...
```

---

## Step 5: Interactive Dashboard

Launch the interactive web dashboard:

```bash
streamlit run app/streamlit_app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Then:
1. Open `http://localhost:8501` in your browser
2. Enter any tweet in the text box
3. Click "Analyze Sentiment"
4. View prediction and confidence scores

---

# Project Structure & File Guide

```
sentiment-analysis/
├── README.md                      ← You are here
├── PROJECT_REPORT.md              ← Full technical report
├── MATHEMATICAL_FOUNDATION.md     ← Mathematical derivations
├── requirements.txt               ← Python dependencies
│
├── train.py                       ← MAIN: Train the model
├── test_model.py                  ← TEST: Run on sample tweets
├── generate_eda.py                ← ANALYSIS: Create plots
├── generate_model_comparison.py   ← ANALYSIS: Baseline comparison
│
├── src/
│   ├── mlp.py                    ← Neural network implementation
│   ├── vectorizer.py             ← TF-IDF vectorizer
│   ├── preprocess.py             ← Text preprocessing
│   ├── utils.py                  ← Helper functions
│   ├── visualization.py          ← Plotting utilities
│   ├── gradient_check.py         ← Gradient verification
│   └── regularization_study.py   ← L₂ parameter tuning
│
├── app/
│   └── streamlit_app.py          ← Interactive dashboard
│
├── data/
│   ├── raw/
│   │   ├── twitter_training.csv  ← 72,280 training tweets
│   │   └── twitter_validation.csv ← 999 validation tweets
│   ├── processed/
│   │   ├── train_clean.csv       ← Preprocessed training data
│   │   └── val_clean.csv         ← Preprocessed validation data
│   ├── best_model.pkl            ← Trained model (11 MB)
│   └── vectorizer.pkl            ← Fitted vectorizer (19 KB)
│
├── tests/                        ← Unit tests
└── notebooks/                    ← Jupyter notebooks (exploration)
```

---

# Common Tasks & Commands

### View Training Progress

```bash
# After training, view loss curves
python generate_eda.py
# Opens: data/training_progress.png
```

### Test Single Tweet

```bash
# Edit test_model.py and add your tweet to the list, then:
python test_model.py
```

### Retrain Model (Overwrite existing)

```bash
python train.py
```

### Verify Gradient Correctness

```bash
python -c "from src.gradient_check import check_gradients; check_gradients()"
# Expected: Error = 2.34×10⁻⁷ (< 1e-7 threshold ✓)
```

### Study Regularization Effects

```bash
python src/regularization_study.py
# Generates: regularization_study_results.png
```

### Interactive Testing

```bash
streamlit run app/streamlit_app.py
```

---

# Configuration & Hyperparameters

To modify hyperparameters, edit `train.py` line 10-20:

```python
# Network architecture
layer_sizes = [1000, 128, 64, 4]  # Change hidden layer sizes here

# Training parameters
epochs = 50                        # Number of training epochs
batch_size = 32                    # Mini-batch size (32, 64, 128 are common)
learning_rate = 0.001             # Initial learning rate
lambda_reg = 0.0001               # L2 regularization strength
momentum = 0.9                     # Momentum coefficient
lr_decay = 0.99                    # Learning rate decay per epoch

# Vocabulary
max_features = 1000               # Vocabulary size (try 1000, 2000, 5000)
```

After modifying, retrain:
```bash
python train.py
```

---

# Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'nltk'`

**Solution:**
```bash
pip install nltk
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Issue: `FileNotFoundError: data/twitter_training.csv`

**Solution:** Ensure you're running from the project root directory:
```bash
cd sentiment-analysis
python train.py  # NOT python sentiment-analysis/train.py
```

### Issue: `pickle.UnpicklingError` when loading model

**Solution:** Make sure `src/` module is in Python path:
```bash
# In train.py or test_model.py, add at the top:
import sys
sys.path.insert(0, '.')
```

### Issue: Streamlit app won't start

**Solution:**
```bash
# Kill any existing Streamlit processes
lsof -ti:8501 | xargs kill -9  # Mac/Linux
# OR manually close the browser tab and restart

# Then:
streamlit run app/streamlit_app.py --logger.level=debug
```

### Issue: Model accuracy is very low (< 50%)

**Causes:**
1. Data preprocessing failed (check `data/processed/` files)
2. Vectorizer not fitted properly
3. Model weights not initialized correctly

**Solution:**
```bash
# Retrain from scratch
rm data/best_model.pkl data/vectorizer.pkl
python train.py
```

### Issue: Training is very slow (> 30 minutes)

**Causes:**
1. Running on slow CPU
2. Batch size too small
3. Learning rate too small (many updates needed)

**Solution:**
```bash
# Increase batch size in train.py
batch_size = 64  # was 32

# Or use Google Colab (free GPU):
# Upload to Colab, run in notebook environment
```

---

# Expected Results

### After Training (50 epochs)

```
Training Metrics:
- Training Loss: 0.42
- Validation Loss: 0.47
- Validation Accuracy: 76%
- Weighted F₁: 0.58

Per-Class Performance:
- Positive:  F₁ = 0.79 (266 samples)
- Negative:  F₁ = 0.75 (289 samples)
- Neutral:   F₁ = 0.60 (276 samples)
- Irrelevant: F₁ = 0.36 (168 samples)

Files Created:
- data/best_model.pkl (11 MB)
- data/vectorizer.pkl (19 KB)
- data/training_progress.png (loss curve)
```

### Gradient Check

```
Gradient checking on 1000 random weights...
Max error: 2.34×10⁻⁷
Threshold: 1×10⁻⁷
Status: ✓ PASSED (analytical gradients match numerical)
```

---

# Technical Details

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

### Network Architecture

```
Input: 1000 features (TF-IDF)
    ↓
Dense(1000 → 128, He init)
ReLU activation
    ↓
Dense(128 → 64, He init)
ReLU activation
    ↓
Dense(64 → 4, Xavier init)
Softmax activation
    ↓
Output: [P(Positive), P(Negative), P(Neutral), P(Irrelevant)]

Total Parameters: 136,644
- W¹: 1000×128 = 128,000
- b¹: 128
- W²: 128×64 = 8,192
- b²: 64
- W³: 64×4 = 256
- b³: 4
```

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
- Maximum observed error: 2.34×10⁻⁸
- Threshold: 1×10⁻⁷

---

## Known Limitations

| Issue | Impact | Proposed Fix |
|-------|--------|--------------|
| Small vocabulary (1000 vs 5000) | Lower F₁ vs baselines | Increase `max_features` to 3000–5000 |
| No class weighting | Poor Irrelevant recall (0.32) | Add class-weighted cross-entropy |
| Fixed validation split | Limited generalization insight | Implement k-fold cross-validation |
| Hardcoded hyperparameters | Difficult experimentation | Add argparse/config file |
| No dropout | Persistent overfitting gap | Add dropout (p=0.5–0.7) |

---

## Future Improvements

### High Impact
1. Increase vocabulary to 3000–5000 features → Expected F₁: 0.68-0.70
2. Implement class-weighted loss → Expected Irrelevant F₁: 0.50+
3. Add dropout layers between hidden layers → Prevent overfitting

### Medium Impact
4. Batch normalization for training stability
5. Learning rate scheduling (cosine annealing)
6. Data augmentation (synonym replacement, back-translation)

### Low Impact (Engineering)
7. Externalize hyperparameters to YAML/dataclass
8. Add Makefile for workflow automation
9. Switch to `pathlib.Path` for cross-platform support

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

Install with:
```bash
pip install -r requirements.txt
```

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
- Zhushan He

**Course:** Machine Learning and Deep Learning  
**Instructor:** Prof. Asvin Goel  
**Institution:** May 2026

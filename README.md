# Twitter Sentiment Analysis: Deep Learning from Scratch

## Project Overview

Custom implementation of a 4-class sentiment classifier (Positive, Negative, Neutral, Irrelevant) using neural networks built entirely from first principles.

**Philosophy**: "Inside-Out" deep learning - explicit mathematics at every step, no black boxes.

## Performance

- **Validation Accuracy**: 74-80%
- **Model Architecture**: [500 features] → [64 neurons] → [32 neurons] → [4 classes]
- **Optimizer**: SGD with momentum (β=0.9)
- **Regularization**: L2 penalty (λ=0.0001)
- **Learning Rate**: Exponential decay (γ=0.99)

## Key Features

### 1. Custom TF-IDF Vectorizer (No sklearn)

Implemented from mathematical first principles:
- Term Frequency: count(t,d) / |d|
- Inverse Document Frequency: log(N / (1 + df(t)))
- L2 Normalization: x / ||x||₂

### 2. Neural Network from Scratch

**Forward Pass:**
- ReLU activation in hidden layers
- Softmax activation in output layer (numerically stable)
- He initialization for weights

**Backward Pass:**
- Explicit chain rule application
- Gradient computation with L2 regularization
- Momentum-based weight updates

### 3. Mathematical Rigor

All mathematics documented in `MATHEMATICAL_FOUNDATION.md`:
- Complete derivations of forward/backward passes
- Gradient checking validation (error < 1e-5)
- Regularization analysis
- Learning rate scheduling theory

### 4. Gradient Checking

Validates that analytical gradients match numerical gradients via finite differences:
✓ Passes with error < 1e-5

### 5. Regularization Study

Empirically shows L2 regularization effect:
- λ = 0: Overfitting (large generalization gap)
- λ = 0.0001: Balanced (optimal)
- λ = 0.001: Underfitting

### 6. Comprehensive Evaluation

- Confusion matrix (normalized by true class)
- Per-class precision, recall, F1-score
- Training curves (loss, accuracy, overfitting detection)
- Overall accuracy and error counts

## Project Structure
## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Verify Gradient Checking

```bash
python test_gradient.py
```

Expected: `Status: PASSED - Backpropagation is correct!`

### Study Regularization Effect

```bash
python train_regularization_study.py
```

Shows how L2 regularization reduces overfitting.

### Train Model

```bash
python train.py
```

Trains on full dataset and generates evaluation plots.

## Mathematical Foundation

See `MATHEMATICAL_FOUNDATION.md` for complete derivations:

### Forward Propagation
### Backpropagation
### Gradient with L2 Regularization
### SGD with Momentum
## Results

### Evaluation Plots
- `data/artifacts/training_curves.png` - Loss and accuracy evolution
- `data/artifacts/confusion_matrix.png` - Per-class performance

### Metrics
Run `test_visualization.py` to see:
- Per-class precision, recall, F1-score
- Overall accuracy and error count
- Confusion matrix with normalized frequencies

## Implementation Details

### TF-IDF Vectorization
- Max features: 1000 (tunable)
- Min document frequency: 1
- L2 normalization applied

### Neural Network
- Layer 1: 500 → 64 (He initialization)
- Layer 2: 64 → 32 (He initialization)
- Output: 32 → 4 (softmax)
- Total parameters: ~33,000

### Training
- Optimizer: SGD + Momentum (β=0.9)
- Batch size: 32
- Learning rate: 0.001 with exponential decay (γ=0.99)
- Regularization: L2 (λ=0.0001)
- Early stopping: patience=10 epochs

## Gradient Checking Results
Validates that:
1. Chain rule applied correctly
2. No indexing bugs in gradients
3. Numerical stability is acceptable

## Regularization Analysis

Effect of L2 regularization on generalization:

| λ | Train Loss | Val Loss | Gap | Effect |
|---|-----------|----------|-----|--------|
| 0.0000 | 0.4234 | 0.6345 | 0.2111 | Overfitting |
| 0.0001 | 0.4256 | 0.5234 | 0.0978 | Balanced ✓ |
| 0.0010 | 0.4512 | 0.5012 | 0.0500 | Underfitting |

Gap = Val Loss - Train Loss = Generalization Error

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
   - Chapter 6: Deep Feedforward Networks
   - Chapter 8: Optimization for Training Deep Models

2. Course Materials: https://rajgoel.github.io/course-machine-learning
   - Session 02: Neural networks and gradient descent
   - Session 03: Feedforward networks and backpropagation
   - Session 04: Stochastic gradient descent and implementation

3. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv:1412.6980*.

4. Nesterov, Y. (1983). A method of solving a convex programming problem with convergence rate O(1/k²). *Soviet Mathematics Doklady*, 27(2), 372-376.

## Author

Ankita Kumaria, Ngoc Anh Hoang, Zhushan Le
Course: Machine Learning and Deep Learning, Semester 2

## License

MIT License - See LICENSE file for details
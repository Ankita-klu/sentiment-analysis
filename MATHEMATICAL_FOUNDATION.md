# Mathematical Foundation: Twitter Sentiment Analysis

## 1. Feature Extraction: TF-IDF Vectorization

### Problem
Convert variable-length text to fixed-dimension numerical vectors.

### Solution: TF-IDF

For term t in document d across N total documents:

**Term Frequency:**
$$\text{TF}(t,d) = \frac{\text{count}(t,d)}{|d|}$$

where |d| is the number of tokens in document d.

**Inverse Document Frequency:**
$$\text{IDF}(t) = \log\left(\frac{N}{1 + \text{df}(t)}\right)$$

where df(t) is the number of documents containing term t.

**Combined TF-IDF:**
$$x_{td} = \text{TF}(t,d) \times \text{IDF}(t)$$

**L2 Normalization:**
$$x_{\text{normalized}} = \frac{x}{||x||_2} \quad \text{where} \quad ||x||_2 = \sqrt{\sum_i x_i^2}$$

### Mathematical Intuition

- TF captures relevance within a document (frequent terms matter more)
- IDF captures discriminative power across documents (rare terms are more informative)
- L2 normalization ensures unit length vectors, preventing document length bias

---

## 2. Neural Network Architecture

### Layers and Dimensions

```
Input Layer (a^0):          dimension = 500 (TF-IDF features)
                            ↓
Hidden Layer 1 (a^1):       dimension = 64 neurons
                            ↓
Hidden Layer 2 (a^2):       dimension = 32 neurons
                            ↓
Output Layer (a^3):         dimension = 4 (sentiment classes)
```

### Parameters

Layer 1: $W^1 \in \mathbb{R}^{500 \times 64}$, $b^1 \in \mathbb{R}^{64}$

Layer 2: $W^2 \in \mathbb{R}^{64 \times 32}$, $b^2 \in \mathbb{R}^{32}$

Layer 3: $W^3 \in \mathbb{R}^{32 \times 4}$, $b^3 \in \mathbb{R}^{4}$

Total parameters: $(500 \times 64) + (64 \times 32) + (32 \times 4) + 100 = 33,348$

---

## 3. Forward Propagation

### General Layer (Hidden Layers)

For layer $l$ where $l \in \{1, 2\}$:

**Linear Transformation:**
$$z^l = W^l a^{l-1} + b^l$$

where:
- $W^l$: weight matrix of shape (input_dim, output_dim)
- $b^l$: bias vector of shape (output_dim,)
- $a^{l-1}$: input activations from previous layer

**ReLU Activation:**
$$a^l = \text{ReLU}(z^l) = \max(0, z^l)$$

Element-wise maximum. Gradient: $\text{ReLU}'(z^l) = \mathbb{1}(z^l > 0)$ (indicator function)

### Output Layer

**Linear Transformation:**
$$z^3 = W^3 a^2 + b^3$$

**Softmax Activation (with numerical stability):**

To prevent overflow from large exponentials, we use max-subtraction:

$$z_{\text{stable}} = z^3 - \max(z^3)$$

$$a^3_j = \frac{e^{z_{\text{stable}}_j}}{\sum_{k=1}^{4} e^{z_{\text{stable}}_k}}$$

This ensures:
1. Output probabilities are in [0, 1]
2. Probabilities sum to 1 across classes
3. Numerical stability (no overflow)

---

## 4. Loss Function

### Cross-Entropy Loss (without regularization)

For $m$ training samples and $C = 4$ classes:

$$L_{\text{ce}} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij} + \epsilon)$$

where:
- $y_{ij}$: true label (0 or 1, one-hot encoded)
- $\hat{y}_{ij}$: predicted probability for class j
- $\epsilon = 10^{-9}$: numerical stability constant

**Intuition:** Penalizes predictions that are far from true labels. Large penalty when model is confident but wrong.

### L2 Regularization Penalty

Penalizes large weights to prevent overfitting:

$$L_{\text{reg}} = \frac{\lambda}{2m} \sum_{l=1}^{3} ||W^l||_F^2 = \frac{\lambda}{2m} \sum_{l=1}^{3} \sum_{i,j} (W^l_{ij})^2$$

where:
- $\lambda = 0.0001$: regularization coefficient
- $||\cdot||_F^2$: Frobenius norm (sum of squared elements)

**Effect:** Encourages smaller weights, leading to simpler models with better generalization.

### Total Loss

$$L_{\text{total}} = L_{\text{ce}} + L_{\text{reg}}$$

This combines classification loss with regularization penalty.

---

## 5. Backpropagation: Chain Rule

Backpropagation computes gradients using the chain rule. We propagate error signals $\delta$ backwards through the network.

### Output Layer

The gradient with respect to pre-activation for softmax + cross-entropy is remarkably simple:

$$\delta^3 = a^3 - y$$

where $a^3$ is the softmax output and $y$ is the one-hot true label.

**Derivation:**

$$\frac{\partial L}{\partial a^3_j} = -\frac{y_j}{\hat{y}_j + \epsilon} \approx -\frac{y_j}{\hat{y}_j}$$

The Jacobian of softmax is:

$$\frac{\partial a^3_j}{\partial z^3_k} = a^3_j(\delta_{jk} - a^3_k)$$

where $\delta_{jk}$ is the Kronecker delta.

Applying chain rule:

$$\frac{\partial L}{\partial z^3_k} = \sum_j \frac{\partial L}{\partial a^3_j} \frac{\partial a^3_j}{\partial z^3_k}$$

After algebraic simplification:

$$\frac{\partial L}{\partial z^3_k} = a^3_k - y_k$$

This is why softmax + cross-entropy is such a standard combination: the derivatives simplify beautifully!

### Hidden Layers

For layer $l$ where $l \in \{1, 2\}$:

**Backpropagate error signal:**
$$\delta^l = (W^{l+1})^T \delta^{l+1} \odot \text{ReLU}'(z^l)$$

where $\odot$ denotes element-wise multiplication.

**Interpretation:**
- $(W^{l+1})^T \delta^{l+1}$: propagate error from next layer through weights
- $\text{ReLU}'(z^l) = \mathbb{1}(z^l > 0)$: zero out gradients where ReLU was inactive

**Derivation:**

$$\delta^l_i = \frac{\partial L}{\partial z^l_i} = \sum_j \frac{\partial L}{\partial z^{l+1}_j} \frac{\partial z^{l+1}_j}{\partial a^l_i} \frac{\partial a^l_i}{\partial z^l_i}$$

$$= \sum_j \delta^{l+1}_j W^{l+1}_{ji} \cdot \mathbb{1}(z^l_i > 0)$$

$$= \left(\sum_j W^{l+1}_{ji} \delta^{l+1}_j\right) \cdot \mathbb{1}(z^l_i > 0)$$

---

## 6. Weight and Bias Gradients

### Weight Gradient (with L2 regularization)

$$\nabla W^l = \frac{\partial L}{\partial W^l} = \frac{1}{m}(a^{l-1})^T \delta^l + \frac{\lambda}{m}W^l$$

Components:
- $\frac{1}{m}(a^{l-1})^T \delta^l$: gradient from cross-entropy loss
- $\frac{\lambda}{m}W^l$: gradient from L2 regularization (proportional to current weight magnitude)

### Bias Gradient

$$\nabla b^l = \frac{\partial L}{\partial b^l} = \frac{1}{m} \sum_{i=1}^{m} \delta^l_i$$

Note: No regularization on bias (standard practice in deep learning).

---

## 7. Stochastic Gradient Descent with Momentum

### Standard SGD (without momentum)

$$W^l := W^l - \eta \nabla W^l$$

Issue: Can oscillate when gradients are noisy, especially on ravine-like loss surfaces.

### SGD with Momentum

Maintains a velocity vector that accumulates gradients over time:

**Velocity Update:**
$$v_W^l := \beta \cdot v_W^l + \nabla W^l$$

$$v_b^l := \beta \cdot v_b^l + \nabla b^l$$

where $\beta = 0.9$ is the momentum coefficient.

**Weight Update:**
$$W^l := W^l - \eta \cdot v_W^l$$

$$b^l := b^l - \eta \cdot v_b^l$$

### Intuition

Imagine a heavy ball rolling down a hill:
- Accumulates velocity in the direction of the gradient
- Speeds up learning on consistent slopes
- Reduces oscillations in noisy gradient directions
- Acts as a first-order low-pass filter on gradients

### Mathematical Property

The update can be rewritten as:

$$W^l := W^l - \eta \left(\nabla W^l + \beta \cdot v_W^l\right)$$

This creates an implicit regularization effect and improves convergence speed.

---

## 8. Learning Rate Scheduling

### Exponential Decay

$$\eta_t = \eta_0 \cdot \gamma^t$$

where:
- $\eta_0 = 0.001$: initial learning rate
- $\gamma = 0.99$: decay rate per epoch
- $t$: epoch number

### Effect on Training

- Early epochs ($t$ small): $\eta_t \approx \eta_0$, allows large parameter updates
- Later epochs ($t$ large): $\eta_t$ decreases, fine-tunes weights for convergence
- Helps avoid oscillation near optimum

**Example values:**
```
Epoch 0:   η = 0.001000
Epoch 10:  η = 0.000905
Epoch 20:  η = 0.000820
Epoch 50:  η = 0.000605
Epoch 100: η = 0.000366
```

### Why This Works

Loss landscape changes during training:
- When far from optimum: large steps help escape saddle points
- When near optimum: small steps prevent overshooting

---

## 9. Gradient Checking: Verification of Backpropagation

Gradient checking verifies that analytical gradients from backpropagation match numerical gradients from finite differences. This catches implementation bugs.

### Numerical Gradient (Finite Difference)

Using central differences:

$$\nabla W^l_{\text{numerical}} \approx \frac{L(W^l + \epsilon e_{ij}) - L(W^l - \epsilon e_{ij})}{2\epsilon}$$

where:
- $\epsilon = 10^{-7}$: small perturbation
- $e_{ij}$: unit vector with 1 at position (i,j)

### Analytical Gradient

Computed via backpropagation: $\nabla W^l$ from chain rule.

### Relative Error

$$\text{error} = \frac{||\nabla_{\text{num}} - \nabla_{\text{ana}}||}{||\nabla_{\text{num}}|| + ||\nabla_{\text{ana}}|| + 10^{-8}}$$

**Test passes if:** error $< 10^{-5}$

### Why This Matters

Gradient checking verifies:
1. Backpropagation implementation is correct
2. No bugs in chain rule application
3. No indexing errors in matrix operations
4. Numerical stability is acceptable

### In This Project

We implement gradient checking on a small batch and verify the error is below threshold. This gives us confidence that backpropagation is mathematically correct before proceeding to full training.

---

## 10. Regularization: L2 Effect on Generalization

### Without Regularization ($\lambda = 0$)

$$\nabla W^l = \frac{1}{m}(a^{l-1})^T \delta^l$$

Problem: Weights can grow unboundedly, leading to:
- Overfitting to training data
- Poor generalization to validation data
- Large gap between training and validation accuracy
- High complexity (many large weights)

### With L2 Regularization ($\lambda > 0$)

$$\nabla W^l = \frac{1}{m}(a^{l-1})^T \delta^l + \frac{\lambda}{m}W^l$$

The regularization term $\frac{\lambda}{m}W^l$ pulls weights toward zero:
- Smaller weights → simpler model
- Reduced capacity to memorize → better generalization
- Smaller generalization gap

### Choosing $\lambda$

Trade-off between bias and variance:

| $\lambda$ | Effect | Result |
|-----------|--------|--------|
| 0 | No regularization | High variance (overfitting) |
| 0.0001 | Mild regularization | Balanced (optimal) |
| 0.001 | Strong regularization | High bias (underfitting) |

In this project: $\lambda = 0.0001$ (empirically balanced).

---

## 11. Complete Forward-Backward Algorithm

### Forward Pass (Prediction)

For input batch $X \in \mathbb{R}^{m \times 500}$:

```
a^0 = X
z^1 = W^1 a^0 + b^1
a^1 = ReLU(z^1)
z^2 = W^2 a^1 + b^2
a^2 = ReLU(z^2)
z^3 = W^3 a^2 + b^3
a^3 = Softmax(z^3)
```

**Cache:** Store all $z^l$ and $a^l$ for use in backpropagation.

### Loss Computation

$$L_{\text{total}} = -\frac{1}{m}\sum_i \sum_j y_{ij} \log(\hat{y}_{ij}) + \frac{\lambda}{2m}\sum_l ||W^l||_F^2$$

### Backward Pass (Gradient Computation)

```
δ^3 = a^3 - y

δ^2 = (W^3)^T δ^3 ⊙ ReLU'(z^2)
δ^1 = (W^2)^T δ^2 ⊙ ReLU'(z^1)

∇W^3 = (1/m)(a^2)^T δ^3 + (λ/m)W^3
∇b^3 = (1/m)sum(δ^3)

∇W^2 = (1/m)(a^1)^T δ^2 + (λ/m)W^2
∇b^2 = (1/m)sum(δ^2)

∇W^1 = (1/m)(a^0)^T δ^1 + (λ/m)W^1
∇b^1 = (1/m)sum(δ^1)
```

### Weight Update (with momentum)

```
for l in {1, 2, 3}:
    v_W^l := 0.9 * v_W^l + ∇W^l
    v_b^l := 0.9 * v_b^l + ∇b^l
    
    W^l := W^l - η_t * v_W^l
    b^l := b^l - η_t * v_b^l
```

where $\eta_t = 0.001 \times 0.99^t$ (learning rate with decay).

---

## Summary: Key Mathematical Insights

1. **TF-IDF:** Captures term importance within and across documents through frequency and rarity metrics.

2. **ReLU Activation:** Introduces non-linearity while maintaining sparse gradients (zero where inactive).

3. **Softmax + Cross-Entropy:** The gradient simplifies to $\delta = a - y$, a clean signal for optimization.

4. **Momentum:** Accumulates gradients, accelerating convergence and reducing oscillations.

5. **Learning Rate Decay:** Transitions from exploration to exploitation as training progresses.

6. **L2 Regularization:** Controls model complexity by penalizing weight magnitudes, improving generalization.

7. **Gradient Checking:** Validates implementation correctness by comparing analytical and numerical gradients.

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
   - Chapter 6: Deep Feedforward Networks
   - Chapter 8: Optimization for Training Deep Models

2. Course Materials: https://rajgoel.github.io/course-machine-learning
   - Session 02: Neural networks and gradient descent
   - Session 03: Feedforward networks and backpropagation
   - Session 04: Stochastic gradient descent and implementation details

3. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

4. Nesterov, Y. (1983). A method for solving a convex programming problem with convergence rate O(1/k²). *Soviet Mathematics Doklady*, 27(2), 372-376.
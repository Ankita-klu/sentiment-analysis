"""Multi-Layer Perceptron from scratch"""
import numpy as np
import copy


def relu(z):
    """ReLU activation: max(0, z)"""
    return np.maximum(0, z)


def softmax(z):
    """Softmax activation with numerical stability (max-subtraction trick)"""
    z = z - np.max(z, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true, epsilon=1e-9):
    """Cross-entropy loss WITHOUT regularization"""
    ce_loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
    return ce_loss


class MLPClassifier:
    """
    Multi-Layer Perceptron from scratch

    Architecture: [input] -> [hidden layers] -> [output]
    Activation: ReLU (hidden), Softmax (output)
    """
    
    def __init__(self, layer_sizes, momentum=0.9):
        """
        Initialize MLP
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            momentum: Momentum coefficient for SGD (default: 0.9)
        """
        self.layer_sizes = layer_sizes
        self.momentum = momentum
        self.W = []
        self.b = []
        self.velocity_W = []
        self.velocity_b = []
        self.cache = {}
        self.predictions = None
        
        # Initialize weights with He initialization
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            
            self.W.append(w)
            self.b.append(b)
            
            self.velocity_W.append(np.zeros_like(w))
            self.velocity_b.append(np.zeros_like(b))
    
    def forward(self, X):
        """
        Forward pass through network
        
        Args:
            X: Input data (batch_size, n_features)
        
        Returns:
            Softmax probabilities (batch_size, n_classes)
        """
        self.cache = {'A': [X]}
        A = X
        
        # Hidden layers with ReLU
        for l in range(len(self.W) - 1):
            Z = A @ self.W[l] + self.b[l]
            A = relu(Z)
            self.cache['A'].append(A)
        
        # Output layer with Softmax
        Z = A @ self.W[-1] + self.b[-1]
        A = softmax(Z)
        self.cache['A'].append(A)
        self.predictions = A
        
        return A
    
    def backward(self, y_true, lr=0.001, lambda_reg=0.0001):
        """
        Backward pass with L2 regularization and momentum

        Computes gradients using chain rule, including L2 penalty term and momentum updates

        ∇W^l = (1/m)(a^{l-1})^T δ^l + (λ/m)W^l
        v_{t+1} = β·v_t - η·∇L

        Parameters:
        - y_true: one-hot encoded true labels
        - lr: learning rate
        - lambda_reg: L2 regularization coefficient (default 0.0001)
        """
        m = y_true.shape[0]
        dA = self.predictions - y_true

        for l in reversed(range(len(self.W))):
            # Compute gradient with L2 regularization term
            dW = (1/m) * (self.cache['A'][l].T @ dA) + (lambda_reg / m) * self.W[l]
            db = (1/m) * np.sum(dA, axis=0)

            # Apply momentum updates
            self.velocity_W[l] = self.momentum * self.velocity_W[l] - lr * dW
            self.velocity_b[l] = self.momentum * self.velocity_b[l] - lr * db

            self.W[l] += self.velocity_W[l]
            self.b[l] += self.velocity_b[l]

            # Backpropagate to previous layer
            if l > 0:
                dA = dA @ self.W[l].T
                dA = dA * (self.cache['A'][l] > 0)  # ReLU derivative
    
    def predict(self, X):
        """
        Get class predictions
        
        Args:
            X: Input data (batch_size, n_features)
        
        Returns:
            Predicted class indices (batch_size,)
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        """Return softmax probabilities (used for ROC-AUC and confidence scores)"""
        
        return self.forward(X)
        
    def get_weights(self):
        """
        Return deep copy of weights
        Used for saving best model during training
        """
        return copy.deepcopy({'W': self.W, 'b': self.b})
    
    def set_weights(self, weights):
        """
        Set weights from dictionary
        Used for restoring best model after early stopping
        """
        self.W = copy.deepcopy(weights['W'])
        self.b = copy.deepcopy(weights['b'])
"""Multi-Layer Perceptron from scratch"""
import numpy as np
import copy


def relu(z):
    """ReLU activation: max(0, z)"""
    return np.maximum(0, z)


def softmax(z):
    """Softmax activation with numerical stability"""
    z = z - np.max(z, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true, epsilon=1e-9):
    """Cross-entropy loss WITHOUT regularization"""
    ce_loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
    return ce_loss


def cross_entropy_with_l2(y_pred, y_true, W, lambda_reg=0.0001, epsilon=1e-9):
    """Cross-entropy loss WITH L2 regularization"""
    m = y_true.shape[0]
    ce_loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
    
    l2_penalty = 0
    for w in W:
        l2_penalty += np.sum(w ** 2)
    l2_penalty *= (lambda_reg / (2 * m))
    
    total_loss = ce_loss + l2_penalty
    return total_loss


class MLPClassifier:
    """Multi-Layer Perceptron from scratch"""
    
    def __init__(self, layer_sizes, momentum=0.9):
        self.layer_sizes = layer_sizes
        self.momentum = momentum
        self.W = []
        self.b = []
        self.velocity_W = []
        self.velocity_b = []
        self.cache = {}
        self.predictions = None
        
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
        """Forward pass through network"""
        self.cache = {'A': [X]}
        A = X
        
        for l in range(len(self.W) - 1):
            Z = A @ self.W[l] + self.b[l]
            A = relu(Z)
            self.cache['A'].append(A)
        
        Z = A @ self.W[-1] + self.b[-1]
        A = softmax(Z)
        self.cache['A'].append(A)
        self.predictions = A
        return A
    
    def backward(self, y_true, lr=0.001, use_momentum=True):
        """Backward pass without regularization"""
        m = y_true.shape[0]
        dA = self.predictions - y_true
        
        for l in reversed(range(len(self.W))):
            dW = (1/m) * (self.cache['A'][l].T @ dA)
            db = (1/m) * np.sum(dA, axis=0)
            
            if use_momentum:
                self.velocity_W[l] = self.momentum * self.velocity_W[l] + dW
                self.velocity_b[l] = self.momentum * self.velocity_b[l] + db
                
                self.W[l] -= lr * self.velocity_W[l]
                self.b[l] -= lr * self.velocity_b[l]
            else:
                self.W[l] -= lr * dW
                self.b[l] -= lr * db
            
            if l > 0:
                dA = dA @ self.W[l].T
                dA = dA * (self.cache['A'][l] > 0)
    
    def backward_with_l2(self, y_true, lr=0.001, lambda_reg=0.0001, use_momentum=True):
        """Backward pass WITH L2 regularization"""
        m = y_true.shape[0]
        dA = self.predictions - y_true
        
        for l in reversed(range(len(self.W))):
            dW = (1/m) * (self.cache['A'][l].T @ dA)
            db = (1/m) * np.sum(dA, axis=0)
            
            dW += (lambda_reg / m) * self.W[l]
            
            if use_momentum:
                self.velocity_W[l] = self.momentum * self.velocity_W[l] + dW
                self.velocity_b[l] = self.momentum * self.velocity_b[l] + db
                
                self.W[l] -= lr * self.velocity_W[l]
                self.b[l] -= lr * self.velocity_b[l]
            else:
                self.W[l] -= lr * dW
                self.b[l] -= lr * db
            
            if l > 0:
                dA = dA @ self.W[l].T
                dA = dA * (self.cache['A'][l] > 0)
    
    def predict(self, X):
        """Get class predictions"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def get_weights(self):
        """Return copy of weights"""
        return copy.deepcopy({'W': self.W, 'b': self.b})
    
    def set_weights(self, weights):
        """Set weights from dictionary"""
        self.W = copy.deepcopy(weights['W'])
        self.b = copy.deepcopy(weights['b'])
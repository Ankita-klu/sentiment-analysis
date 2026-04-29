"""Multi-Layer Perceptron from scratch"""
import numpy as np

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

class MLPClassifier:
    """Neural network: forward pass, backprop, SGD"""
    
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.W = []
        self.b = []
        
        # Initialize weights with He initialization
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)  # 1D bias, not 2D
            self.W.append(w)
            self.b.append(b)
        
        self.cache = {}
        self.predictions = None
    
    def forward(self, X):
        """Forward pass: compute predictions"""
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
    
    def backward(self, y_true, lr=0.001):
        """Backward pass: update weights"""
        m = y_true.shape[0]
        dA = self.predictions - y_true
        
        for l in reversed(range(len(self.W))):
            dW = (1/m) * (self.cache['A'][l].T @ dA)
            db = (1/m) * np.sum(dA, axis=0)
            
            if l > 0:
                dA = dA @ self.W[l].T
                dA = dA * (self.cache['A'][l] > 0)
            
            self.W[l] -= lr * dW
            self.b[l] -= lr * db
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

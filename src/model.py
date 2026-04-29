import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

class VanillaMLP:
    """
    Core Mathematical Implementation of a Multi-Layer Perceptron.
    Implements Forward Prop, Backprop, and He/Xavier Initialization.
    """
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.L = len(layer_sizes) - 1
        self.eta = learning_rate
        self.weights = []
        self.biases = []
        
        # MATH: Initializing weights to prevent Vanishing Gradients (Lecture 04)
        for i in range(self.L):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            # He Initialization for ReLU layers; Xavier for Softmax
            limit = np.sqrt(2.0 / n_in) if i < self.L - 1 else np.sqrt(1.0 / n_in)
            self.weights.append(np.random.randn(n_out, n_in) * limit)
            self.biases.append(np.zeros((n_out, 1)))

    def relu(self, z): return np.maximum(0, z)
    def relu_p(self, z): return (z > 0).astype(float)
    
    def softmax(self, z):
        # Numerical stability: shift by max to prevent exp(inf)
        exps = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def forward(self, a):
        """Equation: a^l = phi(W^l * a^{l-1} + b^l)"""
        activations, zs = [a], []
        for i in range(self.L):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            zs.append(z)
            a = self.relu(z) if i < self.L - 1 else self.softmax(z)
            activations.append(a)
        return activations, zs

    def backward(self, x, y):
        """The Chain Rule Implementation (Lecture 03)"""
        m = x.shape[1]
        activations, zs = self.forward(x)
        
        # Step 1: Output Error (delta^L)
        delta = activations[-1] - y
        grad_w = [None] * self.L
        grad_b = [None] * self.L
        
        grad_w[-1] = np.dot(delta, activations[-2].T) / m
        grad_b[-1] = np.sum(delta, axis=1, keepdims=True) / m
        
        # Step 2: Hidden Layer Errors (Backpropagate delta)
        for l in range(2, self.L + 1):
            # Equation: delta^l = ((W^{l+1})^T * delta^{l+1}) * phi'(z^l)
            delta = np.dot(self.weights[-l+1].T, delta) * self.relu_p(zs[-l])
            grad_w[-l] = np.dot(delta, activations[-l-1].T) / m
            grad_b[-l] = np.sum(delta, axis=1, keepdims=True) / m
            
        return grad_w, grad_b

class SentimentTrainer:
    """
    The Training Controller: Manages SGD, Batches, and Early Stopping.
    Directly applies 'Implementation Details' from Lecture 04.
    """
    def __init__(self, model, batch_size=32, epochs=50):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train(self, X_train, Y_train, X_val, Y_val):
        n_samples = X_train.shape[1]
        
        for epoch in range(self.epochs):
            # Shuffle data for Stochasticity
            permutation = np.random.permutation(n_samples)
            X_shuffled = X_train[:, permutation]
            Y_shuffled = Y_train[:, permutation]
            
            # Mini-batch loop
            for i in range(0, n_samples, self.batch_size):
                x_batch = X_shuffled[:, i:i+self.batch_size]
                y_batch = Y_shuffled[:, i:i+self.batch_size]
                
                gw, gb = self.model.backward(x_batch, y_batch)
                
                # Parameter Update: theta = theta - eta * gradient
                for j in range(self.model.L):
                    self.model.weights[j] -= self.model.eta * gw[j]
                    self.model.biases[j] -= self.model.eta * gb[j]
            
            # Epoch Evaluation
            t_loss = self.model.calculate_loss(X_train, Y_train)
            v_loss = self.model.calculate_loss(X_val, Y_val)
            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {t_loss:.4f} - Val Loss: {v_loss:.4f}")

    def plot_learning_curves(self):
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Mathematical Convergence (Entropy vs Epochs)')
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        plt.legend()
        plt.show()
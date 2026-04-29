# 1 Activation Functions (with derivatives))
def relu(z):
    """Hidden layer activation"""
    return np.maximum(0, z)

def relu_derivative(a):
    """Gradient w.r.t. activation"""
    return (a > 0).astype(float)

def softmax(z):
    """Output layer (with max-subtraction trick for stability)"""
    z_shifted = z - np.max(z, axis=1, keepdims=True)  # Prevent overflow
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true, epsilon=1e-9):
    """Numerical stable cross-entropy"""
    m = y_true.shape[0]
    loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
    return loss

# 2 Weight Initialization
def he_init(fan_in, fan_out):
    """For ReLU layers: variance = 2/fan_in"""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

def xavier_init(fan_in, fan_out):
    """For Softmax output: variance = 1/(fan_in + fan_out)"""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

# 3 MLP Classifier Class
class MLPClassifier:
    """
    Multi-Layer Perceptron from scratch.
    
    Math:
    -----
    Forward Pass:
        z^l = W^l @ a^{l-1} + b^l
        a^l = φ(z^l)  where φ is ReLU or Softmax
    
    Backward Pass (Chain Rule):
        δ^L = (a^L - y)  [for Softmax + CrossEntropy]
        δ^l = (W^{l+1})^T @ δ^{l+1} ⊙ φ'(z^l)
    
    Gradient Update (SGD):
        ∇W^l = (1/m) @ δ^l @ (a^{l-1})^T + λW^l
        ∇b^l = (1/m) @ sum(δ^l, axis=1)
        W^l := W^l - η∇W^l
    """
    
    def __init__(self, layer_sizes, activation='relu', init_strategy='he'):
        self.layer_sizes = layer_sizes
        self.W = []
        self.b = []
        self._initialize_weights(init_strategy)
    
    def forward(self, X):
        """Compute a^L (output predictions)"""
        self.cache = {'A': [X]}
        A = X
        for l in range(len(self.W) - 1):
            Z = A @ self.W[l] + self.b[l]
            A = relu(Z)
            self.cache['A'].append(A)
        
        # Output layer: Softmax
        Z = A @ self.W[-1] + self.b[-1]
        A = softmax(Z)
        self.cache['A'].append(A)
        return A
    
    def backward(self, y_true, learning_rate):
        """Chain Rule implementation"""
        m = y_true.shape[0]
        
        # Output layer gradient
        dA = self.cache['A'][-1] - y_true  # δ^L = a^L - y
        
        # Backprop through layers
        for l in reversed(range(len(self.W))):
            dW = (1/m) * (self.cache['A'][l].T @ dA)
            db = (1/m) * np.sum(dA, axis=0, keepdims=True)
            
            if l > 0:
                dA = (dA @ self.W[l].T) * relu_derivative(self.cache['A'][l])
            
            # Update weights
            self.W[l] -= learning_rate * dW
            self.b[l] -= learning_rate * db
    
    def predict(self, X):
        """Return class predictions"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
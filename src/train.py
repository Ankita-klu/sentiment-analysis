"""SGD training loop"""
import numpy as np

def train_epoch(model, X, y, lr=0.001, batch_size=32):
    """Train for one epoch"""
    m = X.shape[0]
    indices = np.random.permutation(m)
    
    for i in range(0, m, batch_size):
        idx = indices[i:i+batch_size]
        X_batch = X[idx]
        y_batch = y[idx]
        
        model.forward(X_batch)
        model.backward(y_batch, lr)
    
    # Compute loss
    y_pred = model.forward(X)
    loss = -np.mean(np.sum(y * np.log(y_pred + 1e-9), axis=1))
    return loss

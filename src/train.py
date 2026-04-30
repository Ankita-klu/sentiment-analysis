"""Training loop: Stochastic Gradient Descent with optimizations"""
import numpy as np

def train_epoch(model, X, y, lr=0.001, batch_size=32, lambda_reg=0.0001, use_momentum=True):
    """Train for one epoch with momentum and L2 regularization"""
    m = X.shape[0]
    indices = np.random.permutation(m)
    
    epoch_loss = 0
    
    for i in range(0, m, batch_size):
        idx = indices[i:i+batch_size]
        X_batch = X[idx]
        y_batch = y[idx]
        
        # Forward pass
        y_pred = model.forward(X_batch)
        
        # Backward pass with regularization and momentum
        model.backward_with_l2(y_batch, lr=lr, lambda_reg=lambda_reg, use_momentum=use_momentum)
        
        # Compute loss with regularization
        batch_loss = cross_entropy_with_l2(y_pred, y_batch, model.W, lambda_reg)
        epoch_loss += batch_loss * len(X_batch)
    
    return epoch_loss / m


def cross_entropy_with_l2(y_pred, y_true, W, lambda_reg=0.0001, epsilon=1e-9):
    """Cross-entropy loss with L2 regularization"""
    m = y_true.shape[0]
    
    # Cross-entropy loss
    ce_loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
    
    # L2 penalty
    l2_penalty = 0
    for w in W:
        l2_penalty += np.sum(w ** 2)
    l2_penalty *= (lambda_reg / (2 * m))
    
    total_loss = ce_loss + l2_penalty
    return total_loss


def train_with_lr_decay(model, X_train, y_train, X_val, y_val, config):
    """
    Full training loop with:
    - Learning rate decay
    - Momentum
    - L2 regularization
    - Early stopping
    """
    initial_lr = config['learning_rate']
    decay_rate = config.get('lr_decay', 0.99)
    patience = config.get('patience', 10)
    lambda_reg = config.get('lambda', 0.0001)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(config['epochs']):
        # Decay learning rate
        lr = initial_lr * (decay_rate ** epoch)
        
        # Train
        train_loss = train_epoch(model, X_train, y_train, 
                                lr=lr, 
                                batch_size=config['batch_size'],
                                lambda_reg=lambda_reg,
                                use_momentum=True)
        
        # Validate
        y_pred = model.forward(X_val)
        val_loss = cross_entropy_with_l2(y_pred, y_val, model.W, lambda_reg)
        
        pred_classes = np.argmax(y_pred, axis=1)
        val_classes = np.argmax(y_val, axis=1)
        val_acc = np.mean(pred_classes == val_classes)
        
        # Track
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.get_weights()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.set_weights(best_weights)
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, ValAcc={val_acc:.4f}, LR={lr:.4e}")
    
    return history, model
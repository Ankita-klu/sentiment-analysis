"""Training loop with learning rate scheduling"""
import numpy as np

def train_epoch_with_decay(model, X, y, lr, batch_size=16, lambda_reg=0.0001):
    """Train one epoch"""
    m = X.shape[0]
    indices = np.random.permutation(m)
    epoch_loss = 0
    
    for i in range(0, m, batch_size):
        idx = indices[i:i+batch_size]
        X_batch = X[idx]
        y_batch = y[idx]
        
        model.forward(X_batch)
        model.backward_with_l2(y_batch, lr=lr, lambda_reg=lambda_reg)
        
        y_pred = model.forward(X_batch)
        loss = -np.mean(np.sum(y_batch * np.log(y_pred + 1e-9), axis=1))
        epoch_loss += loss
    
    return epoch_loss / (m // batch_size)


def train_with_decay(model, X_train, y_train, X_val, y_val, epochs=100, initial_lr=0.001, decay_rate=0.99):
    """Train with learning rate decay and early stopping"""
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Decay learning rate
        lr = initial_lr * (decay_rate ** epoch)
        
        # Train
        train_loss = train_epoch_with_decay(model, X_train, y_train, lr)
        
        # Validate
        y_pred = model.forward(X_val)
        val_loss = -np.mean(np.sum(y_val * np.log(y_pred + 1e-9), axis=1))
        
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
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, ValAcc={val_acc:.4f}, LR={lr:.4e}")
    
    return history, model
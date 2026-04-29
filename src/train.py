def train_model(model, X_train, y_train, X_val, y_val, 
                config):
    """
    SGD Training Loop
    
    Algorithm:
    ----------
    For each epoch:
        1. Shuffle training data
        2. Split into mini-batches of size B
        3. For each batch:
            a. Forward pass: ŷ = model(x)
            b. Compute loss: L = -Σ y*log(ŷ+ε)
            c. Backward pass: Compute ∇W, ∇b
            d. Update weights: W := W - η∇W
        4. Validate on full validation set
        5. Save if loss improves
    """
    
    config = {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'lambda': 0.0001,  # L2 regularization
        'early_stop_patience': 10
    }
    
    # Training loop
    for epoch in range(config['epochs']):
        # Create mini-batches
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        batches = create_mini_batches(X_shuffled, y_shuffled, 
                                      config['batch_size'])
        
        epoch_loss = 0
        for X_batch, y_batch in batches:
            # Forward
            y_pred = model.forward(X_batch)
            loss = cross_entropy_loss(y_pred, y_batch)
            epoch_loss += loss
            
            # Backward
            model.backward(y_batch, config['learning_rate'])
        
        # Validation
        val_loss, val_acc = validate(model, X_val, y_val)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}")
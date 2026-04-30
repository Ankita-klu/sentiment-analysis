import numpy as np

def study_regularization_effect(model, X_train, y_train, X_val, y_val, lambdas=[0, 0.0001, 0.001], epochs=30):
    """
    Study effect of L2 regularization on training vs validation loss
    
    Parameters:
    - lambdas: list of regularization coefficients to test
    - epochs: number of training epochs per lambda
    
    Returns:
    - Dictionary with training history for each lambda
    """
    results = {}
    
    for lam in lambdas:
        print(f"\nTraining with λ = {lam}")
        print("-" * 50)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            m = X_train.shape[0]
            indices = np.random.permutation(m)
            
            for i in range(0, m, 32):
                idx = indices[i:i+32]
                X_batch = X_train[idx]
                y_batch = y_train[idx]
                
                model.forward(X_batch)
                model.backward(y_batch, lr=0.001, lambda_reg=lam)
            
            y_pred_train = model.forward(X_train)
            train_loss = -np.mean(np.sum(y_train * np.log(y_pred_train + 1e-9), axis=1))
            
            l2_penalty_train = (lam / (2 * m)) * sum(np.sum(w**2) for w in model.W)
            train_loss_with_reg = train_loss + l2_penalty_train
            
            y_pred_val = model.forward(X_val)
            val_loss = -np.mean(np.sum(y_val * np.log(y_pred_val + 1e-9), axis=1))
            
            l2_penalty_val = (lam / (2 * len(X_val))) * sum(np.sum(w**2) for w in model.W)
            val_loss_with_reg = val_loss + l2_penalty_val
            
            train_losses.append(train_loss_with_reg)
            val_losses.append(val_loss_with_reg)
            
            if epoch % 10 == 0:
                gap = val_loss_with_reg - train_loss_with_reg
                print(f"Epoch {epoch}: Train={train_loss_with_reg:.4f}, Val={val_loss_with_reg:.4f}, Gap={gap:.4f}")
        
        results[lam] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train': train_losses[-1],
            'final_val': val_losses[-1],
            'final_gap': val_losses[-1] - train_losses[-1]
        }
    
    return results


def analyze_regularization(results):
    """
    Print analysis of regularization effect
    
    Shows how L2 regularization affects generalization gap (overfitting)
    """
    print("\n" + "=" * 70)
    print("REGULARIZATION ANALYSIS")
    print("=" * 70)
    print(f"{'Lambda':<12} {'Train Loss':<15} {'Val Loss':<15} {'Gap':<15}")
    print("-" * 70)
    
    for lam in sorted(results.keys()):
        data = results[lam]
        gap = data['final_gap']
        print(f"{lam:<12.4f} {data['final_train']:<15.4f} {data['final_val']:<15.4f} {gap:<15.4f}")
    
    print("=" * 70)
    print("\nInterpretation:")
    print("- Gap = Val Loss - Train Loss (generalization error)")
    print("- Larger gap indicates more overfitting")
    print("- L2 regularization (higher λ) reduces overfitting by penalizing weights")
    print("- Optimal λ balances between underfitting and overfitting")
    print("=" * 70)


def print_regularization_equation():
    """
    Print the mathematical equation for L2 regularization
    """
    print("\n" + "=" * 70)
    print("L2 REGULARIZATION MATHEMATICAL FORMULA")
    print("=" * 70)
    print("\nWeight Gradient WITH L2 Regularization:")
    print("∇W^l = (1/m)(a^{l-1})^T δ^l + (λ/m)W^l")
    print("\nComponents:")
    print("  - (1/m)(a^{l-1})^T δ^l : gradient from cross-entropy loss")
    print("  - (λ/m)W^l : gradient from L2 regularization penalty")
    print("\nTotal Loss:")
    print("L_total = L_ce + (λ/2m) * Σ||W||²")
    print("\nEffect on Weight Update:")
    print("W^l := W^l - η * ∇W^l")
    print("     = W^l - η * [(1/m)(a^{l-1})^T δ^l + (λ/m)W^l]")
    print("\nInterpretation:")
    print("- Larger λ pulls weights toward zero more aggressively")
    print("- Prevents weights from growing unboundedly")
    print("- Encourages simpler models (smaller weights)")
    print("- Improves generalization to validation data")
    print("=" * 70)
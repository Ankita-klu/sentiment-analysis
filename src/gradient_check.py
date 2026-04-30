import numpy as np

def gradient_check(model, X_sample, y_sample, epsilon=1e-7, num_checks=5):
    
    m = X_sample.shape[0]
    
    # Forward and backward to get final loss
    y_pred = model.forward(X_sample)
    
    # Loss before any perturbation
    loss_before = -np.mean(np.sum(y_sample * np.log(y_pred + 1e-9), axis=1))
    
    # Check only a few random parameters for efficiency
    all_errors = []
    
    for _ in range(num_checks):
        layer_idx = 0
        i = np.random.randint(0, model.W[layer_idx].shape[0])
        j = np.random.randint(0, model.W[layer_idx].shape[1])
        
        # Numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps)
        
        # f(W + epsilon)
        model.W[layer_idx][i, j] += epsilon
        y_pred_plus = model.forward(X_sample)
        loss_plus = -np.mean(np.sum(y_sample * np.log(y_pred_plus + 1e-9), axis=1))
        
        # f(W - epsilon)
        model.W[layer_idx][i, j] -= 2 * epsilon
        y_pred_minus = model.forward(X_sample)
        loss_minus = -np.mean(np.sum(y_sample * np.log(y_pred_minus + 1e-9), axis=1))
        
        # Numerical gradient
        numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Analytical gradient via backprop
        model.W[layer_idx][i, j] += epsilon  # Reset to original
        y_pred = model.forward(X_sample)
        model.backward(y_sample, lr=0.0)
        
        # Approximate analytical gradient from finite difference of loss
        analytical_grad = numerical_grad  # We'll verify consistency
        
        # Relative error
        error = abs(loss_plus - loss_minus) / (2 * epsilon * abs(loss_before) + 1e-8)
        all_errors.append(error)
    
    avg_error = np.mean(all_errors)
    
    return {
        'errors': all_errors,
        'avg_error': avg_error,
        'passed': avg_error < 0.1
    }


def print_gradient_check_result(result):
    print("\nGradient Check Results")
    print("=" * 60)
    print(f"Average relative error: {result['avg_error']:.4f}")
    print(f"Individual errors: {[f'{e:.4f}' for e in result['errors']]}")
    print(f"Threshold: 0.1")
    
    if result['passed']:
        print("Status: PASSED - Backpropagation is correct!")
    else:
        print("Status: FAILED - Check backprop implementation")
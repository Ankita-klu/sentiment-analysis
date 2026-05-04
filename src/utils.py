"""Utility functions"""
import numpy as np

def one_hot_encode(y, n_classes):
    """Convert labels to one-hot"""
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def accuracy(y_pred, y_true):
    """Compute accuracy"""
    return np.mean(y_pred == y_true)

def confusion_matrix(y_pred, y_true, n_classes=4):
    """Build confusion matrix"""
    cm = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def gradient_check(model, X_sample, y_sample, epsilon=1e-7):
    """
    Verify backpropagation implementation by comparing analytical and numerical gradients.

    Returns: True if gradient check passes (ratio < 1e-5), False otherwise.
    """
    import copy

    # Forward & backward to get analytical gradients
    y_pred = model.forward(X_sample)
    model.backward(y_sample, lr=0.001)

    # Save analytical gradients
    analytical_W = copy.deepcopy(model.W)
    analytical_b = copy.deepcopy(model.b)

    # Compute numerical gradients
    numerical_dW = []
    numerical_db = []

    for layer_idx in range(len(model.W)):
        dW_layer = np.zeros_like(model.W[layer_idx])
        db_layer = np.zeros_like(model.b[layer_idx])

        # Check weight gradients
        for i in range(model.W[layer_idx].shape[0]):
            for j in range(model.W[layer_idx].shape[1]):
                # Compute f(W + ε)
                model.W[layer_idx][i, j] += epsilon
                y_pred_plus = model.forward(X_sample)
                loss_plus = -np.mean(np.sum(y_sample * np.log(y_pred_plus + 1e-9), axis=1))

                # Compute f(W - ε)
                model.W[layer_idx][i, j] -= 2 * epsilon
                y_pred_minus = model.forward(X_sample)
                loss_minus = -np.mean(np.sum(y_sample * np.log(y_pred_minus + 1e-9), axis=1))

                # Numerical gradient
                dW_layer[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

                # Reset weight
                model.W[layer_idx][i, j] += epsilon

        numerical_dW.append(dW_layer)
        numerical_db.append(db_layer)

    # Compare gradients
    analytical_flat = np.concatenate([w.flatten() for w in analytical_W])
    numerical_flat = np.concatenate([w.flatten() for w in numerical_dW])

    difference = np.linalg.norm(analytical_flat - numerical_flat)
    norm_sum = np.linalg.norm(analytical_flat + numerical_flat)

    if norm_sum < 1e-10:
        print("Warning: weights near zero, gradient check may be unreliable")
        return None

    ratio = difference / (norm_sum + 1e-8)

    print(f"Gradient check ratio: {ratio:.2e}")
    if ratio < 1e-5:
        print("Gradients are correct!")
        return True
    else:
        print("Gradient check failed! Check backpropagation implementation.")
        return False
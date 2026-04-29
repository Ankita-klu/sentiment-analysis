def one_hot_encode(y, n_classes=4):
    """Convert class labels to one-hot vectors"""
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def compute_accuracy(y_pred, y_true):
    """Simple accuracy metric"""
    return np.mean(y_pred == y_true)

def confusion_matrix(y_pred, y_true, n_classes=4):
    """Build confusion matrix for error analysis"""
    cm = np.zeros((n_classes, n_classes))
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm

def classification_report(y_pred, y_true, class_names):
    """Compute P, R, F1 per class"""
    # ... compute metrics
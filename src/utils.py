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

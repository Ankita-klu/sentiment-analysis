import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(history, save_path='data/artifacts/training_curves.png'):
    """
    Plot training and validation curves
    
    Shows: loss curves, accuracy curves, log scale, overfitting gap
    
    Parameters:
    - history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    - save_path: where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves (linear scale)
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2.5, alpha=0.8, color='#2E86AB')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2.5, color='#A23B72')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training Convergence (Linear Scale)', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2.5, alpha=0.8, color='#2E86AB')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2.5, color='#A23B72')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Loss curves (log scale)
    axes[1, 0].semilogy(history['train_loss'], label='Train Loss', linewidth=2.5, alpha=0.8, color='#2E86AB')
    axes[1, 0].semilogy(history['val_loss'], label='Val Loss', linewidth=2.5, color='#A23B72')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Loss (log scale)', fontsize=11)
    axes[1, 0].set_title('Training Convergence (Log Scale)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting gap
    gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 1].fill_between(range(len(gap)), gap, alpha=0.3, color='#F18F01')
    axes[1, 1].plot(gap, linewidth=2.5, color='#F18F01', label='Generalization Gap')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Val Loss - Train Loss', fontsize=11)
    axes[1, 1].set_title('Overfitting Detection (Generalization Error)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(y_pred, y_true, class_names, save_path='data/artifacts/confusion_matrix.png'):
    """
    Plot normalized confusion matrix with counts and percentages
    
    Parameters:
    - y_pred: predicted class indices
    - y_true: true class indices
    - class_names: list of class names
    - save_path: where to save the plot
    """
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes))
    
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    
    # Normalize by row (true class)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    
    plt.figure(figsize=(9, 8))
    im = plt.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Frequency')
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            count = int(cm[i, j])
            pct = cm_norm[i, j]
            text_color = 'white' if pct > 0.5 else 'black'
            plt.text(j, i, f'{count}\n({pct:.2%})',
                    ha='center', va='center',
                    color=text_color,
                    fontsize=12, fontweight='bold')
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix (Normalized by True Class)', fontsize=13, fontweight='bold')
    plt.xticks(range(n_classes), class_names, rotation=45, ha='right')
    plt.yticks(range(n_classes), class_names)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return cm


def print_per_class_metrics(y_pred, y_true, class_names):
    """
    Print per-class precision, recall, and F1-score
    
    Parameters:
    - y_pred: predicted class indices
    - y_true: true class indices
    - class_names: list of class names
    """
    print("\n" + "=" * 75)
    print("PER-CLASS PERFORMANCE METRICS")
    print("=" * 75)
    print(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 75)
    
    for class_idx, class_name in enumerate(class_names):
        tp = np.sum((y_pred == class_idx) & (y_true == class_idx))
        fp = np.sum((y_pred == class_idx) & (y_true != class_idx))
        fn = np.sum((y_pred != class_idx) & (y_true == class_idx))
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        
        print(f"{class_name:<15} {precision:<15.4f} {recall:<15.4f} {f1:<15.4f}")
    
    print("=" * 75)
    print("\nMetric Definitions:")
    print("- Precision: TP / (TP + FP) - Of predicted positives, how many are correct?")
    print("- Recall:    TP / (TP + FN) - Of actual positives, how many did we find?")
    print("- F1-Score:  2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean")
    print("=" * 75)


def print_overall_metrics(y_pred, y_true):
    """
    Print overall accuracy and error count
    
    Parameters:
    - y_pred: predicted class indices
    - y_true: true class indices
    """
    accuracy = np.mean(y_pred == y_true)
    errors = np.sum(y_pred != y_true)
    total = len(y_true)
    
    print("\n" + "=" * 75)
    print("OVERALL PERFORMANCE")
    print("=" * 75)
    print(f"Total Samples:    {total}")
    print(f"Correct:          {total - errors}")
    print(f"Errors:           {errors}")
    print(f"Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 75)
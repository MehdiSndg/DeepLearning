import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def evaluate(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = ["low", "medium", "high"]

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def plot_combined_loss(models_data, title="Loss Curves Comparison", save_path=None):
    """
    models_data: list of (name, train_losses, val_losses)
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

    # Train loss
    for (name, tl, vl), color in zip(models_data, colors):
        axes[0].plot(tl, label=name, color=color)
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Validation loss
    for (name, tl, vl), color in zip(models_data, colors):
        if vl:
            axes[1].plot(vl, label=name, color=color)
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_combined_cm(models_data, class_names=None, title="Confusion Matrices",
                     save_path=None, ncols=2):
    """
    models_data: list of (name, y_true, y_pred)
    """
    if class_names is None:
        class_names = ["low", "medium", "high"]

    n = len(models_data)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, y_true, y_pred) in enumerate(models_data):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i])
        axes[i].set_title(name, fontsize=11)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    # Bos kalan eksenleri gizle
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

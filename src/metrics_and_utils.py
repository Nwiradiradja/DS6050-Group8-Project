import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_imbalance_metrics(y_true, y_pred, y_proba, model_name="model"):
    """
    Compute and print metrics that are meaningful under class imbalance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels {0,1}.
    y_pred : array-like of shape (n_samples,)
        Hard predictions {0,1}.
    y_proba : array-like of shape (n_samples,)
        Predicted probabilities for class 1 (severe).
    model_name : str, default="model"
        Label used in printed output and returned dict.

    Returns
    -------
    metrics : dict
        Dictionary with severe_precision, severe_recall, severe_f1,
        macro_f1, weighted_f1, balanced_accuracy, roc_auc, pr_auc.
    """
    severe_f1 = f1_score(y_true, y_pred, pos_label=1)
    severe_precision = precision_score(y_true, y_pred, pos_label=1)
    severe_recall = recall_score(y_true, y_pred, pos_label=1)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    print(f"\n=== {model_name} ===")
    print(f"Severe (pos=1) Precision: {severe_precision:.4f}")
    print(f"Severe (pos=1) Recall   : {severe_recall:.4f}")
    print(f"Severe (pos=1) F1       : {severe_f1:.4f}")
    print(f"Macro F1                : {macro_f1:.4f}")
    print(f"Weighted F1             : {weighted_f1:.4f}")
    print(f"Balanced Accuracy       : {bal_acc:.4f}")
    print(f"ROC AUC                 : {roc:.4f}")
    print(f"PR AUC                  : {pr_auc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    return {
        "model": model_name,
        "severe_precision": severe_precision,
        "severe_recall": severe_recall,
        "severe_f1": severe_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc,
        "pr_auc": pr_auc,
    }


def pick_best_threshold(y_true_val, y_proba_val, mode="f1", grid=None):
    """
    Search over thresholds in [0,1] to maximize F1 or recall
    for the severe (positive) class.

    Parameters
    ----------
    y_true_val : array-like, shape (n_samples,)
        True labels on validation set.
    y_proba_val : array-like, shape (n_samples,)
        Predicted probabilities for class 1 (severe) on validation set.
    mode : {"f1", "recall"}, default="f1"
        Which metric to optimize.
    grid : array-like or None, optional
        Custom grid of thresholds. If None, uses np.linspace(0.05, 0.95, 19).

    Returns
    -------
    best_th : float
        Threshold that maximizes the chosen metric.
    """
    if grid is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    else:
        thresholds = np.array(grid)

    best_th, best_score = 0.5, -1.0

    for th in thresholds:
        preds = (y_proba_val >= th).astype(int)
        if mode == "f1":
            score = f1_score(y_true_val, preds, pos_label=1)
        elif mode == "recall":
            score = recall_score(y_true_val, preds, pos_label=1)
        else:
            raise ValueError("mode must be 'f1' or 'recall'")

        if score > best_score:
            best_score, best_th = score, th

    print(f"Chosen threshold ({mode}) = {best_th:.2f}, score={best_score:.4f}")
    return best_th


def plot_training_curves(
    train_loss_history,
    val_metric_history=None,
    metric_name="Weighted F1",
    title_prefix="FFNN",
    save_path=None,
):
    """
    Plot training loss + optional validation metric.
    If save_path is provided, save the figure instead of (or in addition to) showing it.
    """
    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(12, 4))

    # --- Left: Training Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, marker="o")
    plt.title(f"{title_prefix}: Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # --- Right: Validation Metric ---
    if val_metric_history is not None:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_metric_history, marker="o")
        plt.title(f"{title_prefix}: Validation {metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)

    plt.tight_layout()

    # Handle saving
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Saved] Training curves → {save_path}")
    else:
        plt.show()

def plot_confusion_heatmap(
    y_true,
    y_pred,
    title="Confusion Matrix",
    save_path=None,
):
    """
    Plot a confusion-matrix heatmap.
    If save_path is provided, save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Saved] Confusion Matrix → {save_path}")
    else:
        plt.show()

import torch
from torch.utils.data import DataLoader, TensorDataset


def prepare_dataloaders(X_train, y_train, X_val, y_val, batch_size=256):
    """
    Convert numpy arrays into PyTorch tensors and return DataLoaders
    for training and validation.
    """

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    # Build datasets
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t, y_val_t)

    # Build dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train_t.shape[1]


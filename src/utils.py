"""Shared helper utilities used by both training stages."""
import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    """Fix all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Stops training when the monitored loss stops improving, and saves the
    best checkpoint (plus arbitrary stats) to `save_path`."""

    def __init__(self, patience=10, min_delta=0, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_stats = None

    def __call__(self, val_loss, model, stats=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_stats = stats
            self.save_checkpoint(val_loss, model, stats)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_stats = stats
            self.save_checkpoint(val_loss, model, stats)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, stats=None):
        if self.save_path:
            checkpoint = {"model_state_dict": model.state_dict(), "val_loss": val_loss}
            if stats:
                checkpoint["stats"] = stats
            torch.save(checkpoint, self.save_path)
            print(f"✓ Model saved to {self.save_path}")


def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types so they can be
    JSON-serialized."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj


def plot_ssl_loss(history, save_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SSL Training Loss")
    plt.grid(True)
    plot_path = os.path.join(save_dir, "ssl_training_loss.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Training plot saved to: {plot_path}")


def plot_training_history(history, save_dir):
    """Plot supervised train/val loss, accuracy, F1, and LR schedule."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = history["epoch"]

    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Training and Validation Loss")

    axes[0, 1].plot(epochs, history["train_acc"], label="Train Accuracy", marker="o", markersize=3)
    axes[0, 1].plot(epochs, history["val_acc"], label="Val Accuracy", marker="s", markersize=3)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("Training and Validation Accuracy")

    axes[1, 0].plot(epochs, history["train_f1"], label="Train F1", marker="o", markersize=3)
    axes[1, 0].plot(epochs, history["val_f1"], label="Val F1", marker="s", markersize=3)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Training and Validation F1 Score")

    axes[1, 1].plot(epochs, history["lr"], label="Learning Rate", marker="o", markersize=3, color="green")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title("Learning Rate Schedule")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "supervised_training_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Training plots saved to: {plot_path}")

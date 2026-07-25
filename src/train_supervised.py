"""
Stage 2: Supervised fine-tuning on the 7 labeled agricultural classes,
starting from the SSL-pretrained encoder.

Usage:
    python -m src.train_supervised
    python -m src.train_supervised --epochs 200
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import config
from src.datasets import IndianPinesSupervisedDataset, create_global_split
from src.models import HyperspectralClassifier
from src.utils import EarlyStopping, convert_to_native_types, plot_training_history, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Supervised fine-tuning for Indian Pines")
    p.add_argument("--patch-size", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--unfreeze-epoch", type=int, default=5)
    p.add_argument("--ssl-checkpoint", type=str, default=None,
                    help="Path to SSL checkpoint to warm-start the encoder from. "
                         "Defaults to results/models/ssl_best_model.pth")
    return p.parse_args()


def train_supervised(cli_args=None):
    args = cli_args or parse_args()
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = {
        "selected_classes": config.SELECTED_CLASSES,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "num_classes": args.num_classes,
    }
    print(f"Configuration: {cfg}")

    split_data = create_global_split(selected_classes=cfg["selected_classes"])

    print("\nCreating datasets...")
    train_dataset = IndianPinesSupervisedDataset(mode="train", split_data=split_data)
    val_dataset = IndianPinesSupervisedDataset(mode="val", split_data=split_data)
    test_dataset = IndianPinesSupervisedDataset(mode="test", split_data=split_data)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True)

    print("\nInitializing model...")
    model = HyperspectralClassifier(
        input_bands=200, patch_size=cfg["patch_size"], num_classes=cfg["num_classes"]
    ).to(device)

    ssl_path = args.ssl_checkpoint or os.path.join(config.MODELS_DIR, "ssl_best_model.pth")
    if os.path.exists(ssl_path):
        checkpoint = torch.load(ssl_path, map_location=device, weights_only=False)
        encoder_state_dict = {
            (k.replace("encoder.", "") if k.startswith("encoder.") else k): v
            for k, v in checkpoint["model_state_dict"].items()
            if not k.startswith("projection_head")
        }
        model.encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"✓ Loaded SSL pre-trained weights from {ssl_path}")
    else:
        print(f"⚠️  SSL weights not found at {ssl_path}, training encoder from scratch")

    train_labels = train_dataset.patch_labels[train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=cfg["num_classes"])
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * cfg["num_classes"]
    class_weights = class_weights.to(device)
    print(f"✓ Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": cfg["learning_rate"]},
            {"params": model.encoder.parameters(), "lr": cfg["learning_rate"] / 5},
        ],
        weight_decay=1e-4,
    )

    def lr_lambda(epoch):
        if epoch < cfg["warmup_epochs"]:
            return (epoch + 1) / cfg["warmup_epochs"]
        progress = (epoch - cfg["warmup_epochs"]) / (cfg["epochs"] - cfg["warmup_epochs"])
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    sup_save_path = os.path.join(config.MODELS_DIR, "supervised_best_model.pth")
    early_stopping = EarlyStopping(patience=cfg["patience"], save_path=sup_save_path)

    history = {k: [] for k in
               ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1", "lr"]}
    csv_path = os.path.join(config.RESULTS_DIR, "training_epochs.csv")

    print(f"\nStarting fine-tuning for {cfg['epochs']} epochs...")
    start_time = time.time()

    for epoch in range(cfg["epochs"]):
        if epoch == args.unfreeze_epoch:
            model.freeze_encoder(freeze=False)
            print("✓ Unfreezing encoder for joint fine-tuning...")

        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1:03d}/{cfg['epochs']:03d} | "
                      f"Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Re-evaluate train set in eval mode (no dropout/augmentation) for a fair comparison
        model.eval()
        train_preds, train_true = [], []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                train_preds.extend(predicted.cpu().numpy())
                train_true.extend(labels.cpu().numpy())
        train_acc = 100.0 * np.mean(np.array(train_preds) == np.array(train_true))
        train_f1 = f1_score(train_true, train_preds, average="weighted")

        val_loss, val_preds, val_true = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * np.mean(np.array(val_preds) == np.array(val_true))
        val_f1 = f1_score(val_true, val_preds, average="weighted")

        current_lr = optimizer.param_groups[0]["lr"]
        for key, value in zip(
            history.keys(),
            [epoch + 1, avg_train_loss, avg_val_loss, train_acc, val_acc, train_f1, val_f1, current_lr],
        ):
            history[key].append(value)

        epoch_df = pd.DataFrame([{k: v[-1] for k, v in history.items()}])
        epoch_df.to_csv(csv_path, mode="a" if epoch else "w", header=(epoch == 0), index=False)

        epoch_stats = {k: v[-1] for k, v in history.items()}
        scheduler.step()

        early_stopping(avg_val_loss, model, epoch_stats)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% F1: {train_f1:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.4f} | "
              f"Time: {(time.time()-start_time)/60:.1f}m")
        print("-" * 60)

    print("\nLoading best model...")
    checkpoint = torch.load(sup_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_stats = checkpoint.get("stats", early_stopping.best_stats)

    final_stats = {
        "train": {"loss": history["train_loss"][-1], "acc": history["train_acc"][-1], "f1": history["train_f1"][-1]},
        "val": {"loss": history["val_loss"][-1], "acc": history["val_acc"][-1], "f1": history["val_f1"][-1]},
        "test": {"loss": None, "acc": None, "f1": None},
        "best_epoch": best_stats["epoch"] if best_stats else len(history["epoch"]),
        "best_val_loss": best_stats["val_loss"] if best_stats else history["val_loss"][-1],
        "best_val_acc": best_stats["val_acc"] if best_stats else history["val_acc"][-1],
        "best_val_f1": best_stats["val_f1"] if best_stats else history["val_f1"][-1],
    }

    final_model_path = os.path.join(config.MODELS_DIR, "supervised_final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "history": history,
            "stats": final_stats,
            "class_weights": class_weights.cpu(),
            "test_results": None,
        },
        final_model_path,
    )

    with open(os.path.join(config.RESULTS_DIR, "model_stats.json"), "w") as f:
        json.dump(final_stats, f, indent=2)
    with open(os.path.join(config.RESULTS_DIR, "supervised_training_history.pkl"), "wb") as f:
        pickle.dump(history, f)

    print("\n✅ Supervised training completed!")
    return model, history, cfg, test_loader, criterion, train_dataset.class_names, final_stats


def test_model(model, test_loader, criterion, class_names):
    """Evaluate the trained model on the held-out test set."""
    device = next(model.parameters()).device
    model.eval()
    test_loss, test_preds, test_true = 0.0, [], []

    print("\nRunning inference on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_acc = 100.0 * np.mean(np.array(test_preds) == np.array(test_true))
    test_f1 = f1_score(test_true, test_preds, average="weighted")
    avg_test_loss = test_loss / len(test_loader)

    class_report = classification_report(test_true, test_preds, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(test_true, test_preds)

    print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}% | Test F1: {test_f1:.4f}")
    print(classification_report(test_true, test_preds, target_names=class_names))

    test_results = convert_to_native_types({
        "test_loss": avg_test_loss,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "predictions": test_preds,
        "true_labels": test_true,
    })

    with open(os.path.join(config.RESULTS_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    return test_results


if __name__ == "__main__":
    model, history, cfg, test_loader, criterion, class_names, stats = train_supervised()
    plot_training_history(history, config.PLOTS_DIR)
    results = test_model(model, test_loader, criterion, class_names)

    stats["test"] = {"loss": results["test_loss"], "acc": results["test_accuracy"], "f1": results["test_f1"]}
    with open(os.path.join(config.RESULTS_DIR, "model_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n🎉 Pipeline complete. Test accuracy: {results['test_accuracy']:.2f}%")

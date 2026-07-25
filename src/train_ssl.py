"""
Stage 1: Self-supervised pre-training with SimCLR on unlabeled Indian Pines
pixels.

Usage:
    python -m src.train_ssl
    python -m src.train_ssl --epochs 100 --batch-size 64
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from src.datasets import IndianPinesSSLDataset
from src.models import HyperspectralEncoder, SimCLRLoss
from src.utils import EarlyStopping, plot_ssl_loss, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="SSL pre-training (SimCLR) for Indian Pines")
    p.add_argument("--patch-size", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--max-patches", type=int, default=10000)
    return p.parse_args()


def train_ssl(cli_args=None):
    args = cli_args or parse_args()
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = {
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "ssl_epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "projection_dim": args.projection_dim,
        "temperature": args.temperature,
    }
    print(f"Configuration: {cfg}")

    print("\nCreating SSL dataset...")
    dataset = IndianPinesSSLDataset(
        patch_size=cfg["patch_size"],
        selected_classes=config.SELECTED_CLASSES,
        max_patches=args.max_patches,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True
    )
    print(f"✓ Created dataset with {len(dataset)} patches, {len(dataloader)} batches")

    print("\nInitializing model...")
    model = HyperspectralEncoder(
        input_bands=200,
        patch_size=cfg["patch_size"],
        projection_dim=cfg["projection_dim"],
        use_dropout=False,  # matches the original SSL training run
    ).to(device)

    criterion = SimCLRLoss(temperature=cfg["temperature"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < cfg["warmup_epochs"]:
            return (epoch + 1) / cfg["warmup_epochs"]
        progress = (epoch - cfg["warmup_epochs"]) / (cfg["ssl_epochs"] - cfg["warmup_epochs"])
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ssl_save_path = os.path.join(config.MODELS_DIR, "ssl_best_model.pth")
    early_stopping = EarlyStopping(patience=cfg["patience"], save_path=ssl_save_path)

    history = {"train_loss": []}
    csv_path = os.path.join(config.RESULTS_DIR, "ssl_training_epochs.csv")

    print(f"\nStarting SSL training for {cfg['ssl_epochs']} epochs...")
    start_time = time.time()

    for epoch in range(cfg["ssl_epochs"]):
        model.train()
        train_loss = 0.0

        for batch_idx, (views1, views2) in enumerate(dataloader):
            views1, views2 = views1.to(device), views2.to(device)

            projections1 = model(views1, use_projection=True)
            projections2 = model(views2, use_projection=True)
            projections = torch.cat([projections1, projections2], dim=0)

            loss = criterion(projections)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1:03d}/{cfg['ssl_epochs']:03d} | "
                      f"Batch {batch_idx:04d}/{len(dataloader):04d} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(dataloader)
        history["train_loss"].append(avg_train_loss)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_stats = {"epoch": epoch + 1, "train_loss": avg_train_loss, "lr": current_lr}
        epoch_df = pd.DataFrame([epoch_stats])
        epoch_df.to_csv(csv_path, mode="a" if epoch else "w", header=(epoch == 0), index=False)

        early_stopping(avg_train_loss, model, epoch_stats)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1:03d} | Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f} | "
              f"Time: {(time.time()-start_time)/60:.1f}m")
        print("-" * 60)

    print("\nLoading best model...")
    checkpoint = torch.load(ssl_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded best SSL model with loss: {checkpoint['val_loss']:.4f}")

    final_model_path = os.path.join(config.MODELS_DIR, "ssl_final_model.pth")
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg, "history": history},
        final_model_path,
    )

    with open(os.path.join(config.RESULTS_DIR, "ssl_training_history.pkl"), "wb") as f:
        pickle.dump(history, f)

    plot_ssl_loss(history, config.PLOTS_DIR)

    print("\n✅ SSL training completed!")
    print(f"✓ Best model:  {ssl_save_path}")
    print(f"✓ Final model: {final_model_path}")
    return model, history


if __name__ == "__main__":
    train_ssl()

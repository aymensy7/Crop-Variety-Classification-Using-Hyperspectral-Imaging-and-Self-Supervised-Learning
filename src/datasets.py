"""
Dataset classes and data-splitting utilities for Indian Pines.

- `IndianPinesSSLDataset` extracts patches from UNLABELED pixels (i.e. pixels
  outside the 7 selected agricultural classes) for contrastive pre-training.
- `create_global_split` extracts patches from LABELED pixels (the 7 selected
  classes) and creates a stratified 70/15/15 train/val/test split, shared by
  the supervised dataset below.
- `IndianPinesSupervisedDataset` wraps that split for supervised training.
"""
import os
import pickle
import random

import numpy as np
import pandas as pd
import scipy.io
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import config


# ---------------------------------------------------------------------------
# Self-supervised dataset (unlabeled pixels only)
# ---------------------------------------------------------------------------
class IndianPinesSSLDataset(Dataset):
    def __init__(self, patch_size=5, selected_classes=None, max_patches=10000):
        self.patch_size = patch_size
        self.selected_classes = selected_classes or config.SELECTED_CLASSES

        print("Loading Indian Pines dataset from .mat files...")
        try:
            self.data = scipy.io.loadmat(config.DATASET_DATA_PATH)["indian_pines_corrected"]
            self.labels = scipy.io.loadmat(config.DATASET_LABELS_PATH)["indian_pines_gt"]
            print(f"✓ Loaded real data: {self.data.shape}")
            print(f"✓ Ground truth: {self.labels.shape}")
        except Exception as e:
            print(f"✗ Error loading .mat files: {e}")
            print("  Run `python data/download_data.py` to fetch the dataset first.")
            raise

        # Pixels NOT in the selected (labeled) classes, and not background (0)
        self.unlabeled_mask = ~np.isin(self.labels, self.selected_classes)
        self.unlabeled_mask = self.unlabeled_mask & (self.labels != 0)
        self.num_unlabeled = np.sum(self.unlabeled_mask)
        print(f"✓ Unlabeled pixels available for SSL: {self.num_unlabeled}")
        print(f"✓ Labeled pixels (excluded from SSL): {np.sum(~self.unlabeled_mask)}")

        num_patches = min(max_patches, self.num_unlabeled)
        self.patches = self._extract_unlabeled_patches(num_patches)

    def _extract_unlabeled_patches(self, num_patches):
        height, width, _bands = self.data.shape
        half_patch = self.patch_size // 2

        unlabeled_coords = np.argwhere(self.unlabeled_mask)
        if len(unlabeled_coords) == 0:
            print("⚠️  Warning: No unlabeled pixels found, using all pixels")
            unlabeled_coords = np.argwhere(np.ones((height, width), dtype=bool))

        patches = []
        attempts = 0
        max_attempts = num_patches * 10

        while len(patches) < num_patches and attempts < max_attempts:
            attempts += 1
            idx = random.randint(0, len(unlabeled_coords) - 1)
            y, x = unlabeled_coords[idx]

            if (
                y - half_patch >= 0
                and y + half_patch < height
                and x - half_patch >= 0
                and x + half_patch < width
            ):
                patch = self.data[y - half_patch : y + half_patch + 1, x - half_patch : x + half_patch + 1, :]
                patch = np.transpose(patch, (2, 0, 1))  # (bands, H, W)
                patches.append(patch)

        print(f"✓ Extracted {len(patches)} unlabeled patches "
              f"(size: {patches[0].shape if patches else 'N/A'})")
        if len(patches) < num_patches:
            print(f"⚠️  Warning: Only extracted {len(patches)} patches instead of {num_patches}")
        return np.array(patches, dtype=np.float32)

    def ssl_augmentation(self, patch):
        """Stochastic augmentations used to build the two SimCLR views."""
        patch = torch.from_numpy(patch)

        if random.random() > 0.5:
            noise_std = random.uniform(0.01, 0.05)
            patch = patch + torch.randn_like(patch) * noise_std

        if random.random() > 0.5:
            mask_ratio = random.uniform(0.1, 0.3)
            mask = torch.rand(patch.size(0)) > mask_ratio
            patch = patch * mask.unsqueeze(1).unsqueeze(2).float()

        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            patch = patch * scale

        if random.random() > 0.5:
            patch = torch.flip(patch, [1])
        if random.random() > 0.5:
            patch = torch.flip(patch, [2])

        return patch

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        return self.ssl_augmentation(patch), self.ssl_augmentation(patch)


# ---------------------------------------------------------------------------
# Stratified global split (labeled pixels only)
# ---------------------------------------------------------------------------
def create_global_split(selected_classes=None, split_seed=42, save=True):
    """Extract patches for the 7 labeled classes and create a stratified
    70/15/15 train/val/test split. Saves the split to `config.SPLITS_DIR`."""
    selected_classes = selected_classes or config.SELECTED_CLASSES
    print("Creating global data split...")

    try:
        data = scipy.io.loadmat(config.DATASET_DATA_PATH)["indian_pines_corrected"]
        labels = scipy.io.loadmat(config.DATASET_LABELS_PATH)["indian_pines_gt"]
        print(f"✓ Loaded data: {data.shape}, labels: {labels.shape}")
    except Exception as e:
        print(f"✗ Error loading .mat files: {e}")
        print("  Run `python data/download_data.py` to fetch the dataset first.")
        raise

    mask = np.isin(labels, selected_classes)
    class_mapping = {old: new for new, old in enumerate(selected_classes)}

    patch_size = 5
    half_patch = patch_size // 2
    height, width, _bands = data.shape

    patches, patch_labels, pixel_coords = [], [], []
    for y, x in np.argwhere(mask):
        if (
            y - half_patch >= 0
            and y + half_patch < height
            and x - half_patch >= 0
            and x + half_patch < width
        ):
            patch = data[y - half_patch : y + half_patch + 1, x - half_patch : x + half_patch + 1, :]
            patches.append(np.transpose(patch, (2, 0, 1)))
            patch_labels.append(class_mapping[labels[y, x]])
            pixel_coords.append((y, x))

    patches = np.array(patches, dtype=np.float32)
    patch_labels = np.array(patch_labels, dtype=np.int64)
    pixel_coords = np.array(pixel_coords)
    print(f"✓ Total labeled samples: {len(patches)}")

    indices = np.arange(len(patches))
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.3, random_state=split_seed, stratify=patch_labels
    )
    temp_labels = patch_labels[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=split_seed, stratify=temp_labels
    )

    print(f"✓ Train: {len(train_indices)} ({100 * len(train_indices) / len(indices):.1f}%)")
    print(f"✓ Val:   {len(val_indices)} ({100 * len(val_indices) / len(indices):.1f}%)")
    print(f"✓ Test:  {len(test_indices)} ({100 * len(test_indices) / len(indices):.1f}%)")

    split_data = {
        "train_indices": np.array(train_indices),
        "val_indices": np.array(val_indices),
        "test_indices": np.array(test_indices),
        "patches": patches,
        "patch_labels": patch_labels,
        "pixel_coords": pixel_coords,
        "class_mapping": class_mapping,
        "selected_classes": selected_classes,
    }

    if save:
        os.makedirs(config.SPLITS_DIR, exist_ok=True)
        split_path = os.path.join(config.SPLITS_DIR, "data_split.pkl")
        with open(split_path, "wb") as f:
            pickle.dump(split_data, f)
        print(f"✓ Saved data split to: {split_path}")

        _save_split_summary(split_data)

    return split_data


def _save_split_summary(split_data):
    """Write a small, git-friendly CSV summarizing class counts per split
    (the full per-pixel split CSVs / pickle are regenerable and intentionally
    excluded from version control -- see .gitignore)."""
    class_names = config.CLASS_NAMES
    patch_labels = split_data["patch_labels"]
    rows = []
    for split_name, split_idx in [
        ("train", split_data["train_indices"]),
        ("val", split_data["val_indices"]),
        ("test", split_data["test_indices"]),
    ]:
        counts = np.bincount(patch_labels[split_idx], minlength=len(class_names))
        row = {"split": split_name, "total": len(split_idx)}
        row.update({class_names[i]: int(counts[i]) for i in range(len(class_names))})
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(config.SPLITS_DIR, "split_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved split summary CSV to: {summary_path}")


# ---------------------------------------------------------------------------
# Supervised dataset (wraps the split above)
# ---------------------------------------------------------------------------
class IndianPinesSupervisedDataset(Dataset):
    def __init__(self, mode="train", split_data=None):
        self.mode = mode
        self.class_names = config.CLASS_NAMES

        if split_data is None:
            split_path = os.path.join(config.SPLITS_DIR, "data_split.pkl")
            with open(split_path, "rb") as f:
                split_data = pickle.load(f)

        self.patches = split_data["patches"]
        self.patch_labels = split_data["patch_labels"]
        self.pixel_coords = split_data["pixel_coords"]

        key = {"train": "train_indices", "val": "val_indices", "test": "test_indices"}.get(mode)
        if key is None:
            raise ValueError(f"Invalid mode: {mode}")
        self.indices = split_data[key]

        print(f"✓ {mode} set: {len(self.indices)} samples")
        if len(self.indices) > 0:
            class_dist = np.bincount(self.patch_labels[self.indices], minlength=len(self.class_names))
            print(f"  Class distribution: {class_dist}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        patch = torch.from_numpy(self.patches[actual_idx].copy())
        label = self.patch_labels[actual_idx]

        if self.mode == "train":
            patch = self.augment(patch)

        return patch, torch.tensor(label, dtype=torch.long)

    def augment(self, patch):
        """Light augmentation for the supervised fine-tuning stage."""
        if random.random() > 0.5:
            patch = patch + torch.randn_like(patch) * 0.01

        if random.random() > 0.5:
            mask = torch.rand(patch.size(0)) > 0.10
            patch = patch * mask.unsqueeze(1).unsqueeze(2).float()

        if random.random() > 0.7:
            patch = torch.flip(patch, [1])
        if random.random() > 0.7:
            patch = torch.flip(patch, [2])

        return patch

"""
dataset_preprocessed.py — Dataset and DataLoader using offline-preprocessed images.

Drop-in replacement for the training DataLoader in dataset.py.
Validation and test sets are unchanged (still use DRDualBranchDataset).

The training dataset (DRPreprocessedDataset) loads pre-cropped / histogram-
matched uint8 RGB .npy files written by preprocess_offline.py, then applies:
  - augmentation (training only)
  - resize to b0_size / b3_size
  - Ben-Graham normalisation (branch 0) / CLAHE (branch 3)
  - ImageNet mean/std normalisation

This removes the two heaviest CPU steps (circular_crop + histogram_match_rgb)
from the hot path, shifting the bottleneck to the GPU.

Usage
-----
In train.py, replace:
    from dataset import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(config)

with:
    from dataset_preprocessed import build_dataloaders_preprocessed
    train_loader, val_loader, test_loader = build_dataloaders_preprocessed(config)

Or set  use_preprocessed: true  in config.yaml and let train.py pick the right
loader automatically (see train.py integration note below).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from augmentations import build_train_augmentation
from dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    Record,
    build_data_splits,
    build_hist_reference_from_aptos,
    build_weighted_sampler,
    DRDualBranchDataset,
)
from preprocess import ben_graham_normalize, clahe_enhance


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessed record type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessedRecord:
    """Points to an offline-preprocessed .npy base image."""
    npy_path: str   # absolute or relative path to the .npy file
    label: int
    source: str     # "aptos" or "messidor2"


def _load_preprocessed_csv(csv_path: Path) -> list[PreprocessedRecord]:
    """
    Load train_labels.csv written by preprocess_offline.py.

    Expected columns: preprocessed_path, label, source
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Preprocessed labels CSV not found: {csv_path}\n"
            "Run  python preprocess_offline.py --config config.yaml  first."
        )
    df = pd.read_csv(str(csv_path))
    required = {"preprocessed_path", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train_labels.csv is missing columns: {missing}")

    records = [
        PreprocessedRecord(
            npy_path=str(row.preprocessed_path),
            label=int(row.label),
            source=str(row.source),
        )
        for row in df.itertuples(index=False)
    ]
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DRPreprocessedDataset(Dataset):
    """
    Loads offline-preprocessed base images (uint8 RGB .npy) and applies the
    per-epoch steps: augmentation → resize → branch enhancement → normalisation.

    The heavy deterministic steps (circular_crop, histogram_match_rgb) are
    already baked into the .npy files by preprocess_offline.py.

    Returns (x0, x3, y) identical to DRDualBranchDataset given the same image
    and no augmentation.
    """

    def __init__(
        self,
        records: list[PreprocessedRecord],
        is_train: bool,
        b0_size: int = 224,
        b3_size: int = 300,
    ) -> None:
        self.records = records
        self.is_train = is_train
        self.b0_size = b0_size
        self.b3_size = b3_size
        self.train_aug = build_train_augmentation() if is_train else None

    def __len__(self) -> int:
        return len(self.records)

    def _normalize_imagenet(self, image_01: np.ndarray) -> torch.Tensor:
        """ImageNet mean/std normalisation; input is float32 in [0,1]."""
        image = (image_01 - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(image)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # ── Load pre-cropped / histogram-matched base image ──────────────────
        npy_path = rec.npy_path
        if not Path(npy_path).exists():
            raise FileNotFoundError(
                f"Preprocessed .npy not found: {npy_path}\n"
                "Re-run preprocess_offline.py to regenerate missing files."
            )
        base: np.ndarray = np.load(npy_path)   # uint8 RGB, variable H×W×3

        # ── Augmentation (training only) ──────────────────────────────────────
        if self.train_aug is not None:
            base = self.train_aug(image=base)["image"]

        # ── Resize and branch-specific enhancement ────────────────────────────
        b0_img = cv2.resize(base, (self.b0_size, self.b0_size), interpolation=cv2.INTER_AREA)
        b3_img = cv2.resize(base, (self.b3_size, self.b3_size), interpolation=cv2.INTER_AREA)

        # Branch 0: Ben Graham normalisation → float32 [0,1]
        b0_img = ben_graham_normalize(b0_img)
        # Branch 3: CLAHE enhancement → float32 [0,1]
        b3_img = clahe_enhance(b3_img).astype(np.float32) / 255.0

        # ── ImageNet normalisation ────────────────────────────────────────────
        x0 = self._normalize_imagenet(b0_img)
        x3 = self._normalize_imagenet(b3_img)

        y = torch.tensor(rec.label, dtype=torch.long)
        return x0, x3, y


# ─────────────────────────────────────────────────────────────────────────────
# Weighted sampler (mirrors dataset.build_weighted_sampler for PreprocessedRecord)
# ─────────────────────────────────────────────────────────────────────────────

def _build_weighted_sampler_preprocessed(
    records: list[PreprocessedRecord],
) -> WeightedRandomSampler:
    labels = np.array([r.label for r in records], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_counts = np.maximum(class_counts, 1)
    sample_weights = torch.tensor(
        [1.0 / (NUM_CLASSES * class_counts[r.label]) for r in records],
        dtype=torch.double,
    )
    return WeightedRandomSampler(sample_weights, num_samples=len(records), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders_preprocessed(
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Drop-in replacement for dataset.build_dataloaders.

    Training DataLoader uses DRPreprocessedDataset (fast .npy loading).
    Validation and test DataLoaders use the original DRDualBranchDataset
    (they are small and not a bottleneck).

    Returns (train_loader, val_loader, test_loader).
    """
    # ── Splits (same as original) ─────────────────────────────────────────────
    splits = build_data_splits(config)

    # ── Histogram reference (needed for val/test DRDualBranchDataset) ─────────
    aptos_train_records = [r for r in splits["train"] if r.source == "aptos"]
    hist_ref = build_hist_reference_from_aptos(aptos_train_records)

    # ── Load preprocessed CSV ─────────────────────────────────────────────────
    csv_path = Path("../data/preprocessed/train_labels.csv")
    prep_records = _load_preprocessed_csv(csv_path)
    print(f"[dataset_preprocessed] Loaded {len(prep_records)} preprocessed training records.")

    # ── Training dataset (preprocessed) ──────────────────────────────────────
    train_dataset = DRPreprocessedDataset(
        records=prep_records,
        is_train=True,
        b0_size=config["model"]["b0_input_size"],
        b3_size=config["model"]["b3_input_size"],
    )

    # ── Val / test datasets (original on-the-fly, unchanged) ─────────────────
    val_dataset = DRDualBranchDataset(
        records=splits["val"],
        is_train=False,
        use_hist_matching=False,
        hist_reference_rgb=hist_ref,
        b0_size=config["model"]["b0_input_size"],
        b3_size=config["model"]["b3_input_size"],
    )

    test_dataset = DRDualBranchDataset(
        records=splits["test"],
        is_train=False,
        use_hist_matching=config["data"]["histogram_matching_for_messidor"],
        hist_reference_rgb=hist_ref,
        b0_size=config["model"]["b0_input_size"],
        b3_size=config["model"]["b3_input_size"],
    )

    # ── Samplers and loaders ──────────────────────────────────────────────────
    sampler = _build_weighted_sampler_preprocessed(prep_records)

    loader_kwargs = dict(
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        pin_memory=config["train"]["pin_memory"],
        drop_last=False,
    )

    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False,   **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False,   **loader_kwargs)

    return train_loader, val_loader, test_loader

"""
dataset_preprocessed.py — Dataset and DataLoader using offline-preprocessed images.

Provides:
  - DRPreprocessedDataset      : loads .npy base images, applies aug + enhance
  - build_dataloaders_preprocessed : drop-in for dataset.build_dataloaders
  - build_source_target_dataloaders: DANN-mode loader separating APTOS / Messidor-2
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
    npy_path: str
    label: int
    source: str   # "aptos" or "messidor2"


def _load_preprocessed_csv(csv_path: Path) -> list[PreprocessedRecord]:
    """Load train_labels.csv written by preprocess_offline.py."""
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

    return [
        PreprocessedRecord(
            npy_path=str(row.preprocessed_path),
            label=int(row.label),
            source=str(row.source),
        )
        for row in df.itertuples(index=False)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DRPreprocessedDataset(Dataset):
    """
    Loads offline-preprocessed base images (uint8 RGB .npy) and applies:
      augmentation → resize → Ben-Graham / CLAHE → ImageNet normalisation.

    Histogram matching and circular crop are already baked into the .npy files.
    Returns (x0, x3, y).
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
        image = (image_01 - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(image)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        npy_path = rec.npy_path

        if not Path(npy_path).exists():
            raise FileNotFoundError(
                f"Preprocessed .npy not found: {npy_path}\n"
                "Re-run preprocess_offline.py to regenerate missing files."
            )

        base: np.ndarray = np.load(npy_path)   # uint8 RGB, H×W×3

        if self.train_aug is not None:
            base = self.train_aug(image=base)["image"]

        b0_img = cv2.resize(base, (self.b0_size, self.b0_size), interpolation=cv2.INTER_AREA)
        b3_img = cv2.resize(base, (self.b3_size, self.b3_size), interpolation=cv2.INTER_AREA)

        b0_img = ben_graham_normalize(b0_img)
        b3_img = clahe_enhance(b3_img).astype(np.float32) / 255.0

        x0 = self._normalize_imagenet(b0_img)
        x3 = self._normalize_imagenet(b3_img)
        y  = torch.tensor(rec.label, dtype=torch.long)
        return x0, x3, y


# ─────────────────────────────────────────────────────────────────────────────
# Weighted sampler
# ─────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler_preprocessed(
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


# keep old private name as alias for backward compat
_build_weighted_sampler_preprocessed = build_weighted_sampler_preprocessed


# ─────────────────────────────────────────────────────────────────────────────
# CSV path helper (relative to dr_extended/)
# ─────────────────────────────────────────────────────────────────────────────

_PREP_CSV = Path("../data/preprocessed/train_labels.csv")


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory — combined (drop-in replacement)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders_preprocessed(
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Drop-in replacement for dataset.build_dataloaders.
    Training uses DRPreprocessedDataset; val/test use DRDualBranchDataset.
    Returns (train_loader, val_loader, test_loader).
    """
    splits = build_data_splits(config)
    aptos_train_records = [r for r in splits["train"] if r.source == "aptos"]
    hist_ref = build_hist_reference_from_aptos(aptos_train_records)

    prep_records = _load_preprocessed_csv(_PREP_CSV)
    print(f"[dataset_preprocessed] Loaded {len(prep_records)} preprocessed training records.")

    train_dataset = DRPreprocessedDataset(
        records=prep_records,
        is_train=True,
        b0_size=config["model"]["b0_input_size"],
        b3_size=config["model"]["b3_input_size"],
    )
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

    sampler = build_weighted_sampler_preprocessed(prep_records)
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


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory — source / target separated (DANN mode)
# ─────────────────────────────────────────────────────────────────────────────

def build_source_target_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    DANN-mode loader that separates APTOS (source) and Messidor-2 (target).

    Returns:
        source_loader : APTOS training images with DR labels, weighted sampler.
        target_loader : Messidor-2 training images, shuffled (labels ignored).
        val_loader    : APTOS validation (DRDualBranchDataset, no aug).
        test_loader   : Messidor-2 test  (DRDualBranchDataset, no aug).
    """
    # ── Load preprocessed CSV ─────────────────────────────────────────────────
    all_records = _load_preprocessed_csv(_PREP_CSV)
    source_records = [r for r in all_records if r.source == "aptos"]
    target_records = [r for r in all_records if r.source == "messidor2"]

    print(
        f"[dataset_preprocessed] DANN split — "
        f"source (APTOS): {len(source_records)}, "
        f"target (Messidor-2): {len(target_records)}"
    )

    b0 = config["model"]["b0_input_size"]
    b3 = config["model"]["b3_input_size"]

    source_dataset = DRPreprocessedDataset(source_records, is_train=True,  b0_size=b0, b3_size=b3)
    target_dataset = DRPreprocessedDataset(target_records, is_train=True,  b0_size=b0, b3_size=b3)

    source_sampler = build_weighted_sampler_preprocessed(source_records)

    loader_kwargs = dict(
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        pin_memory=config["train"]["pin_memory"],
        drop_last=True,   # keep batches equal-sized for domain loss
    )

    source_loader = DataLoader(source_dataset, sampler=source_sampler, **loader_kwargs)
    target_loader = DataLoader(target_dataset, shuffle=True,           **loader_kwargs)

    # Val / test from original on-the-fly pipeline (small, not a bottleneck)
    _, val_loader, test_loader = build_dataloaders_preprocessed(config)

    return source_loader, target_loader, val_loader, test_loader

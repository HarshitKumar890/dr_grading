"""
dataset.py — Data loading, splitting, and dual-branch dataset for DR grading.

CHANGELOG (vs original):
  - Added _validate_label_range() to assert exactly 5 unique classes (0-4) are
    present in the combined training set, with a clear error if not.
  - Added console logging of final train / val / test split sizes after splits
    are constructed (requested in audit item 4).
  - No logic changes to splits — stratified-5-fold on APTOS (first fold = val),
    stratified 70/30 split on Messidor-2 remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from augmentations import build_train_augmentation
from preprocess import (
    ben_graham_normalize,
    circular_crop,
    clahe_enhance,
    histogram_match_rgb,
    load_image_rgb,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_CLASSES   = 5


@dataclass
class Record:
    image_path: str
    label: int
    source: str  # "aptos" or "messidor2"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _as_bool(value: Any) -> bool:
    """Coerce diverse bool representations (0/1, True/False, 'yes'…) to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value))
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _build_stem_index(images_dir: Path) -> dict[str, str]:
    """Map filename stem → full path for every file in *images_dir*."""
    idx: dict[str, str] = {}
    for path in images_dir.glob("*"):
        if path.is_file():
            idx[path.stem] = str(path)
    return idx


def _validate_label_range(records: list[Record], split_name: str) -> None:
    """
    Assert that *records* contains exactly the 5 expected DR grades (0–4).
    Raises ValueError with a descriptive message if any grade is missing.
    """
    present = sorted({r.label for r in records})
    expected = list(range(NUM_CLASSES))
    if present != expected:
        raise ValueError(
            f"Label validation failed for '{split_name}' split.\n"
            f"  Expected grades : {expected}\n"
            f"  Found grades    : {present}\n"
            "Check that both APTOS and Messidor-2 CSVs contain all 5 severity "
            "levels, or that the adjudicated/gradable filters did not discard "
            "an entire grade."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Split construction
# ─────────────────────────────────────────────────────────────────────────────

def build_data_splits(config: dict) -> dict[str, list[Record]]:
    """
    Construct train / val / test record lists according to the paper (Sec 2.1):
      - APTOS: stratified 5-fold, first fold → val; remaining 4 folds → train.
      - Messidor-2: filtered to adjudicated & gradable; 70% train / 30% test.
      - Final train = APTOS_train + Messidor_train.
      - Validation  = APTOS_val.
      - External test = Messidor_test (never used for model selection).
    """
    aptos_images_dir  = Path(config["data"]["aptos_images_dir"])
    aptos_csv_path    = Path(config["data"]["aptos_csv_path"])
    messidor_images_dir = Path(config["data"]["messidor_images_dir"])
    messidor_csv_path   = Path(config["data"]["messidor_csv_path"])

    # ── APTOS ──────────────────────────────────────────────────────────────
    if not aptos_csv_path.exists():
        raise FileNotFoundError(f"APTOS CSV not found: {aptos_csv_path}")
    if not aptos_images_dir.exists():
        raise FileNotFoundError(f"APTOS images directory not found: {aptos_images_dir}")

    aptos_df = pd.read_csv(aptos_csv_path)

    required_aptos = {"id_code", "diagnosis"}
    missing_aptos = required_aptos - set(aptos_df.columns)
    if missing_aptos:
        raise ValueError(f"APTOS CSV is missing required columns: {missing_aptos}")

    aptos_df = aptos_df[["id_code", "diagnosis"]].copy()
    aptos_stem_map = _build_stem_index(aptos_images_dir)
    aptos_df["image_path"] = aptos_df["id_code"].map(aptos_stem_map)
    aptos_df = aptos_df.dropna(subset=["image_path"]).reset_index(drop=True)
    aptos_df["diagnosis"] = aptos_df["diagnosis"].astype(int)

    if aptos_df.empty:
        raise ValueError("No APTOS images could be matched to the CSV id_codes. "
                         "Check that aptos_images_dir contains the correct files.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, val_index = next(skf.split(aptos_df["image_path"], aptos_df["diagnosis"]))

    aptos_train = aptos_df.iloc[train_index].copy()
    aptos_val   = aptos_df.iloc[val_index].copy()

    # ── Messidor-2 ─────────────────────────────────────────────────────────
    if not messidor_csv_path.exists():
        raise FileNotFoundError(f"Messidor-2 CSV not found: {messidor_csv_path}")
    if not messidor_images_dir.exists():
        raise FileNotFoundError(f"Messidor-2 images directory not found: {messidor_images_dir}")

    messidor_df = pd.read_csv(messidor_csv_path)

    required_messidor = {"image_filename", "grade"}
    missing_mess = required_messidor - set(messidor_df.columns)
    if missing_mess:
        raise ValueError(f"Messidor-2 CSV is missing required columns: {missing_mess}")

    # Filter to adjudicated images only (if column present).
    if "adjudicated" in messidor_df.columns:
        before = len(messidor_df)
        messidor_df = messidor_df[messidor_df["adjudicated"].apply(_as_bool)]
        print(f"[dataset] Messidor-2: kept {len(messidor_df)}/{before} adjudicated rows.")

    # Filter to gradable images only (if column present).
    if "gradable" in messidor_df.columns:
        before = len(messidor_df)
        messidor_df = messidor_df[messidor_df["gradable"].apply(_as_bool)]
        print(f"[dataset] Messidor-2: kept {len(messidor_df)}/{before} gradable rows.")

    messidor_df = messidor_df.copy()
    messidor_df["grade"] = messidor_df["grade"].astype(int)
    messidor_df["image_path"] = messidor_df["image_filename"].map(
        lambda x: str(messidor_images_dir / str(x))
    )
    before = len(messidor_df)
    messidor_df = messidor_df[messidor_df["image_path"].map(lambda p: Path(p).exists())]
    messidor_df = messidor_df.reset_index(drop=True)
    if len(messidor_df) < before:
        print(f"[dataset] Messidor-2: {before - len(messidor_df)} image paths not found on disk.")

    if messidor_df.empty:
        raise ValueError("No Messidor-2 images found after filtering. "
                         "Check messidor_images_dir and image_filename values in the CSV.")

    mess_train, mess_test = train_test_split(
        messidor_df,
        test_size=0.3,
        random_state=42,
        stratify=messidor_df["grade"] if messidor_df["grade"].nunique() > 1 else None,
    )

    # ── Assemble record lists ───────────────────────────────────────────────
    train_records: list[Record] = [
        Record(str(r.image_path), int(r.diagnosis), "aptos")
        for r in aptos_train.itertuples(index=False)
    ] + [
        Record(str(r.image_path), int(r.grade), "messidor2")
        for r in mess_train.itertuples(index=False)
    ]

    val_records: list[Record] = [
        Record(str(r.image_path), int(r.diagnosis), "aptos")
        for r in aptos_val.itertuples(index=False)
    ]

    test_records: list[Record] = [
        Record(str(r.image_path), int(r.grade), "messidor2")
        for r in mess_test.itertuples(index=False)
    ]

    # ── Validate that all 5 grades are present in training data ────────────
    _validate_label_range(train_records, "train")

    # ── Log split sizes ─────────────────────────────────────────────────────
    aptos_train_n = sum(1 for r in train_records if r.source == "aptos")
    mess_train_n  = sum(1 for r in train_records if r.source == "messidor2")
    print(
        f"\n[dataset] Split sizes:\n"
        f"  Train : {len(train_records):>5}  "
        f"(APTOS={aptos_train_n}, Messidor-2={mess_train_n})\n"
        f"  Val   : {len(val_records):>5}  (APTOS only)\n"
        f"  Test  : {len(test_records):>5}  (Messidor-2 only)\n"
    )

    return {"train": train_records, "val": val_records, "test": test_records}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DRDualBranchDataset(Dataset):
    """
    Returns two pre-processed views of each fundus image plus its label:
      - x0: Ben-Graham-normalised, resized to b0_size × b0_size (Branch 0).
      - x3: CLAHE-enhanced, resized to b3_size × b3_size (Branch 3).
    Both are ImageNet-normalised float32 tensors (C×H×W).
    """

    def __init__(
        self,
        records: list[Record],
        is_train: bool,
        use_hist_matching: bool,
        hist_reference_rgb: np.ndarray | None,
        b0_size: int = 224,
        b3_size: int = 300,
    ):
        self.records = records
        self.is_train = is_train
        self.use_hist_matching = use_hist_matching
        self.hist_reference_rgb = hist_reference_rgb
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
        image_rgb = load_image_rgb(rec.image_path)

        # Step 1: circular crop to isolate retinal disc (Sec 2.2).
        base = circular_crop(image_rgb, target_size=None)

        # Step 2: histogram matching for Messidor-2 images only (Sec 2.2).
        if self.use_hist_matching and rec.source == "messidor2":
            if self.hist_reference_rgb is None:
                raise ValueError(
                    "hist_reference_rgb must be provided when use_hist_matching=True"
                )
            base_512 = cv2.resize(base, (512, 512), interpolation=cv2.INTER_AREA)
            base = histogram_match_rgb(base_512, self.hist_reference_rgb)

        # Step 3: training augmentations (none for val/test, Sec 2.2).
        if self.train_aug is not None:
            base = self.train_aug(image=base)["image"]

        # Step 4: resize and apply branch-specific enhancement.
        b0_img = cv2.resize(base, (self.b0_size, self.b0_size), interpolation=cv2.INTER_AREA)
        b3_img = cv2.resize(base, (self.b3_size, self.b3_size), interpolation=cv2.INTER_AREA)

        # Branch 0: Ben Graham normalisation → [0,1] float32
        b0_img = ben_graham_normalize(b0_img)
        # Branch 3: CLAHE enhancement → [0,1] float32
        b3_img = clahe_enhance(b3_img).astype(np.float32) / 255.0

        # Step 5: ImageNet mean/std normalisation for both branches.
        x0 = self._normalize_imagenet(b0_img)
        x3 = self._normalize_imagenet(b3_img)

        y = torch.tensor(rec.label, dtype=torch.long)
        return x0, x3, y


# ─────────────────────────────────────────────────────────────────────────────
# Histogram reference selection
# ─────────────────────────────────────────────────────────────────────────────

def build_hist_reference_from_aptos(aptos_records: list[Record]) -> np.ndarray:
    """
    Build a mean reference image from a sample of APTOS training images for
    histogram matching. Averaging over multiple images produces a more stable
    colour/intensity target than using a single arbitrary image, which reduces
    domain shift variance when matching Messidor-2 images.

    Uses up to 50 evenly-spaced records for efficiency.

    Returns:
        512×512 uint8 RGB mean reference image.
    """
    if not aptos_records:
        raise ValueError("No APTOS records available to build histogram reference.")

    # Sample up to 50 evenly-spaced images for a stable mean
    n = min(50, len(aptos_records))
    step = max(1, len(aptos_records) // n)
    sampled = aptos_records[::step][:n]

    accum = None
    count = 0
    for rec in sampled:
        try:
            rgb = load_image_rgb(rec.image_path)
            crop = circular_crop(rgb, target_size=None)
            resized = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
            accum = resized if accum is None else accum + resized
            count += 1
        except Exception:
            continue

    if count == 0:
        raise ValueError("Could not load any APTOS reference images.")

    mean_ref = (accum / count).clip(0, 255).astype(np.uint8)
    print(f"[dataset] Histogram reference built from {count} APTOS images (mean).")
    return mean_ref


# ─────────────────────────────────────────────────────────────────────────────
# Weighted sampler
# ─────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler(records: list[Record]) -> WeightedRandomSampler:
    """
    Class-balanced weighted sampler (Sec 2.7).
    Weight for sample i = 1 / (num_classes * class_count[label_i]).
    """
    labels = np.array([r.label for r in records], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_counts = np.maximum(class_counts, 1)  # guard against zero counts

    sample_weights = torch.tensor(
        [1.0 / (NUM_CLASSES * class_counts[r.label]) for r in records],
        dtype=torch.double,
    )
    return WeightedRandomSampler(sample_weights, num_samples=len(records), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build and return (train_loader, val_loader, test_loader)."""
    splits = build_data_splits(config)

    aptos_train_records = [r for r in splits["train"] if r.source == "aptos"]
    hist_ref = build_hist_reference_from_aptos(aptos_train_records)

    train_dataset = DRDualBranchDataset(
        records=splits["train"],
        is_train=True,
        use_hist_matching=config["data"]["histogram_matching_for_messidor"],
        hist_reference_rgb=hist_ref,
        b0_size=config["model"]["b0_input_size"],
        b3_size=config["model"]["b3_input_size"],
    )

    val_dataset = DRDualBranchDataset(
        records=splits["val"],
        is_train=False,
        use_hist_matching=False,  # val is APTOS only → no histogram matching needed
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

    sampler = build_weighted_sampler(splits["train"])

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
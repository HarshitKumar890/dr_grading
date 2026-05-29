"""
preprocess_offline.py — Offline preprocessing for the DR grading training set.

Runs circular crop + histogram matching (Messidor-2 only) once per image and
saves the result as a uint8 RGB .npy file.  During training, the dataset loads
these cached base images and only applies the cheap per-epoch steps
(augmentation, resize, Ben-Graham / CLAHE, ImageNet normalisation).

Output layout
-------------
data/preprocessed/train/
    <stem>.npy          # uint8 RGB array, variable H×W×3
data/preprocessed/train_labels.csv
    preprocessed_path, label, source

Usage
-----
    cd dr_baseline
    python preprocess_offline.py --config config.yaml

Re-running overwrites existing files (idempotent).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ── local imports (run from dr_baseline/) ────────────────────────────────────
from dataset import Record, build_data_splits, build_hist_reference_from_aptos
from preprocess import circular_crop, histogram_match_rgb, load_image_rgb


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stem(image_path: str) -> str:
    """Return the filename stem, e.g. '000c1434d8d7' from a full path."""
    return Path(image_path).stem


def _out_path(out_dir: Path, image_path: str, source: str) -> Path:
    """
    Build the output .npy path.  Messidor-2 filenames can collide across bases,
    so prefix with source to guarantee uniqueness.
    """
    stem = _stem(image_path)
    if source == "messidor2":
        # Use parent folder name as extra disambiguator (e.g. Base11_Base11_<stem>)
        parts = Path(image_path).parts
        # grab last two directory components before the filename
        prefix = "_".join(parts[-3:-1]) if len(parts) >= 3 else source
        filename = f"{prefix}_{stem}.npy"
    else:
        filename = f"{stem}.npy"
    return out_dir / filename


# ─────────────────────────────────────────────────────────────────────────────
# Core preprocessing for one record
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_record(
    rec: Record,
    out_dir: Path,
    use_hist_matching: bool,
    hist_reference_rgb: np.ndarray | None,
    overwrite: bool = True,
) -> str:
    """
    Preprocess a single record and save to disk.

    Steps (mirrors DRDualBranchDataset.__getitem__ steps 1-2):
      1. Load original image as uint8 RGB.
      2. Circular crop (no resize yet — keep full resolution).
      3. If Messidor-2 and hist matching enabled:
           resize base to 512×512, apply histogram_match_rgb.
      4. Save as .npy (uint8 RGB).

    Returns the path to the saved .npy file.
    """
    npy_path = _out_path(out_dir, rec.image_path, rec.source)

    if not overwrite and npy_path.exists():
        return str(npy_path)

    # Step 1: load
    image_rgb = load_image_rgb(rec.image_path)

    # Step 2: circular crop (deterministic, no resize)
    base = circular_crop(image_rgb, target_size=None)

    # Step 3: histogram matching for Messidor-2
    if use_hist_matching and rec.source == "messidor2":
        if hist_reference_rgb is None:
            raise ValueError("hist_reference_rgb required for Messidor-2 histogram matching.")
        base_512 = cv2.resize(base, (512, 512), interpolation=cv2.INTER_AREA)
        base = histogram_match_rgb(base_512, hist_reference_rgb)

    # Step 4: save
    np.save(str(npy_path), base)
    return str(npy_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_offline_preprocessing(config: dict, overwrite: bool = True) -> None:
    """
    Preprocess the full training split and write train_labels.csv.

    Args:
        config   : loaded config dict (from config.yaml).
        overwrite: if False, skip images whose .npy already exists.
    """
    # Resolve output directory relative to config's output_dir parent
    # so it sits alongside data/ regardless of cwd.
    out_dir = Path("../data/preprocessed/train")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = Path("../data/preprocessed/train_labels.csv")

    print(f"[offline] Output directory : {out_dir.resolve()}")
    print(f"[offline] Labels CSV       : {csv_out.resolve()}")

    # ── Build splits (same logic as training) ────────────────────────────────
    splits = build_data_splits(config)
    train_records: list[Record] = splits["train"]

    # ── Build histogram reference from APTOS training subset ─────────────────
    use_hist = bool(config["data"].get("histogram_matching_for_messidor", True))
    hist_ref: np.ndarray | None = None
    if use_hist:
        aptos_records = [r for r in train_records if r.source == "aptos"]
        hist_ref = build_hist_reference_from_aptos(aptos_records)

    # ── Process each record ───────────────────────────────────────────────────
    rows: list[dict] = []
    errors: list[str] = []

    for rec in tqdm(train_records, desc="Preprocessing", unit="img", dynamic_ncols=True):
        try:
            npy_path = preprocess_record(
                rec,
                out_dir=out_dir,
                use_hist_matching=use_hist,
                hist_reference_rgb=hist_ref,
                overwrite=overwrite,
            )
            rows.append({
                "preprocessed_path": npy_path,
                "label": rec.label,
                "source": rec.source,
            })
        except Exception as exc:
            errors.append(f"{rec.image_path}: {exc}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=["preprocessed_path", "label", "source"])
    df.to_csv(str(csv_out), index=False)

    print(f"\n[offline] Done.")
    print(f"  Processed : {len(rows):>5} images")
    print(f"  Errors    : {len(errors):>5} images")
    print(f"  CSV saved : {csv_out.resolve()}")

    if errors:
        print("\n[offline] Failed images:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more.")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline preprocessing for DR training set")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip images whose .npy already exists (resume mode)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_offline_preprocessing(config, overwrite=not args.no_overwrite)

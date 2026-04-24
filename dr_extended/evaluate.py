"""
evaluate.py — Evaluation script for extended DR grading model.

Supports both baseline and extended (DANN + attention) checkpoints.
The model forward() now returns (logits, fused); this script unpacks correctly.

Usage:
    python evaluate.py --config config.yaml \
        --checkpoint outputs/checkpoints/best_model.pth --split both
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset_preprocessed import build_dataloaders_preprocessed
from model import DRGradingModel
from utils import compute_qwk_from_logits, confusion_matrix_from_logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate extended DR grading model")
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pth")
    p.add_argument("--split",      default="both", choices=["val", "test", "both"])
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_eval(
    model: DRGradingModel,
    loader,
    device: torch.device,
    use_amp: bool,
    split_name: str,
) -> tuple[float, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for x0, x3, y in tqdm(loader, desc=f"Evaluating {split_name}"):
            x0 = x0.to(device, non_blocking=True)
            x3 = x3.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits, _ = model(x0, x3)   # unpack tuple

            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)

    qwk = compute_qwk_from_logits(logits_cat, labels_cat)
    cm  = confusion_matrix_from_logits(logits_cat, labels_cat, num_classes=5)

    print(f"\n{split_name.upper()} Results:")
    print(f"  QWK: {qwk:.4f}")
    print(f"  Confusion Matrix (5×5):\n{cm}")
    return qwk, cm


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    device = torch.device("cuda:0")
    print(f"Using device: {device} | GPU: {torch.cuda.get_device_name(0)}")

    _, val_loader, test_loader = build_dataloaders_preprocessed(config)

    model = DRGradingModel(
        num_classes=5, fusion_dim=config["model"]["fusion_dim"]
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "best_val_qwk" in ckpt:
        print(f"Checkpoint best val QWK: {ckpt['best_val_qwk']:.4f}")

    use_amp    = bool(config["train"]["use_amp"])
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split in {"val", "both"}:
        val_qwk, val_cm = run_eval(model, val_loader, device, use_amp, "val")
        np.save(output_dir / "val_confusion_matrix.npy", val_cm)
        (output_dir / "val_qwk.txt").write_text(f"{val_qwk:.6f}\n")
        print(f"✓ Saved val results to {output_dir}")

    if args.split in {"test", "both"}:
        test_qwk, test_cm = run_eval(model, test_loader, device, use_amp, "test")
        np.save(output_dir / "test_confusion_matrix.npy", test_cm)
        (output_dir / "test_qwk.txt").write_text(f"{test_qwk:.6f}\n")
        print(f"✓ Saved test results to {output_dir}")


if __name__ == "__main__":
    main()

"""
evaluate.py — Evaluation script for dual-branch DR grading baseline.

Evaluates trained model on validation (APTOS) and/or test (Messidor-2) splits.
Computes Quadratic Weighted Kappa (QWK) and confusion matrices.

Usage:
    python evaluate.py --config config.yaml --checkpoint outputs/checkpoints/best_model.pth --split both

CHANGELOG:
  - Enhanced documentation for evaluation workflow
  - Verified robust checkpoint loading with error handling
  - Ensured proper split separation (val=APTOS, test=Messidor-2)
  - Added result saving to output directory (QWK txt files, confusion matrix npy)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset import build_dataloaders
from model import DRGradingModel
from utils import compute_qwk_from_logits, confusion_matrix_from_logits


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate dual-branch DR baseline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help="Evaluation split: val (APTOS), test (Messidor-2), or both",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_eval(model: torch.nn.Module, loader, device: torch.device, use_amp: bool, split_name: str) -> tuple[float, np.ndarray]:
    """
    Run evaluation on a single data split.
    
    Args:
        model: trained DRGradingModel
        loader: DataLoader for the split
        device: torch device (cuda:0)
        use_amp: whether to use automatic mixed precision
        split_name: name of split for logging ("val" or "test")
    
    Returns:
        qwk: Quadratic Weighted Kappa score
        cm: confusion matrix (num_classes × num_classes)
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x0, x3, y in tqdm(loader, desc=f"Evaluating {split_name}"):
            x0 = x0.to(device, non_blocking=True)
            x3 = x3.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x0, x3)

            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)

    qwk = compute_qwk_from_logits(logits_cat, labels_cat)
    cm = confusion_matrix_from_logits(logits_cat, labels_cat, num_classes=5)

    print(f"\n{split_name.upper()} Results:")
    print(f"  QWK: {qwk:.4f}")
    print(f"  Confusion Matrix (5×5):\n{cm}")

    return qwk, cm


def main() -> None:
    """
    Main evaluation workflow.
    
    Loads best checkpoint and evaluates on requested splits:
      - val: APTOS validation set (internal validation)
      - test: Messidor-2 test set (external cross-dataset evaluation)
      - both: evaluate on both splits
    
    Results are saved to output directory:
      - {split}_qwk.txt: QWK score
      - {split}_confusion_matrix.npy: confusion matrix
    """
    args = parse_args()
    config = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Evaluation expects cuda:0.")

    device = torch.device("cuda:0")
    print(f"Using device: {device} | GPU: {torch.cuda.get_device_name(0)}")

    _, val_loader, test_loader = build_dataloaders(config)

    model = DRGradingModel(num_classes=5, fusion_dim=config["model"]["fusion_dim"]).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if "best_val_qwk" in checkpoint:
        print(f"Checkpoint best validation QWK: {checkpoint['best_val_qwk']:.4f}")

    use_amp = bool(config["train"]["use_amp"])

    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate on requested splits
    if args.split in {"val", "both"}:
        val_qwk, val_cm = run_eval(model, val_loader, device, use_amp, split_name="val")
        np.save(output_dir / "val_confusion_matrix.npy", val_cm)
        with open(output_dir / "val_qwk.txt", "w", encoding="utf-8") as f:
            f.write(f"{val_qwk:.6f}\n")
        print(f"✓ Saved validation results to {output_dir}")

    if args.split in {"test", "both"}:
        test_qwk, test_cm = run_eval(model, test_loader, device, use_amp, split_name="test")
        np.save(output_dir / "test_confusion_matrix.npy", test_cm)
        with open(output_dir / "test_qwk.txt", "w", encoding="utf-8") as f:
            f.write(f"{test_qwk:.6f}\n")
        print(f"✓ Saved test results to {output_dir}")


if __name__ == "__main__":
    main()

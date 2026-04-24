"""
train.py — Training script for dual-branch EfficientNet DR grading baseline.

Implements the training strategy described in Section 2.7 of the paper:
  - Adam optimizer with lr=1e-4
  - Mixed precision training (AMP) for RTX 4060 8GB memory efficiency
  - Weighted random sampler for class balance
  - Early stopping based on validation QWK (patience=5)
  - TensorBoard logging and training curve visualization

CHANGELOG:
  - Enhanced documentation for training loop and evaluation
  - Verified complete checkpoint saving (epoch, model, optimizer, best_qwk, config)
  - Ensured proper mixed precision training with GradScaler
  - Added robust early stopping with patience monitoring
  - Confirmed TensorBoard logging for all metrics
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Suppress OpenCV TIFF warnings from Messidor-2 proprietary metadata tags
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
from tqdm import tqdm

from dataset import build_dataloaders
from dataset_preprocessed import build_dataloaders_preprocessed
from losses import coral_loss
from model import DRGradingModel
from utils import compute_qwk_from_logits, plot_training_curves, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Dual-Branch EfficientNet DR baseline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_epoch(model: torch.nn.Module, loader, device: torch.device, use_amp: bool) -> tuple[float, float]:
    """
    Evaluate model on validation set for one epoch.
    
    Args:
        model: DRGradingModel instance
        loader: validation DataLoader
        device: torch device (cuda:0)
        use_amp: whether to use automatic mixed precision
    
    Returns:
        avg_loss: average CORAL loss over validation set
        qwk: Quadratic Weighted Kappa score
    """
    model.eval()
    all_logits = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for x0, x3, y in tqdm(loader, desc="Validation", leave=False):
            x0 = x0.to(device, non_blocking=True)
            x3 = x3.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x0, x3)
                loss = coral_loss(logits, y, num_classes=5)

            losses.append(loss.item())
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    qwk = compute_qwk_from_logits(logits_cat, labels_cat)
    return float(sum(losses) / max(1, len(losses))), float(qwk)


def train() -> None:
    """
    Main training loop implementing paper Section 2.7 methodology.
    
    Training configuration:
      - Optimizer: Adam (lr=1e-4, weight_decay=0.0)
      - Loss: CORAL ordinal regression loss
      - Sampler: Weighted random sampler for class balance
      - Mixed precision: AMP with GradScaler for memory efficiency
      - Early stopping: patience=5 epochs on validation QWK
      - Checkpointing: save best model based on validation QWK
    """
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["train"]["seed"])

    # Verify CUDA availability (required for RTX 4060 8GB training)
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if not cuda_available:
        raise RuntimeError("CUDA is not available. This baseline requires GPU training on cuda:0.")

    device = torch.device("cuda:0")
    print(f"Using device: {device} | GPU: {torch.cuda.get_device_name(0)}")

    torch.backends.cudnn.benchmark = True

    use_preprocessed = bool(config.get("use_preprocessed", False))
    if use_preprocessed:
        print("Using offline-preprocessed dataset (fast .npy loading).")
        train_loader, val_loader, _ = build_dataloaders_preprocessed(config)
    else:
        train_loader, val_loader, _ = build_dataloaders(config)

    model = DRGradingModel(num_classes=5, fusion_dim=config["model"]["fusion_dim"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    use_amp = bool(config["train"]["use_amp"])
    scaler = GradScaler('cuda', enabled=use_amp)

    output_dir = Path(config["output"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))

    best_qwk = -1.0
    epochs_without_improve = 0
    patience = config["train"]["early_stopping_patience"]

    history = {"train_loss": [], "val_qwk": []}

    for epoch in range(config["train"]["epochs"]):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['train']['epochs']}")
        for x0, x3, y in pbar:
            x0 = x0.to(device, non_blocking=True)
            x3 = x3.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x0, x3)
                loss = coral_loss(logits, y, num_classes=5)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            del x0, x3

        train_loss = running_loss / max(1, len(train_loader))
        val_loss, val_qwk = evaluate_epoch(model, val_loader, device, use_amp)

        history["train_loss"].append(train_loss)
        history["val_qwk"].append(val_qwk)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("QWK/val", val_qwk, epoch)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_qwk={val_qwk:.4f}"
        )

        # Save checkpoint if validation QWK improved
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            epochs_without_improve = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_qwk": best_qwk,
                    "config": config,
                },
                str(ckpt_dir / "best_model.pth"),
            )
            print(f"✓ Saved best checkpoint with Val QWK={best_qwk:.4f}")
        else:
            epochs_without_improve += 1

        # Early stopping check
        if epochs_without_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1} (patience={patience}).")
            break

    writer.close()
    plot_training_curves(history, str(output_dir))

    print(f"\nTraining complete. Best validation QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    train()

"""
utils.py — Utility functions for DR grading baseline.

Provides helper functions for:
  - Reproducible random seeding
  - QWK computation from CORAL logits
  - Confusion matrix generation
  - Checkpoint saving and loading
  - Training curve visualization

CHANGELOG:
  - Enhanced documentation for all functions
  - Verified QWK computation: sigmoid → threshold → sum → cohen_kappa_score
  - Ensured checkpoint saving includes complete training state
  - Added publication-quality plotting with proper styling
"""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: random seed value (default 42 per paper)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_qwk_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Quadratic Weighted Kappa from CORAL logits.
    
    CORAL logits represent K-1 binary thresholds. To get the predicted class:
      1. Apply sigmoid to get probabilities for each threshold
      2. Threshold at 0.5 to get binary decisions
      3. Sum the binary decisions to get the predicted ordinal class
    
    Example:
        logits = [-2, -1, 3, 4] → probs = [0.12, 0.27, 0.95, 0.98]
        → binary = [0, 0, 1, 1] → predicted class = 2
    
    Args:
        logits: (N, num_classes-1) CORAL logits
        labels: (N,) ground truth ordinal labels
    
    Returns:
        qwk: Quadratic Weighted Kappa score in range [-1, 1]
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).sum(dim=1)
    return cohen_kappa_score(labels.cpu().numpy(), preds.cpu().numpy(), weights="quadratic")


def confusion_matrix_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> np.ndarray:
    """
    Generate confusion matrix from CORAL logits.
    
    Args:
        logits: (N, num_classes-1) CORAL logits
        labels: (N,) ground truth ordinal labels
        num_classes: number of ordinal classes (default 5 for DR grades 0-4)
    
    Returns:
        cm: (num_classes, num_classes) confusion matrix
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).sum(dim=1)
    return confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=list(range(num_classes)))


def save_checkpoint(state: dict, checkpoint_path: str) -> None:
    """
    Save training checkpoint with complete state for resumable training.
    
    Expected state dict keys:
      - epoch: current epoch number
      - model_state_dict: model parameters
      - optimizer_state_dict: optimizer state
      - best_val_qwk: best validation QWK achieved
      - config: training configuration dict
    
    Args:
        state: dictionary containing training state
        checkpoint_path: path to save checkpoint file
    """
    os.makedirs(str(Path(checkpoint_path).parent), exist_ok=True)
    torch.save(state, checkpoint_path)


def plot_training_curves(history: dict, output_dir: str) -> None:
    """
    Generate publication-quality training curve plots.
    
    Creates two separate plots:
      1. Training loss curve (CORAL loss over epochs)
      2. Validation QWK curve (QWK over epochs)
    
    Args:
        history: dict with keys 'train_loss' and 'val_qwk' (lists of values per epoch)
        output_dir: directory to save plot PNG files
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot 1: Training Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train CORAL Loss", linewidth=2, color='#2E86AB')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "train_loss_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Validation QWK
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_qwk"], label="Val QWK", linewidth=2, color='#A23B72')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("QWK", fontsize=12)
    plt.title("Validation QWK Curve", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "val_qwk_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

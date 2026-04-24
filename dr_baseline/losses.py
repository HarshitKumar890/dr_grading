"""
losses.py — CORAL ordinal regression loss for DR grading.

Implements the Cumulative Ordinal Regression with Logistic loss (CORAL)
as described in Section 2.5 of the paper. CORAL treats ordinal classification
as K-1 binary classification tasks with rank consistency.

For a label k ∈ {0, 1, ..., K-1}, the binary targets are:
  - threshold i is exceeded if k > i (i.e., k ≥ i+1)
  - this creates cumulative binary targets: [1,1,...,1,0,0,...,0]

CHANGELOG:
  - Added proper docstrings explaining CORAL methodology
  - Ensured BCEWithLogitsLoss uses default reduction='mean'
  - Verified rank-consistent binary target generation
"""

import torch
import torch.nn as nn


def labels_to_levels(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    Convert ordinal labels to binary cumulative targets for CORAL.
    
    For K=5 classes (labels 0-4), we have K-1=4 binary thresholds.
    Label k produces binary targets where threshold i is exceeded if k > i.
    
    Example:
        label=0 → [0, 0, 0, 0]  (no thresholds exceeded)
        label=2 → [1, 1, 0, 0]  (first 2 thresholds exceeded)
        label=4 → [1, 1, 1, 1]  (all thresholds exceeded)
    
    Args:
        labels: (B,) tensor of ordinal labels in range [0, num_classes-1]
        num_classes: number of ordinal classes (default 5 for DR grades)
    
    Returns:
        levels: (B, num_classes-1) binary tensor of cumulative targets
    """
    # Create comparison: label > [0, 1, 2, 3] for num_classes=5
    # This gives True where the label exceeds each threshold
    levels = labels.unsqueeze(1) > torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    return levels.float()


def coral_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    Compute CORAL ordinal regression loss.
    
    CORAL loss is the binary cross-entropy between predicted logits and
    cumulative binary targets derived from ordinal labels. This enforces
    rank consistency: if threshold i is exceeded, all thresholds < i must
    also be exceeded.
    
    Args:
        logits: (B, num_classes-1) raw logits from CORAL head
        labels: (B,) ordinal labels in range [0, num_classes-1]
        num_classes: number of ordinal classes (default 5)
    
    Returns:
        loss: scalar CORAL loss (mean over batch and thresholds)
    """
    levels = labels_to_levels(labels, num_classes=num_classes)
    # BCEWithLogitsLoss with default reduction='mean' averages over all elements
    return nn.BCEWithLogitsLoss(reduction='mean')(logits, levels)

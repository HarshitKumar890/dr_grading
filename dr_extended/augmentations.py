"""
augmentations.py — Training data augmentation for DR grading.

Implements the augmentation strategy described in Section 2.2 of the paper:
  - Horizontal flip (p=0.5)
  - Random brightness and contrast adjustment
  - Random gamma correction

These augmentations are applied ONLY during training to improve generalization
and prevent overfitting. No augmentations are applied during validation or testing.

CHANGELOG:
  - Added documentation confirming paper-compliant augmentation strategy
  - Verified: only horizontal flip, brightness/contrast, and gamma are used
  - No rotation, vertical flip, or other augmentations per paper specification
"""

import albumentations as A


def build_train_augmentation() -> A.Compose:
    """
    Build training augmentation pipeline per paper Section 2.2.
    
    Augmentations applied:
      1. HorizontalFlip: 50% probability (fundus images are symmetric)
      2. RandomBrightnessContrast: ±20% adjustment to handle lighting variations
      3. RandomGamma: gamma ∈ [0.8, 1.2] to simulate exposure differences
    
    Returns:
        Albumentations Compose object for training augmentation
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ]
    )

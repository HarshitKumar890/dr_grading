"""
model.py — Dual-branch EfficientNet + attention gate + CORAL head.

Architecture (Section 2.3–2.5):
  - Branch 0: EfficientNet-B0, pretrained, classifier removed → 1280-d features
               projected to fusion_dim (1536) via Linear.
  - Branch 3: EfficientNet-B3, pretrained, classifier removed → 1536-d features
               projected to fusion_dim (1536) via Linear (identity-like, but keeps
               the interface consistent and allows fusion_dim to be changed freely).
  - Fusion gate: concat(f0_proj, f3_proj) → MLP → sigmoid gate g ∈ (0,1)^fusion_dim
                 fused = g ⊙ f0_proj + (1-g) ⊙ f3_proj        (Eq. 1-2 in paper)
  - CORAL head: Linear(fusion_dim, num_classes - 1) → K-1 = 4 logits.

CHANGELOG (vs original):
  - Added explicit proj_b3 Linear to project B3 features to fusion_dim.
    Previously f3 was used raw (1536-d), which only worked if fusion_dim == 1536.
    Now both branches are projected symmetrically, making fusion_dim a free
    hyperparameter and the architecture consistent with the paper description.
  - Added B0_FEATURE_DIM / B3_FEATURE_DIM constants for clarity.
  - Minor: moved feature-dim constants out of __init__ for readability.
"""

import timm
import torch
import torch.nn as nn

# Native feature dimensions output by each backbone with num_classes=0.
_B0_FEATURE_DIM = 1280
_B3_FEATURE_DIM = 1536


class DRGradingModel(nn.Module):
    """
    Dual-resolution attention-gated EfficientNet with CORAL ordinal head.

    Args:
        num_classes: number of DR severity grades (default 5, giving K-1=4 logits).
        fusion_dim : common projection dimension for both branches (default 1536).
    """

    def __init__(self, num_classes: int = 5, fusion_dim: int = 1536):
        super().__init__()

        # ── Backbones (classifiers removed, returns pooled feature vectors) ──
        self.b0 = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.b3 = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)

        # ── Branch projections → common fusion_dim ──
        # FIX: proj_b3 added so both branches live in the same space and
        # fusion_dim is not hard-coupled to the B3 native dimension (1536).
        self.proj_b0 = nn.Linear(_B0_FEATURE_DIM, fusion_dim)
        self.proj_b3 = nn.Linear(_B3_FEATURE_DIM, fusion_dim)

        # ── Attention / gating MLP (squeeze-and-excitation style, Sec 2.4) ──
        # Input: [f0_proj || f3_proj]  →  2*fusion_dim
        # Output: gate g ∈ (0,1)^fusion_dim
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        # ── CORAL ordinal regression head (Sec 2.5) ──
        # Outputs K-1 logits for K classes (4 logits for 5 DR grades).
        self.coral = nn.Linear(fusion_dim, num_classes - 1)

    def forward(self, x0: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: (B, 3, 224, 224) — Ben-Graham-normalised input for B0 branch.
            x3: (B, 3, 300, 300) — CLAHE-enhanced input for B3 branch.

        Returns:
            logits: (B, num_classes - 1) CORAL logits.
        """
        # ── Feature extraction ──
        f0 = self.b0(x0)   # (B, 1280)
        f3 = self.b3(x3)   # (B, 1536)

        # ── Project both branches to fusion_dim ──
        f0 = self.proj_b0(f0)   # (B, fusion_dim)
        f3 = self.proj_b3(f3)   # (B, fusion_dim)

        # ── Attention gate (Eq. 1-2) ──
        f_concat = torch.cat([f0, f3], dim=1)   # (B, 2*fusion_dim)
        g = self.fusion_gate(f_concat)           # (B, fusion_dim), values ∈ (0,1)
        fused = g * f0 + (1.0 - g) * f3         # (B, fusion_dim)

        # ── CORAL ordinal head ──
        return self.coral(fused)                 # (B, num_classes-1)
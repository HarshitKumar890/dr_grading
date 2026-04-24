"""
model.py — Dual-branch EfficientNet + attention gate + CORAL head.

Extended with:
  - GradientReversalLayer (GRL) for domain adversarial adaptation (DANN)
  - Domain classifier head operating on fused features
  - Forward hooks on conv_head layers for attention consistency loss
  - forward() now returns (logits, fused) tuple
  - forward_domain() for domain classification with gradient reversal
  - compute_fused_attention() static method for attention consistency
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Native feature dimensions output by each backbone with num_classes=0.
_B0_FEATURE_DIM = 1280
_B3_FEATURE_DIM = 1536


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer
# ─────────────────────────────────────────────────────────────────────────────

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal layer for domain adversarial training (DANN).
    Forward pass: identity.
    Backward pass: negate and scale gradients by lambda_.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Apply gradient reversal with scaling factor lambda_."""
    return GradientReversalFunction.apply(x, lambda_)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class DRGradingModel(nn.Module):
    """
    Dual-resolution attention-gated EfficientNet with CORAL ordinal head,
    domain adversarial classifier, and attention consistency support.

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
        self.proj_b0 = nn.Linear(_B0_FEATURE_DIM, fusion_dim)
        self.proj_b3 = nn.Linear(_B3_FEATURE_DIM, fusion_dim)

        # ── Attention / gating MLP (squeeze-and-excitation style, Sec 2.4) ──
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        # ── CORAL ordinal regression head (Sec 2.5) ──
        self.coral = nn.Linear(fusion_dim, num_classes - 1)

        # ── Domain classifier (DANN Extension 1) ──
        self.domain_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # ── Feature map hooks for attention consistency (Extension 2) ──
        # Hooks capture the output of conv_head (last conv before global pool)
        # Shape: (B, C, H, W) — B0: (B,1280,7,7) at 224px, B3: (B,1536,10,10) at 300px
        self.b0_feat_map: torch.Tensor | None = None
        self.b3_feat_map: torch.Tensor | None = None
        self.b0.conv_head.register_forward_hook(self._hook_b0)
        self.b3.conv_head.register_forward_hook(self._hook_b3)

    # ── Hook callbacks ────────────────────────────────────────────────────────

    def _hook_b0(self, module, input, output) -> None:
        self.b0_feat_map = output

    def _hook_b3(self, module, input, output) -> None:
        self.b3_feat_map = output

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x0: torch.Tensor, x3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x0: (B, 3, 224, 224) — Ben-Graham-normalised input for B0 branch.
            x3: (B, 3, 300, 300) — CLAHE-enhanced input for B3 branch.

        Returns:
            logits: (B, num_classes - 1) CORAL logits.
            fused : (B, fusion_dim) fused feature vector (for domain loss).
        """
        # ── Feature extraction (also fills b0_feat_map / b3_feat_map via hooks) ──
        f0 = self.b0(x0)   # (B, 1280)
        f3 = self.b3(x3)   # (B, 1536)

        # ── Project both branches to fusion_dim ──
        f0 = self.proj_b0(f0)   # (B, fusion_dim)
        f3 = self.proj_b3(f3)   # (B, fusion_dim)

        # ── Attention gate ──
        f_concat = torch.cat([f0, f3], dim=1)   # (B, 2*fusion_dim)
        g = self.fusion_gate(f_concat)           # (B, fusion_dim)
        fused = g * f0 + (1.0 - g) * f3         # (B, fusion_dim)

        # ── CORAL ordinal head ──
        logits = self.coral(fused)               # (B, num_classes-1)

        return logits, fused

    # ── Domain forward ────────────────────────────────────────────────────────

    def forward_domain(
        self, fused: torch.Tensor, lambda_grl: float
    ) -> torch.Tensor:
        """
        Domain classification with gradient reversal.

        Args:
            fused     : (B, fusion_dim) fused feature vector from forward().
            lambda_grl: GRL scaling factor (increases during training).

        Returns:
            domain_logits: (B, 1) raw logits (0=source/APTOS, 1=target/Messidor).
        """
        reversed_fused = grad_reverse(fused, lambda_grl)
        return self.domain_classifier(reversed_fused)

    # ── Attention map ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_fused_attention(
        feat0: torch.Tensor, feat3: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a fused spatial attention map from the two branch feature maps.
        Maps are min-max normalised to [0,1] per sample before averaging to
        keep the attention consistency loss in a small, stable range (< 1).

        Args:
            feat0: (B, C0, H0, W0) — B0 conv_head output, typically (B,1280,7,7).
            feat3: (B, C3, H3, W3) — B3 conv_head output, typically (B,1536,10,10).

        Returns:
            att: (B, 1, 224, 224) fused attention map in [0, 1].
        """
        # L2 norm across channel dimension → spatial saliency map
        att0 = torch.norm(feat0, p=2, dim=1, keepdim=True)   # (B,1,H0,W0)
        att3 = torch.norm(feat3, p=2, dim=1, keepdim=True)   # (B,1,H3,W3)

        # Min-max normalise per sample to [0,1] — prevents huge MSE values
        att0 = (att0 - att0.amin(dim=[2, 3], keepdim=True)) / (
            att0.amax(dim=[2, 3], keepdim=True) + 1e-8
        )
        att3 = (att3 - att3.amin(dim=[2, 3], keepdim=True)) / (
            att3.amax(dim=[2, 3], keepdim=True) + 1e-8
        )

        # Upsample both to a common 224×224 spatial size
        att0 = F.interpolate(att0, size=(224, 224), mode="bilinear", align_corners=False)
        att3 = F.interpolate(att3, size=(224, 224), mode="bilinear", align_corners=False)

        return (att0 + att3) / 2.0

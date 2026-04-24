"""
Ensemble Evaluation Script for DR Grading Models
Evaluates ensemble of baseline and attention-consistency models using two fusion methods:
1. Logit averaging
2. Probability averaging

Also includes Test-Time Augmentation (TTA) via horizontal flipping:
- Predictions are averaged over original + horizontally flipped images.
- Fundus images are left-right symmetric, so this reduces random asymmetries.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm

from model import DRGradingModel
from dataset_preprocessed import build_dataloaders_preprocessed


def load_model(checkpoint_path, num_classes, device):
    """Load a trained model from checkpoint."""
    model = DRGradingModel(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict with strict=False to ignore missing domain_classifier keys
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model


def evaluate_ensemble(model_base, model_attn, test_loader, device):
    """
    Evaluate ensemble using two fusion methods (with and without TTA).

    Standard methods:
      1. Logit averaging  — average raw logits of both models, then sigmoid → threshold → sum
      2. Probability averaging — sigmoid each model's logits, average probs, threshold → sum

    TTA methods (horizontal flip):
      3. Logit averaging + TTA  — average 4 logit tensors (orig/flip × base/attn)
      4. Probability averaging + TTA — sigmoid all 4, average probs, threshold → sum

    Both models output 4 logits (5 classes).
    """
    all_labels         = []
    all_preds_logit    = []
    all_preds_prob     = []
    all_preds_tta_logit = []
    all_preds_tta_prob  = []

    with torch.no_grad():
        for x0, x3, labels in tqdm(test_loader, desc="Evaluating ensemble"):
            x0     = x0.to(device)
            x3     = x3.to(device)
            labels = labels.to(device)

            # ── Original pass ──────────────────────────────────────────────
            base_orig = model_base(x0, x3)
            attn_orig = model_attn(x0, x3)
            logits_base_orig = base_orig[0] if isinstance(base_orig, tuple) else base_orig
            logits_attn_orig = attn_orig[0] if isinstance(attn_orig, tuple) else attn_orig

            # ── Flipped pass ───────────────────────────────────────────────
            x0_flip = torch.flip(x0, dims=[-1])
            x3_flip = torch.flip(x3, dims=[-1])
            base_flip = model_base(x0_flip, x3_flip)
            attn_flip = model_attn(x0_flip, x3_flip)
            logits_base_flip = base_flip[0] if isinstance(base_flip, tuple) else base_flip
            logits_attn_flip = attn_flip[0] if isinstance(attn_flip, tuple) else attn_flip

            # ── Method 1: Logit averaging (no TTA) ────────────────────────
            avg_logits = (logits_base_orig + logits_attn_orig) / 2.0
            preds_logit = (torch.sigmoid(avg_logits) > 0.5).sum(dim=1)

            # ── Method 2: Probability averaging (no TTA) ──────────────────
            avg_probs = (torch.sigmoid(logits_base_orig) + torch.sigmoid(logits_attn_orig)) / 2.0
            preds_prob = (avg_probs > 0.5).sum(dim=1)

            # ── Method 3: Logit averaging + TTA ───────────────────────────
            avg_logits_tta = (logits_base_orig + logits_attn_orig +
                              logits_base_flip + logits_attn_flip) / 4.0
            preds_tta_logit = (torch.sigmoid(avg_logits_tta) > 0.5).sum(dim=1)

            # ── Method 4: Probability averaging + TTA ─────────────────────
            avg_probs_tta = (torch.sigmoid(logits_base_orig) +
                             torch.sigmoid(logits_attn_orig) +
                             torch.sigmoid(logits_base_flip) +
                             torch.sigmoid(logits_attn_flip)) / 4.0
            preds_tta_prob = (avg_probs_tta > 0.5).sum(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds_logit.extend(preds_logit.cpu().numpy())
            all_preds_prob.extend(preds_prob.cpu().numpy())
            all_preds_tta_logit.extend(preds_tta_logit.cpu().numpy())
            all_preds_tta_prob.extend(preds_tta_prob.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds_logit),
        np.array(all_preds_prob),
        np.array(all_preds_tta_logit),
        np.array(all_preds_tta_prob),
    )


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hardcoded checkpoint paths
    baseline_ckpt = "E:/DR_PROJECT/dr_baseline/outputs/checkpoints/best_model.pth"
    attention_ckpt = "E:/DR_PROJECT/dr_extended/outputs/checkpoints/best_model.pth"
    
    print("\n" + "="*60)
    print("Ensemble Evaluation: Baseline + Attention-Consistency Models")
    print("="*60)
    
    # Load config
    config_path = Path("config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build test dataloader
    print("\nLoading test data...")
    _, _, test_loader = build_dataloaders_preprocessed(config)
    print(f"Test set size: {len(test_loader.dataset)} samples")
    
    # Load models (both have 5 classes / 4 CORAL logits)
    print("\nLoading models...")
    print(f"Baseline model: {baseline_ckpt}")
    model_base = load_model(baseline_ckpt, num_classes=5, device=device)
    print(f"Attention model: {attention_ckpt}")
    model_attn = load_model(attention_ckpt, num_classes=5, device=device)
    
    # Evaluate ensemble
    print("\nEvaluating ensemble (standard + TTA with horizontal flip)...")
    all_labels, preds_logit, preds_prob, preds_tta_logit, preds_tta_prob = evaluate_ensemble(
        model_base, model_attn, test_loader, device
    )

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # ── Standard results ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS (No TTA)")
    print("="*60)

    qwk_logit = cohen_kappa_score(all_labels, preds_logit, weights='quadratic')
    cm_logit  = confusion_matrix(all_labels, preds_logit, labels=[0, 1, 2, 3, 4])
    print("\n1. LOGIT AVERAGING")
    print(f"   QWK: {qwk_logit:.4f}")
    print(f"   Confusion Matrix:\n{cm_logit}")

    qwk_prob = cohen_kappa_score(all_labels, preds_prob, weights='quadratic')
    cm_prob  = confusion_matrix(all_labels, preds_prob, labels=[0, 1, 2, 3, 4])
    print("\n2. PROBABILITY AVERAGING")
    print(f"   QWK: {qwk_prob:.4f}")
    print(f"   Confusion Matrix:\n{cm_prob}")

    np.save(outputs_dir / "ensemble_logit_cm.npy", cm_logit)
    np.save(outputs_dir / "ensemble_prob_cm.npy",  cm_prob)

    # ── TTA results ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS (TTA — horizontal flip)")
    print("="*60)

    qwk_tta_logit = cohen_kappa_score(all_labels, preds_tta_logit, weights='quadratic')
    cm_tta_logit  = confusion_matrix(all_labels, preds_tta_logit, labels=[0, 1, 2, 3, 4])
    print("\n3. LOGIT AVERAGING + TTA")
    print(f"   QWK: {qwk_tta_logit:.4f}")
    print(f"   Confusion Matrix:\n{cm_tta_logit}")

    qwk_tta_prob = cohen_kappa_score(all_labels, preds_tta_prob, weights='quadratic')
    cm_tta_prob  = confusion_matrix(all_labels, preds_tta_prob, labels=[0, 1, 2, 3, 4])
    print("\n4. PROBABILITY AVERAGING + TTA")
    print(f"   QWK: {qwk_tta_prob:.4f}")
    print(f"   Confusion Matrix:\n{cm_tta_prob}")

    np.save(outputs_dir / "ensemble_tta_logit_cm.npy", cm_tta_logit)
    np.save(outputs_dir / "ensemble_tta_prob_cm.npy",  cm_tta_prob)

    print(f"\n✓ All confusion matrices saved to {outputs_dir}/")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Logit Averaging          QWK: {qwk_logit:.4f}")
    print(f"Probability Averaging    QWK: {qwk_prob:.4f}")
    print(f"Logit Averaging + TTA    QWK: {qwk_tta_logit:.4f}  ← TTA")
    print(f"Probability Avg  + TTA   QWK: {qwk_tta_prob:.4f}  ← TTA")

    best_qwk  = max(qwk_logit, qwk_prob, qwk_tta_logit, qwk_tta_prob)
    best_name = {
        qwk_logit:     "Logit Averaging",
        qwk_prob:      "Probability Averaging",
        qwk_tta_logit: "Logit Averaging + TTA",
        qwk_tta_prob:  "Probability Averaging + TTA",
    }[best_qwk]
    print(f"\n→ Best method: {best_name} (QWK = {best_qwk:.4f})")

    # Note about missing grades
    unique_labels = np.unique(all_labels)
    if len(unique_labels) < 5:
        missing_grades = set([0, 1, 2, 3, 4]) - set(unique_labels)
        print(f"\nNote: grades {missing_grades} absent — CM rows/cols will be zeros.")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()

"""
Ensemble Evaluation Script for IDRiD Disease Grading Test Set
Evaluates ensemble of baseline and attention-consistency models with:
  - Standard fusion (no TTA): logit averaging, probability averaging
  - TTA fusion (horizontal flip): logit averaging + TTA, probability averaging + TTA
"""

import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm

from model import DRGradingModel
from preprocess import circular_crop, ben_graham_normalize, clahe_enhance

# ── Paths ─────────────────────────────────────────────────────────────────────
BASELINE_CKPT = "E:/DR_PROJECT/dr_baseline/outputs/checkpoints/best_model.pth"
ATTN_CKPT     = "E:/DR_PROJECT/dr_extended/outputs/checkpoints/best_model.pth"
LABELS_CSV    = "E:/DR_PROJECT/data/data1/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
IMG_DIR       = "E:/DR_PROJECT/data/data1/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
OUTPUT_DIR    = "E:/DR_PROJECT/dr_extended/outputs"


def load_model(checkpoint_path, num_classes, device):
    model = DRGradingModel(num_classes=num_classes).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def preprocess_image(image_path):
    """Return (x0, x3) tensors for a single image."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img0 = circular_crop(img, target_size=224)
    img0 = ben_graham_normalize(img0)
    x0   = torch.from_numpy(img0).permute(2, 0, 1).float() / 255.0
    x0   = x0.unsqueeze(0)

    img3 = circular_crop(img, target_size=300)
    img3 = clahe_enhance(img3)
    x3   = torch.from_numpy(img3).permute(2, 0, 1).float() / 255.0
    x3   = x3.unsqueeze(0)

    return x0, x3


def get_logits(model, x0, x3):
    """Forward pass; handle models that return (logits, fused) or just logits."""
    out = model(x0, x3)
    return out[0] if isinstance(out, tuple) else out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"\nLoading baseline  : {BASELINE_CKPT}")
    model_base = load_model(BASELINE_CKPT, num_classes=5, device=device)
    print(f"Loading attention : {ATTN_CKPT}")
    model_attn = load_model(ATTN_CKPT,     num_classes=5, device=device)

    # ── Load labels ───────────────────────────────────────────────────────────
    df      = pd.read_csv(LABELS_CSV)
    img_dir = Path(IMG_DIR)
    print(f"\nIDRiD test set: {len(df)} images")

    all_labels      = []
    preds_logit     = []
    preds_prob      = []
    preds_tta_logit = []
    preds_tta_prob  = []

    print("\nEvaluating (standard + TTA)...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_name = row["Image name"]
            label    = int(row["Retinopathy grade"])

            img_path = img_dir / f"{img_name}.jpg"
            if not img_path.exists():
                print(f"  Warning: not found — {img_path.name}")
                continue

            x0, x3 = preprocess_image(img_path)
            x0, x3 = x0.to(device), x3.to(device)

            # ── Original pass ─────────────────────────────────────────────
            logits_base_orig = get_logits(model_base, x0, x3)
            logits_attn_orig = get_logits(model_attn, x0, x3)

            # ── Flipped pass ──────────────────────────────────────────────
            x0_flip = torch.flip(x0, [-1])
            x3_flip = torch.flip(x3, [-1])
            logits_base_flip = get_logits(model_base, x0_flip, x3_flip)
            logits_attn_flip = get_logits(model_attn, x0_flip, x3_flip)

            # ── Standard: logit averaging (no TTA) ───────────────────────
            avg_l = (logits_base_orig + logits_attn_orig) / 2.0
            preds_logit.append((torch.sigmoid(avg_l) > 0.5).sum(dim=1).item())

            # ── Standard: probability averaging (no TTA) ─────────────────
            avg_p = (torch.sigmoid(logits_base_orig) + torch.sigmoid(logits_attn_orig)) / 2.0
            preds_prob.append((avg_p > 0.5).sum(dim=1).item())

            # ── TTA: logit averaging ──────────────────────────────────────
            avg_l_tta = (logits_base_orig + logits_attn_orig +
                         logits_base_flip + logits_attn_flip) / 4.0
            preds_tta_logit.append((torch.sigmoid(avg_l_tta) > 0.5).sum(dim=1).item())

            # ── TTA: probability averaging ────────────────────────────────
            avg_p_tta = (torch.sigmoid(logits_base_orig) +
                         torch.sigmoid(logits_attn_orig) +
                         torch.sigmoid(logits_base_flip) +
                         torch.sigmoid(logits_attn_flip)) / 4.0
            preds_tta_prob.append((avg_p_tta > 0.5).sum(dim=1).item())

            all_labels.append(label)

    all_labels      = np.array(all_labels)
    preds_logit     = np.array(preds_logit)
    preds_prob      = np.array(preds_prob)
    preds_tta_logit = np.array(preds_tta_logit)
    preds_tta_prob  = np.array(preds_tta_prob)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def metrics(labels, preds):
        qwk = cohen_kappa_score(labels, preds, weights="quadratic")
        cm  = confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])
        return qwk, cm

    qwk_logit,     cm_logit     = metrics(all_labels, preds_logit)
    qwk_prob,      cm_prob      = metrics(all_labels, preds_prob)
    qwk_tta_logit, cm_tta_logit = metrics(all_labels, preds_tta_logit)
    qwk_tta_prob,  cm_tta_prob  = metrics(all_labels, preds_tta_prob)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("RESULTS — IDRiD Disease Grading (Standard Ensemble)")
    print("="*65)
    print(f"\n1. Logit Averaging       QWK: {qwk_logit:.4f}")
    print(f"   Confusion Matrix:\n{cm_logit}")
    print(f"\n2. Probability Averaging QWK: {qwk_prob:.4f}")
    print(f"   Confusion Matrix:\n{cm_prob}")

    print("\n" + "="*65)
    print("RESULTS — IDRiD Disease Grading (TTA — horizontal flip)")
    print("="*65)
    print(f"\n3. Logit Averaging + TTA       QWK: {qwk_tta_logit:.4f}")
    print(f"   Confusion Matrix:\n{cm_tta_logit}")
    print(f"\n4. Probability Averaging + TTA QWK: {qwk_tta_prob:.4f}")
    print(f"   Confusion Matrix:\n{cm_tta_prob}")

    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    print(f"Logit Averaging          : {qwk_logit:.4f}")
    print(f"Probability Averaging    : {qwk_prob:.4f}")
    print(f"Logit Averaging + TTA    : {qwk_tta_logit:.4f}  ← TTA")
    print(f"Probability Avg  + TTA   : {qwk_tta_prob:.4f}  ← TTA")

    best = max(qwk_logit, qwk_prob, qwk_tta_logit, qwk_tta_prob)
    name = {qwk_logit: "Logit Averaging", qwk_prob: "Probability Averaging",
            qwk_tta_logit: "Logit Averaging + TTA",
            qwk_tta_prob:  "Probability Averaging + TTA"}[best]
    print(f"\n→ Best: {name} (QWK = {best:.4f})")

    unique = np.unique(all_labels)
    if len(unique) < 5:
        missing = set(range(5)) - set(unique.tolist())
        print(f"\nNote: grades {missing} absent — CM rows/cols will be zeros.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "idrid_ensemble_logit_cm.npy",     cm_logit)
    np.save(out / "idrid_ensemble_prob_cm.npy",      cm_prob)
    np.save(out / "idrid_ensemble_tta_logit_cm.npy", cm_tta_logit)
    np.save(out / "idrid_ensemble_tta_prob_cm.npy",  cm_tta_prob)
    print(f"\n✓ Confusion matrices saved to {out}/")
    print("="*65)


if __name__ == "__main__":
    main()

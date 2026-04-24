"""
evaluate_idrid.py — Evaluate baseline model on IDRiD Disease Grading test set.

Usage:
    python evaluate_idrid.py
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
CHECKPOINT   = "E:/DR_PROJECT/dr_baseline/outputs/checkpoints/best_model.pth"
LABELS_CSV   = "E:/DR_PROJECT/data/data1/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
IMG_DIR      = "E:/DR_PROJECT/data/data1/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
OUTPUT_DIR   = "E:/DR_PROJECT/dr_baseline/outputs"


def preprocess_image(image_path):
    """Preprocess a single image into (x0, x3) tensors."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img0 = circular_crop(img, target_size=224)
    img0 = ben_graham_normalize(img0)
    x0 = torch.from_numpy(img0).permute(2, 0, 1).float() / 255.0
    x0 = x0.unsqueeze(0)

    img3 = circular_crop(img, target_size=300)
    img3 = clahe_enhance(img3)
    x3 = torch.from_numpy(img3).permute(2, 0, 1).float() / 255.0
    x3 = x3.unsqueeze(0)

    return x0, x3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    model = DRGradingModel(num_classes=5).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if "best_val_qwk" in ckpt:
        print(f"Checkpoint best val QWK: {ckpt['best_val_qwk']:.4f}")

    # Load labels
    df = pd.read_csv(LABELS_CSV)
    img_dir = Path(IMG_DIR)

    all_labels, all_preds = [], []

    print(f"\nEvaluating on IDRiD Disease Grading test set ({len(df)} images)...")
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

            logits = model(x0, x3)          # (1, 4)
            probs  = torch.sigmoid(logits)  # (1, 4)
            pred   = (probs > 0.5).sum(dim=1).item()

            all_labels.append(label)
            all_preds.append(pred)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    # Metrics
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    cm  = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])

    print("\n" + "="*60)
    print("RESULTS — IDRiD Disease Grading Test Set (Baseline)")
    print("="*60)
    print(f"QWK  : {qwk:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Missing grades note
    unique = np.unique(all_labels)
    if len(unique) < 5:
        missing = set(range(5)) - set(unique.tolist())
        print(f"\nNote: grades {missing} absent from test set (zero rows/cols in CM).")

    # Save
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "idrid_baseline_cm.npy", cm)
    with open(out / "idrid_baseline_qwk.txt", "w") as f:
        f.write(f"{qwk:.6f}\n")
    print(f"\n✓ Saved to {out}/")
    print("="*60)


if __name__ == "__main__":
    main()

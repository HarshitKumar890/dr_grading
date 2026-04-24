# DR Baseline: Dual-Branch EfficientNet + Attention Fusion + CORAL

This folder implements the baseline model specification for diabetic retinopathy grading:
- Branch 0: EfficientNet-B0 on Ben-Graham normalized input (224x224)
- Branch 3: EfficientNet-B3 on CLAHE-enhanced input (300x300)
- Attention gate fusion
- CORAL ordinal regression head (5 grades -> 4 logits)

## Files
- `preprocess.py`: circular crop, Ben-Graham normalization, CLAHE, histogram matching
- `augmentations.py`: train-time Albumentations policy
- `dataset.py`: APTOS/Messidor split logic and dual-branch dataset
- `model.py`: dual-backbone fusion model
- `losses.py`: CORAL loss helpers
- `train.py`: training loop, AMP, early stopping, best checkpoint
- `evaluate.py`: validation/test evaluation, QWK, confusion matrix
- `utils.py`: metrics, plotting, checkpoint helpers
- `config.yaml`: all key paths and hyperparameters

## Dataset Expectations
Update `config.yaml` paths as needed.

APTOS labels CSV must contain:
- `id_code`
- `diagnosis` in `{0,1,2,3,4}`

Messidor labels CSV must contain:
- `image_filename`
- `grade` in `{0,1,2,3,4}`
- optional `adjudicated` (if present, only True rows used)
- optional `gradable` (if present, only True rows used)

## Splits Implemented
- APTOS: stratified 5-fold (`n_splits=5`, `random_state=42`), first fold used as validation
- Messidor-2: stratified random split 70% train / 30% test on filtered adjudicated+gradable subset
- Final train: `APTOS_train + Messidor_train`
- Validation: `APTOS_val`
- External test: `Messidor_test`

## Install
From your activated venv:

```powershell
Set-Location e:\DR_PROJECT\dr_baseline
pip install -r requirements.txt
```

## Train
```powershell
Set-Location e:\DR_PROJECT\dr_baseline
python train.py --config config.yaml
```

What training does:
- verifies `torch.cuda.is_available()` and binds device to `cuda:0`
- uses `WeightedRandomSampler` for class imbalance
- uses mixed precision AMP
- monitors validation QWK
- saves best model at `outputs/checkpoints/best_model.pth`
- early stops with patience from config

## Evaluate
```powershell
Set-Location e:\DR_PROJECT\dr_baseline
python evaluate.py --config config.yaml --checkpoint outputs/checkpoints/best_model.pth --split both
```

Outputs:
- QWK for selected split(s)
- confusion matrices saved as `.npy`
- `val_qwk.txt` and `test_qwk.txt`

## Important Notes
- Histogram matching is applied to Messidor samples only.
- Augmentation is applied only in training mode.
- Validation/test do not use augmentation.
- If your Messidor image directory is nested by center folders, either flatten it or provide full relative filenames in `labels.csv`.

"""
train.py — Extended training script with DANN + Attention Consistency.

Two training modes controlled by config.yaml:

  domain_adaptation.enabled = true  →  DANN mode
    - source_loader (APTOS, labelled) + target_loader (Messidor-2, unlabelled)
    - loss = coral_loss + lambda_grl * domain_loss + lambda_att * att_loss

  domain_adaptation.enabled = false  →  baseline mode
    - single combined train_loader (APTOS + Messidor-2, all labelled)
    - loss = coral_loss + lambda_att * att_loss   (domain_loss = 0)

Attention consistency (Extension 2) is applied in both modes when
attention_consistency.enabled = true.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import torch
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_preprocessed import (
    build_dataloaders_preprocessed,
    build_source_target_dataloaders,
)
from losses import coral_loss
from model import DRGradingModel
from utils import compute_qwk_from_logits, plot_training_curves, save_checkpoint, set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Extended DR grading model")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_lambda_grl(epoch: int, cfg: dict) -> float:
    """Linear ramp from lambda_start to lambda_max over ramp_epochs."""
    start   = cfg["lambda_start"]
    max_val = cfg["lambda_max"]
    ramp    = cfg["ramp_epochs"]
    if epoch < ramp:
        return start + (max_val - start) * (epoch / ramp)
    return max_val


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_epoch(
    model: DRGradingModel,
    loader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    all_logits, all_labels, losses = [], [], []

    with torch.no_grad():
        for x0, x3, y in tqdm(loader, desc="Validation", leave=False):
            x0 = x0.to(device, non_blocking=True)
            x3 = x3.to(device, non_blocking=True)
            y  = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits, _ = model(x0, x3)
                loss = coral_loss(logits, y, num_classes=5)

            losses.append(loss.item())
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    qwk = compute_qwk_from_logits(logits_cat, labels_cat)
    return float(sum(losses) / max(1, len(losses))), float(qwk)


# ─────────────────────────────────────────────────────────────────────────────
# Attention consistency loss helper
# ─────────────────────────────────────────────────────────────────────────────

def attention_consistency_loss(
    model: DRGradingModel,
    x0: torch.Tensor,
    x3: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """
    Compute MSE between flipped-original attention and attention-of-flipped.
    att_orig is in the computation graph; att_flip_target is detached.
    """
    # att_orig: computed from feature maps already filled by the preceding forward()
    att_orig = DRGradingModel.compute_fused_attention(
        model.b0_feat_map, model.b3_feat_map
    )

    x0_flip = torch.flip(x0, dims=[-1])
    x3_flip = torch.flip(x3, dims=[-1])

    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=use_amp):
            model(x0_flip, x3_flip)   # fills hooks with flipped feature maps
        att_flip_target = DRGradingModel.compute_fused_attention(
            model.b0_feat_map, model.b3_feat_map
        )

    # Flip att_orig spatially and compare to att_flip_target
    return F.mse_loss(torch.flip(att_orig, dims=[-1]), att_flip_target)


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch — DANN mode
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_dann(
    model: DRGradingModel,
    source_loader,
    target_loader,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    config: dict,
) -> float:
    model.train()
    da_cfg  = config["domain_adaptation"]
    att_cfg = config.get("attention_consistency", {})
    use_att = bool(att_cfg.get("enabled", False))
    lambda_att         = float(att_cfg.get("lambda_att", 0.1))
    lambda_grl         = get_lambda_grl(epoch, da_cfg)
    lambda_target_coral = float(da_cfg.get("lambda_target_coral", 0.2))

    running_loss = 0.0
    target_iter  = iter(target_loader)

    pbar = tqdm(source_loader, desc=f"Epoch {epoch+1} [DANN]")
    for src_x0, src_x3, src_y in pbar:
        # ── Fetch target batch (cycle if exhausted) ───────────────────────────
        try:
            tgt_x0, tgt_x3, tgt_y = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            tgt_x0, tgt_x3, tgt_y = next(target_iter)

        src_x0 = src_x0.to(device, non_blocking=True)
        src_x3 = src_x3.to(device, non_blocking=True)
        src_y  = src_y.to(device, non_blocking=True)
        tgt_x0 = tgt_x0.to(device, non_blocking=True)
        tgt_x3 = tgt_x3.to(device, non_blocking=True)
        tgt_y  = tgt_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            # ── Source forward ────────────────────────────────────────────────
            logits_s, fused_s = model(src_x0, src_x3)
            loss_coral_src = coral_loss(logits_s, src_y, num_classes=5)

            # ── Target forward (semi-supervised: also apply CORAL with lower weight) ──
            logits_t, fused_t = model(tgt_x0, tgt_x3)
            loss_coral_tgt = coral_loss(logits_t, tgt_y, num_classes=5)

            # ── Domain adversarial loss ───────────────────────────────────────
            # Source label = 0, target label = 1
            dom_s = model.forward_domain(fused_s, lambda_grl)
            dom_t = model.forward_domain(fused_t, lambda_grl)
            dom_logits  = torch.cat([dom_s, dom_t], dim=0)
            dom_targets = torch.cat([
                torch.zeros_like(dom_s),
                torch.ones_like(dom_t),
            ], dim=0)
            loss_domain = F.binary_cross_entropy_with_logits(dom_logits, dom_targets)

            # ── Attention consistency (source only, if enabled) ───────────────
            if use_att:
                loss_att = attention_consistency_loss(
                    model, src_x0, src_x3, device, use_amp
                )
            else:
                loss_att = torch.tensor(0.0, device=device)

            total_loss = (
                loss_coral_src
                + lambda_target_coral * loss_coral_tgt
                + lambda_grl * loss_domain
                + lambda_att * loss_att
            )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
        pbar.set_postfix({
            "coral_s": f"{loss_coral_src.item():.4f}",
            "coral_t": f"{loss_coral_tgt.item():.4f}",
            "dom":     f"{loss_domain.item():.4f}",
            "λ_grl":   f"{lambda_grl:.4f}",
        })

    return running_loss / max(1, len(source_loader))


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch — baseline mode (combined loader)
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_baseline(
    model: DRGradingModel,
    train_loader,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    config: dict,
) -> float:
    model.train()
    att_cfg    = config.get("attention_consistency", {})
    use_att    = bool(att_cfg.get("enabled", False))
    lambda_att = float(att_cfg.get("lambda_att", 0.1))

    running_loss = 0.0
    total_epochs = config["train"]["epochs"]

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for x0, x3, y in pbar:
        x0 = x0.to(device, non_blocking=True)
        x3 = x3.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(x0, x3)
            loss_coral = coral_loss(logits, y, num_classes=5)

            if use_att:
                loss_att = attention_consistency_loss(
                    model, x0, x3, device, use_amp
                )
            else:
                loss_att = torch.tensor(0.0, device=device)

            total_loss = loss_coral + lambda_att * loss_att

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
        pbar.set_postfix({
            "coral": f"{loss_coral.item():.4f}",
            "att":   f"{loss_att.item():.4f}",
        })

    return running_loss / max(1, len(train_loader))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    args   = parse_args()
    config = load_config(args.config)

    set_seed(config["train"]["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU required.")

    device = torch.device("cuda:0")
    print(f"torch.cuda.is_available(): True")
    print(f"Using device: {device} | GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

    # ── Select training mode ──────────────────────────────────────────────────
    da_cfg         = config.get("domain_adaptation", {})
    use_domain_adapt = bool(da_cfg.get("enabled", False))

    if use_domain_adapt:
        print("Training mode: DANN (domain adversarial adaptation)")
        source_loader, target_loader, val_loader, test_loader = (
            build_source_target_dataloaders(config)
        )
    else:
        print("Training mode: baseline (combined loader)")
        train_loader, val_loader, _ = build_dataloaders_preprocessed(config)

    # ── Model, optimiser, scaler ──────────────────────────────────────────────
    model = DRGradingModel(
        num_classes=5, fusion_dim=config["model"]["fusion_dim"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    use_amp = bool(config["train"]["use_amp"])
    scaler  = GradScaler("cuda", enabled=use_amp)

    # ── Output dirs ───────────────────────────────────────────────────────────
    output_dir = Path(config["output"]["output_dir"])
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_qwk            = -1.0
    epochs_no_improve   = 0
    patience            = config["train"]["early_stopping_patience"]
    history             = {"train_loss": [], "val_qwk": []}

    for epoch in range(config["train"]["epochs"]):
        torch.cuda.empty_cache()

        if use_domain_adapt:
            train_loss = train_epoch_dann(
                model, source_loader, target_loader,
                optimizer, scaler, device, use_amp, epoch, config,
            )
        else:
            train_loss = train_epoch_baseline(
                model, train_loader,
                optimizer, scaler, device, use_amp, epoch, config,
            )

        val_loss, val_qwk = evaluate_epoch(model, val_loader, device, use_amp)

        history["train_loss"].append(train_loss)
        history["val_qwk"].append(val_qwk)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("QWK/val",    val_qwk,    epoch)

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_qwk={val_qwk:.4f}"
        )

        if val_qwk > best_qwk:
            best_qwk        = val_qwk
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "epoch":               epoch + 1,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_qwk":        best_qwk,
                    "config":              config,
                },
                str(ckpt_dir / "best_model.pth"),
            )
            print(f"✓ Saved best checkpoint  Val QWK={best_qwk:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience}).")
            break

    writer.close()
    plot_training_curves(history, str(output_dir))
    print(f"\nTraining complete. Best validation QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    train()

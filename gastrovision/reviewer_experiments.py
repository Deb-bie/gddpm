"""
reviewer_experiments.py
========================
Additional experiments addressing reviewer feedback:

1. run_hybrid_ablation()
   Trains and evaluates the three-stage HybridV2 ablation:
     - CNN-only  : efficientnetv2_rw_s (already trained, loaded from checkpoint)
     - CNN+proj  : hybrid_cnn_proj_only (new, trained here)
     - Full hybrid: hybrid_cnn_transformer_v2 (already trained, loaded from checkpoint)

2. run_synthetic_only()
   Trains a classifier on ONLY synthetic images for rare classes + real images for
   common classes. Tests whether SD-generated images carry any discriminative signal
   independent of the real-data dilution effect.

Results are saved to RESULTS_DIR/reviewer_experiments.json.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import f1_score, accuracy_score

import config as _config
from config import args, DEVICE, RESULTS_DIR, CKPT_DIR, SPLITS_DIR, SYNTH_DIR
from dataset import GastroVisionDataset
from models import get_model, load_checkpoint
from train import train_model
from evaluate import evaluate_split


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _rare_f1(f1_array, rare_classes=None):
    rc = rare_classes or _config.RARE_CLASSES
    # exclude ultra-rare
    ultra = set(getattr(_config, "ULTRA_RARE", []))
    rc = [c for c in rc if c not in ultra]
    return float(np.mean([f1_array[c] for c in rc]))


# ─────────────────────────────────────────────────────────────────────────────
# 1. HybridV2 ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_ablation():
    """
    Three-stage ablation isolating the Transformer encoder's contribution:
      CNN-only   → CNN + proj (mean-pool) → Full hybrid (Transformer)
    CNN-only uses the already-trained EfficientNetV2-S checkpoint.
    CNN+proj is trained fresh here with the same protocol.
    Full hybrid uses the already-trained checkpoint.
    """
    print("\n" + "="*60)
    print("HYBRID ABLATION (reviewer experiment)")
    print("="*60)

    out = {}

    # Stage A: CNN-only — load existing EfficientNetV2-S checkpoint
    print("\n[A] CNN-only (EfficientNetV2-S checkpoint)")
    try:
        m = load_checkpoint("efficientnetv2_rw_s", suffix="")
        res = evaluate_split(m, split="test")
        out["cnn_only"] = {
            "rare_f1": _rare_f1(res["f1"]),
            "macro_f1": float(np.mean(res["f1"])),
            "acc": res["acc"],
        }
        print(f"  rare_f1={out['cnn_only']['rare_f1']:.4f}  macro_f1={out['cnn_only']['macro_f1']:.4f}")
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  CNN-only load failed: {e}")
        out["cnn_only"] = {"error": str(e)}

    # Stage B: CNN + proj (mean-pool, no Transformer) — train fresh
    print("\n[B] CNN + proj only (hybrid_cnn_proj_only) — training...")
    try:
        model_name = "hybrid_cnn_proj_only"
        m = get_model(model_name)
        train_model(m, model_name=model_name, suffix="", augmented=False)
        res = evaluate_split(m, split="test")
        out["cnn_proj_only"] = {
            "rare_f1": _rare_f1(res["f1"]),
            "macro_f1": float(np.mean(res["f1"])),
            "acc": res["acc"],
        }
        print(f"  rare_f1={out['cnn_proj_only']['rare_f1']:.4f}  macro_f1={out['cnn_proj_only']['macro_f1']:.4f}")
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  CNN+proj training failed: {e}")
        out["cnn_proj_only"] = {"error": str(e)}

    # Stage C: Full hybrid — load existing checkpoint
    print("\n[C] Full hybrid (HybridCNNTransformerV2 checkpoint)")
    try:
        m = load_checkpoint("hybrid_cnn_transformer_v2", suffix="")
        res = evaluate_split(m, split="test")
        out["full_hybrid"] = {
            "rare_f1": _rare_f1(res["f1"]),
            "macro_f1": float(np.mean(res["f1"])),
            "acc": res["acc"],
        }
        print(f"  rare_f1={out['full_hybrid']['rare_f1']:.4f}  macro_f1={out['full_hybrid']['macro_f1']:.4f}")
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Full hybrid load failed: {e}")
        out["full_hybrid"] = {"error": str(e)}

    print("\nHybrid ablation summary:")
    for k, v in out.items():
        if "rare_f1" in v:
            print(f"  {k:<20} rare_f1={v['rare_f1']:.4f}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic-only experiment
# ─────────────────────────────────────────────────────────────────────────────

def _build_synth_only_dataset():
    """
    For rare classes: use ONLY synthetic images (from SYNTH_DIR).
    For common classes: use real training images as normal.
    This dataset tests whether synthetic images carry any discriminative signal.
    """
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    import os

    ultra = set(getattr(_config, "ULTRA_RARE", []))
    rare = [c for c in _config.RARE_CLASSES if c not in ultra]

    # Real dataset for common classes only
    train_ds_full = GastroVisionDataset(
        split_file=SPLITS_DIR / "train.txt",
        root=_config.IMAGE_ROOT_DIR,
        augment=True,
    )

    # Filter to common classes only
    common_indices = [i for i, (_, lbl) in enumerate(train_ds_full.samples) if lbl not in rare]
    real_common_ds = Subset(train_ds_full, common_indices)

    # Synthetic images for rare classes
    synth_datasets = []
    for cls_idx in rare:
        synth_cls_dir = SYNTH_DIR / f"class_{cls_idx:02d}"
        if not synth_cls_dir.exists():
            print(f"  Warning: synthetic dir not found for class {cls_idx}: {synth_cls_dir}")
            continue
        imgs = list(synth_cls_dir.glob("*.png")) + list(synth_cls_dir.glob("*.jpg"))
        if not imgs:
            print(f"  Warning: no synthetic images for class {cls_idx}")
            continue

        transform = T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        class SynthClassDataset(torch.utils.data.Dataset):
            def __init__(self, paths, label, tfm):
                self.paths = paths
                self.label = label
                self.tfm = tfm
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                from PIL import Image as PILImage
                img = PILImage.open(self.paths[i]).convert("RGB")
                return self.tfm(img), self.label

        synth_datasets.append(SynthClassDataset(imgs, cls_idx, transform))
        print(f"  Class {cls_idx}: {len(imgs)} synthetic images")

    if not synth_datasets:
        raise RuntimeError("No synthetic images found. Run SD generation first (S3 strategy).")

    combined = ConcatDataset([real_common_ds] + synth_datasets)
    print(f"  Synthetic-only dataset: {len(real_common_ds)} real (common) + "
          f"{sum(len(d) for d in synth_datasets)} synthetic (rare) = {len(combined)} total")
    return combined


def run_synthetic_only():
    """
    Train EfficientNetV2-S on:
      - Real images for common classes
      - ONLY synthetic images for rare classes (no real rare-class images)

    Compare to:
      - S1 (real only, including real rare images): rare-F1 ~0.584
      - S3 (real + synthetic): rare-F1 ~0.540

    If synthetic-only rare-F1 ≈ S3, dilution is the dominant failure mode.
    If synthetic-only rare-F1 ≈ 0 (random), texture mismatch is dominant.
    """
    print("\n" + "="*60)
    print("SYNTHETIC-ONLY EXPERIMENT (reviewer experiment)")
    print("="*60)

    out = {}

    try:
        # Build synthetic-only training set
        synth_only_ds = _build_synth_only_dataset()
        train_loader = DataLoader(
            synth_only_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Validation and test loaders (real images, standard)
        val_ds = GastroVisionDataset(
            split_file=SPLITS_DIR / "val.txt",
            root=_config.IMAGE_ROOT_DIR,
            augment=False,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Train EfficientNetV2-S with same protocol as main experiments
        model_name = "efficientnetv2_rw_s"
        model = get_model(model_name)

        from train import _train_phase
        from losses import FocalLoss
        criterion = FocalLoss(gamma=args.gamma, num_classes=_config.NUM_CLASSES)

        # Stage 1: freeze backbone, train head
        model.freeze_backbones() if hasattr(model, "freeze_backbones") else None
        _train_phase(model, train_loader, val_loader, criterion,
                     epochs=args.freeze_epochs, phase="freeze",
                     model_name=model_name, suffix="_synth_only")

        # Stage 2: unfreeze all
        model.unfreeze_all() if hasattr(model, "unfreeze_all") else None
        _train_phase(model, train_loader, val_loader, criterion,
                     epochs=args.fine_tune_epochs, phase="finetune",
                     model_name=model_name, suffix="_synth_only")

        # Evaluate on test set
        res = evaluate_split(model, split="test")
        ultra = set(getattr(_config, "ULTRA_RARE", []))
        rare = [c for c in _config.RARE_CLASSES if c not in ultra]

        out["synth_only"] = {
            "rare_f1": _rare_f1(res["f1"]),
            "macro_f1": float(np.mean(res["f1"])),
            "acc": res["acc"],
            "per_rare_f1": {c: float(res["f1"][c]) for c in rare},
        }
        print(f"\nSynthetic-only rare_f1: {out['synth_only']['rare_f1']:.4f}")
        print("  Per-class F1:")
        for c, f in out["synth_only"]["per_rare_f1"].items():
            print(f"    class {c}: {f:.4f}")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Synthetic-only experiment failed: {e}")
        import traceback; traceback.print_exc()
        out["synth_only"] = {"error": str(e)}

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all_reviewer_experiments():
    results = {}

    print("\n>>> Running hybrid ablation...")
    results["hybrid_ablation"] = run_hybrid_ablation()

    print("\n>>> Running synthetic-only experiment...")
    results["synthetic_only"] = run_synthetic_only()

    out_path = RESULTS_DIR / "reviewer_experiments.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReviewer experiments saved to {out_path}")

    return results

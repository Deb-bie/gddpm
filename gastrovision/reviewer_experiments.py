"""
reviewer_experiments.py
========================
Additional experiments addressing reviewer feedback:

1. run_hybrid_ablation()
   Three-stage ablation isolating the Transformer encoder's contribution:
     CNN-only   (EfficientNetV2-S, existing checkpoint)
     CNN+proj   (HybridCNNProjOnly, trained here)
     Full hybrid (HybridCNNTransformerV2, existing checkpoint)

2. run_synthetic_only()
   Trains EfficientNetV2-S on ONLY synthetic images for rare classes (+ real
   images for common classes). Tests whether SD images carry any discriminative
   signal independent of the real-data dilution effect.

Results saved to RESULTS_DIR/reviewer_experiments.json.
"""

import json
import csv
import tempfile
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image as PILImage
from torch.utils.data import DataLoader, ConcatDataset, Dataset

import config as _config
from config import args, DEVICE, RESULTS_DIR, CKPT_DIR, SPLITS_DIR, SYNTH_DIR, OUTPUT_DIR
from dataset import GastroVisionDataset
from models import get_model, load_checkpoint
from train import train_classifier
from evaluate import evaluate_split


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _rare_f1(f1_array):
    ultra = set(getattr(_config, "ULTRA_RARE", []))
    rc = [c for c in _config.RARE_CLASSES if c not in ultra]
    return float(np.mean([f1_array[c] for c in rc]))


# ─────────────────────────────────────────────────────────────────────────────
# 1. HybridV2 internal ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_ablation():
    """
    CNN-only → CNN+proj → Full Hybrid  (three-stage ablation).
    CNN-only and Full Hybrid load existing checkpoints.
    CNN+proj trains a fresh HybridCNNProjOnly model.
    """
    print("\n" + "="*60)
    print("HYBRID ABLATION (reviewer experiment)")
    print("="*60)

    train_csv = SPLITS_DIR / args.train_csv
    val_csv   = SPLITS_DIR / args.val_csv
    out = {}

    # ── A: CNN-only ──────────────────────────────────────────────────────────
    print("\n[A] CNN-only: loading EfficientNetV2-S checkpoint")
    try:
        m = load_checkpoint("efficientnetv2_rw_s", suffix="")
        res = evaluate_split(m, split="test")
        out["cnn_only"] = {
            "rare_f1":  _rare_f1(res["f1"]),
            "macro_f1": float(np.mean(res["f1"])),
            "acc":      res["acc"],
        }
        print(f"  rare_f1={out['cnn_only']['rare_f1']:.4f}  "
              f"macro_f1={out['cnn_only']['macro_f1']:.4f}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  CNN-only failed: {e}")
        out["cnn_only"] = {"error": str(e)}

    # ── B: CNN + proj (no Transformer) ───────────────────────────────────────
    print("\n[B] CNN+proj: training HybridCNNProjOnly...")
    try:
        train_classifier("hybrid_cnn_proj_only", train_csv, val_csv, augmented=False)
        m = load_checkpoint("hybrid_cnn_proj_only", suffix="")
        res = evaluate_split(m, split="test")
        out["cnn_proj_only"] = {
            "rare_f1":  _rare_f1(res["f1"]),
            "macro_f1": float(np.mean(res["f1"])),
            "acc":      res["acc"],
        }
        print(f"  rare_f1={out['cnn_proj_only']['rare_f1']:.4f}  "
              f"macro_f1={out['cnn_proj_only']['macro_f1']:.4f}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  CNN+proj failed: {e}")
        out["cnn_proj_only"] = {"error": str(e)}

    # ── C: Full hybrid ────────────────────────────────────────────────────────
    print("\n[C] Full hybrid: loading HybridCNNTransformerV2 checkpoint")
    try:
        m = load_checkpoint("hybrid_cnn_transformer_v2", suffix="")
        res = evaluate_split(m, split="test")
        out["full_hybrid"] = {
            "rare_f1":  _rare_f1(res["f1"]),
            "macro_f1": float(np.mean(res["f1"])),
            "acc":      res["acc"],
        }
        print(f"  rare_f1={out['full_hybrid']['rare_f1']:.4f}  "
              f"macro_f1={out['full_hybrid']['macro_f1']:.4f}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Full hybrid failed: {e}")
        out["full_hybrid"] = {"error": str(e)}

    print("\nHybrid ablation summary:")
    for k, v in out.items():
        rf = v.get("rare_f1", "ERR")
        print(f"  {k:<20} rare_f1={rf}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic-only experiment
# ─────────────────────────────────────────────────────────────────────────────

class _SynthClassDataset(Dataset):
    """Loads synthetic images from a folder for a single class label."""
    _transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, paths, label):
        self.paths = paths
        self.label = label

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = PILImage.open(self.paths[i]).convert("RGB")
        return self._transform(img), self.label


def _build_synth_only_csv(tmp_dir: Path) -> Path:
    """
    Write a CSV for GastroVisionDataset that:
      - includes ALL real training images for common classes
      - includes NO real images for rare classes
      - includes synthetic images for rare classes via a separate dataset (returned separately)

    Returns path to the common-only CSV.
    """
    import pandas as pd
    train_csv = SPLITS_DIR / args.train_csv
    df = pd.read_csv(train_csv)

    ultra = set(getattr(_config, "ULTRA_RARE", []))
    rare  = set(c for c in _config.RARE_CLASSES if c not in ultra)

    # Keep only common-class rows
    common_df = df[~df["label"].isin(rare)].copy()
    out_csv = tmp_dir / "synth_only_common.csv"
    common_df.to_csv(out_csv, index=False)
    print(f"  Common-class real rows: {len(common_df)}")
    return out_csv


def run_synthetic_only():
    """
    Train EfficientNetV2-S on:
      - Real images for common classes (from train CSV, rare rows removed)
      - ONLY synthetic images for rare classes (from SYNTH_DIR)

    Interpretation guide:
      If synthetic-only rare-F1 is near zero → synthetic images have no class
        signal; texture mismatch is dominant failure mode.
      If synthetic-only rare-F1 ≈ S3 (0.540) → dilution is the dominant
        failure mode (real+synth mixing hurts, but synth alone is no worse).
    """
    print("\n" + "="*60)
    print("SYNTHETIC-ONLY EXPERIMENT (reviewer experiment)")
    print("="*60)

    out = {}

    try:
        import tempfile, pandas as pd

        ultra = set(getattr(_config, "ULTRA_RARE", []))
        rare  = sorted(c for c in _config.RARE_CLASSES if c not in ultra)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # ── Build common-only CSV ─────────────────────────────────────────
            common_csv = _build_synth_only_csv(tmp_path)

            # ── Load common-class real dataset ────────────────────────────────
            real_ds = GastroVisionDataset(common_csv, split="train", heavy=False)

            # ── Build synthetic datasets per rare class ────────────────────────
            synth_datasets = []
            for cls_idx in rare:
                # Try multiple possible synth folder names
                candidates = [
                    SYNTH_DIR / f"class_{cls_idx:02d}",
                    OUTPUT_DIR / "synthetic" / f"class_{cls_idx:02d}",
                    OUTPUT_DIR / f"class_{cls_idx:02d}",
                ]
                synth_dir = next((d for d in candidates if d.exists()), None)
                if synth_dir is None:
                    print(f"  Warning: no synth dir for class {cls_idx}")
                    continue
                imgs = sorted(list(synth_dir.glob("*.png")) + list(synth_dir.glob("*.jpg")))
                if not imgs:
                    print(f"  Warning: no images in {synth_dir}")
                    continue
                synth_datasets.append(_SynthClassDataset(imgs, cls_idx))
                print(f"  Class {cls_idx}: {len(imgs)} synthetic images from {synth_dir.name}")

            if not synth_datasets:
                raise RuntimeError(
                    "No synthetic images found. Run SD generation (S3 strategy) first."
                )

            combined = ConcatDataset([real_ds] + synth_datasets)
            n_real   = len(real_ds)
            n_synth  = sum(len(d) for d in synth_datasets)
            print(f"  Total: {n_real} real (common) + {n_synth} synthetic (rare) = {len(combined)}")

            # ── Write a synthetic-augmented CSV for train_classifier ───────────
            # train_classifier expects a CSV path, so we create one that has
            # common real rows; synth rows are handled by a wrapper below.
            # Simpler: write a combined CSV with synthetic paths as absolute paths.
            synth_rows = []
            for ds in synth_datasets:
                for path in ds.paths:
                    synth_rows.append({
                        "image_path": str(path),
                        "label": ds.label,
                        "class_name": _config.CLASS_NAMES[ds.label]
                            if hasattr(_config, "CLASS_NAMES") else str(ds.label),
                    })

            common_df = pd.read_csv(common_csv)
            synth_df  = pd.DataFrame(synth_rows)
            synth_df["label"] = synth_df["label"].astype(int)

            combined_csv = tmp_path / "synth_only_combined.csv"
            pd.concat([common_df, synth_df], ignore_index=True).to_csv(
                combined_csv, index=False
            )

            val_csv = SPLITS_DIR / args.val_csv

            # ── Backup the real S1 checkpoint before training ─────────────────
            import shutil
            s1_ckpt    = CKPT_DIR / "sota_efficientnetv2_rw_s.pt"
            s1_backup  = CKPT_DIR / "sota_efficientnetv2_rw_s_s1_backup.pt"
            synth_ckpt = CKPT_DIR / "sota_efficientnetv2_rw_s_synth_only.pt"

            if s1_ckpt.exists():
                shutil.copy2(s1_ckpt, s1_backup)
                print(f"  Backed up S1 checkpoint to {s1_backup.name}")

            # ── Train on synth-only dataset ───────────────────────────────────
            print("\n  Training EfficientNetV2-S on synth-only dataset...")
            train_classifier("efficientnetv2_rw_s", combined_csv, val_csv,
                             augmented=False)
            # Save as synth_only checkpoint
            if s1_ckpt.exists():
                shutil.copy2(s1_ckpt, synth_ckpt)

            # ── Evaluate synth-only model ─────────────────────────────────────
            m = load_checkpoint("efficientnetv2_rw_s", suffix="")
            res = evaluate_split(m, split="test")
            del m; torch.cuda.empty_cache()

            # ── Restore the original S1 checkpoint ───────────────────────────
            if s1_backup.exists():
                shutil.copy2(s1_backup, s1_ckpt)
                s1_backup.unlink()
                print(f"  Restored S1 checkpoint")

            out["synth_only"] = {
                "rare_f1":  _rare_f1(res["f1"]),
                "macro_f1": float(np.mean(res["f1"])),
                "acc":      res["acc"],
                "per_rare_f1": {str(c): float(res["f1"][c]) for c in rare},
                "interpretation": (
                    "Compare to S1 rare_f1=0.584 (real only) and "
                    "S3 rare_f1=0.540 (real+synth). "
                    "Near-zero => texture mismatch dominant. "
                    "Near S3 => dilution dominant."
                ),
            }
            print(f"\n  Synthetic-only rare_f1: {out['synth_only']['rare_f1']:.4f}")
            print("  Per-class F1 (rare classes):")
            for c, f in out["synth_only"]["per_rare_f1"].items():
                print(f"    class {c}: {f:.4f}")

    except Exception as e:
        print(f"  Synthetic-only experiment failed: {e}")
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

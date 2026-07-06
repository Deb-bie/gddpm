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
    test_csv  = SPLITS_DIR / args.test_csv
    out = {}

    # Build a test DataLoader (no augmentation, same pattern as evaluate.py)
    from torch.utils.data import DataLoader as _DL
    _test_ds  = GastroVisionDataset(test_csv, "val")   # "val" = no augmentation
    _test_ldr = _DL(_test_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

    # If running standalone, RARE_CLASSES / ULTRA_RARE may be empty — populate
    # from CLASS_COUNTS using the same threshold logic as split creation.
    if not _config.RARE_CLASSES:
        _config.RARE_CLASSES = sorted(
            [c for c, n in _config.CLASS_COUNTS.items()
             if n < args.rare_threshold]
        )
    if not _config.ULTRA_RARE:
        _config.ULTRA_RARE = sorted(
            [c for c, n in _config.CLASS_COUNTS.items()
             if n < args.ultraRare_threshold]
        )
    print(f"  RARE_CLASSES={_config.RARE_CLASSES}  ULTRA_RARE={_config.ULTRA_RARE}")

    def _eval(model):
        """Run evaluate_split and return a dict with acc, macro_f1, rare_f1."""
        from sklearn.metrics import f1_score as _f1
        acc, yt, yp, _ = evaluate_split(model, _test_ldr)
        f1_per_class    = _f1(yt, yp, average=None, zero_division=0,
                              labels=list(range(_config.NUM_CLASSES)))
        return {
            "rare_f1":  _rare_f1(f1_per_class),
            "macro_f1": float(f1_per_class.mean()),
            "acc":      acc,
        }

    # ── A: CNN-only ──────────────────────────────────────────────────────────
    print("\n[A] CNN-only: loading EfficientNetV2-S checkpoint")
    try:
        m = load_checkpoint("efficientnetv2_rw_s", suffix="")
        out["cnn_only"] = _eval(m)
        print(f"  rare_f1={out['cnn_only']['rare_f1']:.4f}  "
              f"macro_f1={out['cnn_only']['macro_f1']:.4f}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  CNN-only failed: {e}")
        out["cnn_only"] = {"error": str(e)}

    # ── B: CNN + proj (no Transformer) ───────────────────────────────────────
    print("\n[B] CNN+proj: training HybridCNNProjOnly...")
    try:
        ckpt_b = CKPT_DIR / "sota_hybrid_cnn_proj_only.pt"
        if ckpt_b.exists():
            print("  Checkpoint exists — skipping training")
        else:
            train_classifier("hybrid_cnn_proj_only", train_csv, val_csv, augmented=False)
        m = load_checkpoint("hybrid_cnn_proj_only", suffix="")
        out["cnn_proj_only"] = _eval(m)
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
        out["full_hybrid"] = _eval(m)
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
# 3. LoRA rank sweep  (ranks 8, 16, 32)
# ─────────────────────────────────────────────────────────────────────────────

def run_lora_rank_sweep(ranks=(8, 16, 32)):
    """
    For each LoRA rank in `ranks`:
      1. Domain-adapt SD (skip if adapter already saved for that rank).
      2. Generate synthetic images to a rank-specific subdirectory.
      3. Train EfficientNetV2-S S3 with those synthetics (skip if ckpt exists).
      4. Evaluate val-set rare-class F1.

    rank-32 reuses the adapter already produced by the main pipeline
    (sd_gastrovision_lora_ema_adapter).  Ranks 8 and 16 each require a fresh
    ~15 k-step domain-adaptation run (~2-3 h each on an A100).

    Output paths:
      adapters  : CKPT_DIR / sd_gastrovision_lora_ema_adapter_r{rank}
      synthetics: SYNTH_DIR / rank{rank} / {cls_idx}/synth_*.png
      checkpoint: CKPT_DIR / sota_efficientnetv2_rw_s_aug_r{rank}.pt
    """
    import shutil, gc
    import pandas as pd
    from torch.utils.data import DataLoader as _DL
    from sklearn.metrics import f1_score as _f1_score
    from generate import domain_adapt_sd, generate_synthetic
    import config as _config

    print("\n" + "=" * 60)
    print("LORA RANK SWEEP (reviewer experiment)")
    print("=" * 60)

    # Ensure RARE_CLASSES populated (same guard as hybrid ablation)
    if not _config.RARE_CLASSES:
        _config.RARE_CLASSES = sorted(
            [c for c, n in _config.CLASS_COUNTS.items() if n < args.rare_threshold]
        )
    if not _config.ULTRA_RARE:
        _config.ULTRA_RARE = sorted(
            [c for c, n in _config.CLASS_COUNTS.items() if n < args.ultraRare_threshold]
        )

    train_csv = SPLITS_DIR / args.train_csv
    val_csv   = SPLITS_DIR / args.val_csv

    # Val DataLoader — built once, reused across ranks
    from dataset import GastroVisionDataset
    _val_ds  = GastroVisionDataset(val_csv, "val")
    _val_ldr = _DL(_val_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=4, pin_memory=True)

    def _eval_val(model):
        from sklearn.metrics import f1_score as _f1
        acc, yt, yp, _ = evaluate_split(model, _val_ldr)
        f1_per = _f1(yt, yp, average=None, zero_division=0,
                     labels=list(range(_config.NUM_CLASSES)))
        return {"rare_f1": _rare_f1(f1_per), "macro_f1": float(f1_per.mean()), "acc": acc}

    out = {}

    for rank in ranks:
        print(f"\n{'─'*55}\nRank {rank}\n{'─'*55}")

        # ── Step A: Domain adaptation ─────────────────────────────────────
        rank_adapter = CKPT_DIR / f"sd_gastrovision_lora_ema_adapter_r{rank}"

        if rank_adapter.exists():
            print(f"  [A] Adapter r{rank} exists — skipping domain adaptation")
        elif rank == 32:
            # Reuse the adapter already produced by the main pipeline
            default_adapter = CKPT_DIR / "sd_gastrovision_lora_ema_adapter"
            if default_adapter.exists():
                shutil.copytree(default_adapter, rank_adapter)
                print(f"  [A] Copied existing rank-32 adapter → {rank_adapter.name}")
            else:
                print(f"  [A] No rank-32 adapter found — running domain adaptation")
                args.lora_rank = rank
                domain_adapt_sd()
                default_adapter = CKPT_DIR / "sd_gastrovision_lora_ema_adapter"
                shutil.copytree(default_adapter, rank_adapter)
        else:
            print(f"  [A] Running domain adaptation with lora_rank={rank} ...")
            saved_rank = args.lora_rank
            args.lora_rank = rank
            # Remove any stale resume checkpoint so we start fresh
            resume = CKPT_DIR / "resume_sd_lora.pt"
            if resume.exists():
                resume.rename(CKPT_DIR / f"resume_sd_lora_r{saved_rank}_backup.pt")
            domain_adapt_sd()
            args.lora_rank = saved_rank
            # Move the freshly-saved adapter to the rank-specific path
            default_adapter = CKPT_DIR / "sd_gastrovision_lora_ema_adapter"
            if default_adapter.exists():
                shutil.copytree(default_adapter, rank_adapter)
                shutil.rmtree(default_adapter)
            # Restore rank-32 adapter so the main pipeline is unaffected
            r32_adapter = CKPT_DIR / "sd_gastrovision_lora_ema_adapter_r32"
            if r32_adapter.exists():
                shutil.copytree(r32_adapter, default_adapter)

        # ── Step B: Synthetic generation ──────────────────────────────────
        rank_synth_dir = SYNTH_DIR / f"rank{rank}"
        rank_synth_csv = rank_synth_dir / "synthetic_train.csv"

        if rank_synth_csv.exists():
            print(f"  [B] Synthetics r{rank} exist — skipping generation")
            synth_df = pd.read_csv(rank_synth_csv)
        else:
            print(f"  [B] Generating synthetics with rank-{rank} adapter ...")
            rank_synth_dir.mkdir(parents=True, exist_ok=True)
            # Temporarily swap the EMA adapter path to the rank-specific one
            default_adapter = CKPT_DIR / "sd_gastrovision_lora_ema_adapter"
            backed_up = False
            if default_adapter.exists() and not default_adapter.samefile(rank_adapter):
                default_adapter.rename(CKPT_DIR / "sd_gastrovision_lora_ema_adapter_main_bak")
                backed_up = True
            if not default_adapter.exists():
                shutil.copytree(rank_adapter, default_adapter)

            # Temporarily redirect SYNTH_DIR
            saved_synth = _config.SYNTH_DIR if hasattr(_config, "SYNTH_DIR") else SYNTH_DIR
            import config as _c2
            _c2.SYNTH_DIR = rank_synth_dir

            try:
                synth_df = generate_synthetic()
                synth_df.to_csv(rank_synth_csv, index=False)
            finally:
                _c2.SYNTH_DIR = saved_synth
                # Restore main adapter
                if backed_up:
                    if default_adapter.exists():
                        shutil.rmtree(default_adapter)
                    bak = CKPT_DIR / "sd_gastrovision_lora_ema_adapter_main_bak"
                    if bak.exists():
                        bak.rename(default_adapter)

        # ── Step C: Build augmented CSV ───────────────────────────────────
        aug_csv = SPLITS_DIR / f"train_aug_r{rank}.csv"
        if not aug_csv.exists():
            real_df  = pd.read_csv(train_csv)
            # Fix synthetic paths to be absolute
            synth_df2 = synth_df.copy()
            synth_df2["image_path"] = synth_df2["image_path"].apply(
                lambda p: str(rank_synth_dir / Path(p).name) if not Path(p).is_absolute() else p
            )
            aug_df = pd.concat(
                [real_df[["image_path", "label", "class_name"]], synth_df2],
                ignore_index=True,
            )
            aug_df.to_csv(aug_csv, index=False)
            print(f"  [C] Built aug CSV: {len(real_df)} real + {len(synth_df2)} synth")
        else:
            print(f"  [C] aug CSV r{rank} exists — reusing")

        # ── Step D: Train S3 with rank-specific synthetics ────────────────
        ckpt_rank = CKPT_DIR / f"sota_efficientnetv2_rw_s_aug_r{rank}.pt"
        ckpt_aug  = CKPT_DIR / "sota_efficientnetv2_rw_s_aug.pt"

        if ckpt_rank.exists():
            print(f"  [D] S3 checkpoint r{rank} exists — skipping training")
        else:
            print(f"  [D] Training EfficientNetV2-S S3 with rank-{rank} synthetics ...")
            # Back up the main aug checkpoint if it exists
            bak_aug = CKPT_DIR / "sota_efficientnetv2_rw_s_aug_main_bak.pt"
            if ckpt_aug.exists():
                shutil.copy2(ckpt_aug, bak_aug)
                ckpt_aug.unlink()
            train_classifier("efficientnetv2_rw_s", aug_csv, val_csv, augmented=True)
            if ckpt_aug.exists():
                shutil.copy2(ckpt_aug, ckpt_rank)
            # Restore main aug checkpoint
            if bak_aug.exists():
                if ckpt_aug.exists():
                    ckpt_aug.unlink()
                bak_aug.rename(ckpt_aug)

        # ── Step E: Evaluate on val set ───────────────────────────────────
        try:
            from models import get_model, load_checkpoint
            # Load rank-specific checkpoint directly
            import torch
            model = get_model("efficientnetv2_rw_s")
            state = torch.load(ckpt_rank, map_location=DEVICE)
            model.load_state_dict(state)
            model = model.to(DEVICE).eval()
            result = _eval_val(model)
            del model; torch.cuda.empty_cache()
            out[f"rank{rank}"] = result
            print(f"  [E] rank={rank}  rare_f1={result['rare_f1']:.4f}  "
                  f"macro_f1={result['macro_f1']:.4f}")
        except Exception as e:
            print(f"  [E] Eval failed for rank {rank}: {e}")
            out[f"rank{rank}"] = {"error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────
    print("\nLoRA rank sweep summary (val-set rare-F1):")
    for k, v in out.items():
        rf = v.get("rare_f1", "ERR")
        mf = v.get("macro_f1", "ERR")
        print(f"  {k:<10}  rare_f1={rf}  macro_f1={mf}")
    best = max(out, key=lambda k: out[k].get("rare_f1", -1))
    print(f"  → Best: {best}  (rare_f1={out[best].get('rare_f1', 'N/A'):.4f})")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-seed stability  (DINOv2 S1 + S3, 3 seeds)
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_seed(model_name="dino_vit_base", seeds=(42, 123, 456)):
    """
    Train `model_name` under S1 (real only) and S3 (SD synthetic) with each
    seed in `seeds`.  For each seed × strategy, record test-set rare-class F1
    and macro-F1.  Reports mean ± std across seeds.

    Checkpoints:
      CKPT_DIR / sota_{model_name}_seed{seed}.pt          (S1)
      CKPT_DIR / sota_{model_name}_aug_seed{seed}.pt      (S3)
    """
    import torch, numpy as np
    from torch.utils.data import DataLoader as _DL
    from sklearn.metrics import f1_score as _f1_score
    import pandas as pd
    import config as _config

    print("\n" + "=" * 60)
    print(f"MULTI-SEED STABILITY  model={model_name}  seeds={list(seeds)}")
    print("=" * 60)

    # Ensure RARE_CLASSES populated
    if not _config.RARE_CLASSES:
        _config.RARE_CLASSES = sorted(
            [c for c, n in _config.CLASS_COUNTS.items() if n < args.rare_threshold]
        )
    if not _config.ULTRA_RARE:
        _config.ULTRA_RARE = sorted(
            [c for c, n in _config.CLASS_COUNTS.items() if n < args.ultraRare_threshold]
        )

    train_csv = SPLITS_DIR / args.train_csv
    val_csv   = SPLITS_DIR / args.val_csv
    test_csv  = SPLITS_DIR / args.test_csv
    aug_csv   = SPLITS_DIR / args.aug_train_csv

    from dataset import GastroVisionDataset
    _test_ds  = GastroVisionDataset(test_csv, "val")
    _test_ldr = _DL(_test_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

    def _set_seed(s):
        torch.manual_seed(s)
        np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def _eval_test(model):
        from sklearn.metrics import f1_score as _f1
        acc, yt, yp, _ = evaluate_split(model, _test_ldr)
        f1_per = _f1(yt, yp, average=None, zero_division=0,
                     labels=list(range(_config.NUM_CLASSES)))
        return {"rare_f1": _rare_f1(f1_per), "macro_f1": float(f1_per.mean()), "acc": acc}

    results_s1, results_s3 = [], []
    import shutil
    from models import get_model

    for seed in seeds:
        print(f"\n── Seed {seed} ──────────────────────────────")
        _set_seed(seed)

        # ── S1 ────────────────────────────────────────────────────────────
        ckpt_s1      = CKPT_DIR / f"sota_{model_name}.pt"
        ckpt_s1_seed = CKPT_DIR / f"sota_{model_name}_seed{seed}.pt"

        if ckpt_s1_seed.exists():
            print(f"  S1 seed={seed}: checkpoint exists — loading")
        else:
            # Back up seed-42 checkpoint (or current) so we can restore
            bak = CKPT_DIR / f"sota_{model_name}_main_bak.pt"
            if ckpt_s1.exists():
                shutil.copy2(ckpt_s1, bak)
                ckpt_s1.unlink()
            _set_seed(seed)
            train_classifier(model_name, train_csv, val_csv, augmented=False)
            if ckpt_s1.exists():
                shutil.copy2(ckpt_s1, ckpt_s1_seed)
            if bak.exists():
                if ckpt_s1.exists():
                    ckpt_s1.unlink()
                bak.rename(ckpt_s1)

        try:
            state = torch.load(ckpt_s1_seed, map_location=DEVICE)
            model = get_model(model_name)
            model.load_state_dict(state)
            model = model.to(DEVICE).eval()
            res = _eval_test(model)
            del model; torch.cuda.empty_cache()
            results_s1.append(res)
            print(f"  S1 seed={seed}: rare_f1={res['rare_f1']:.4f}  "
                  f"macro_f1={res['macro_f1']:.4f}")
        except Exception as e:
            print(f"  S1 seed={seed} eval failed: {e}")
            results_s1.append({"rare_f1": float("nan"), "macro_f1": float("nan"), "error": str(e)})

        # ── S3 ────────────────────────────────────────────────────────────
        ckpt_s3      = CKPT_DIR / f"sota_{model_name}_aug.pt"
        ckpt_s3_seed = CKPT_DIR / f"sota_{model_name}_aug_seed{seed}.pt"

        if not aug_csv.exists():
            print(f"  S3 seed={seed}: aug CSV not found — skipping S3")
            results_s3.append({"rare_f1": float("nan"), "macro_f1": float("nan"),
                                "error": "aug_csv missing"})
        else:
            if ckpt_s3_seed.exists():
                print(f"  S3 seed={seed}: checkpoint exists — loading")
            else:
                bak = CKPT_DIR / f"sota_{model_name}_aug_main_bak.pt"
                if ckpt_s3.exists():
                    shutil.copy2(ckpt_s3, bak)
                    ckpt_s3.unlink()
                _set_seed(seed)
                train_classifier(model_name, aug_csv, val_csv, augmented=True)
                if ckpt_s3.exists():
                    shutil.copy2(ckpt_s3, ckpt_s3_seed)
                if bak.exists():
                    if ckpt_s3.exists():
                        ckpt_s3.unlink()
                    bak.rename(ckpt_s3)

            try:
                state = torch.load(ckpt_s3_seed, map_location=DEVICE)
                model = get_model(model_name)
                model.load_state_dict(state)
                model = model.to(DEVICE).eval()
                res = _eval_test(model)
                del model; torch.cuda.empty_cache()
                results_s3.append(res)
                print(f"  S3 seed={seed}: rare_f1={res['rare_f1']:.4f}  "
                      f"macro_f1={res['macro_f1']:.4f}")
            except Exception as e:
                print(f"  S3 seed={seed} eval failed: {e}")
                results_s3.append({"rare_f1": float("nan"), "macro_f1": float("nan"),
                                    "error": str(e)})

    # ── Aggregate ──────────────────────────────────────────────────────────
    def _agg(results):
        rfs = [r["rare_f1"] for r in results if not np.isnan(r["rare_f1"])]
        mfs = [r["macro_f1"] for r in results if not np.isnan(r["macro_f1"])]
        return {
            "rare_f1_mean":  float(np.mean(rfs))  if rfs else float("nan"),
            "rare_f1_std":   float(np.std(rfs))   if rfs else float("nan"),
            "macro_f1_mean": float(np.mean(mfs))  if mfs else float("nan"),
            "macro_f1_std":  float(np.std(mfs))   if mfs else float("nan"),
            "per_seed":      results,
        }

    agg_s1 = _agg(results_s1)
    agg_s3 = _agg(results_s3)

    print(f"\nMulti-seed summary — {model_name}")
    print(f"  S1  rare_f1 = {agg_s1['rare_f1_mean']:.4f} ± {agg_s1['rare_f1_std']:.4f}"
          f"  macro_f1 = {agg_s1['macro_f1_mean']:.4f} ± {agg_s1['macro_f1_std']:.4f}")
    print(f"  S3  rare_f1 = {agg_s3['rare_f1_mean']:.4f} ± {agg_s3['rare_f1_std']:.4f}"
          f"  macro_f1 = {agg_s3['macro_f1_mean']:.4f} ± {agg_s3['macro_f1_std']:.4f}")

    return {"s1": agg_s1, "s3": agg_s3, "model": model_name, "seeds": list(seeds)}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all_reviewer_experiments():
    results = {}

    print("\n>>> Running hybrid ablation...")
    results["hybrid_ablation"] = run_hybrid_ablation()

    print("\n>>> Running synthetic-only experiment...")
    results["synthetic_only"] = run_synthetic_only()

    if getattr(args, "run_lora_rank_sweep", False):
        print("\n>>> Running LoRA rank sweep...")
        results["lora_rank_sweep"] = run_lora_rank_sweep(ranks=(8, 16, 32))

    if getattr(args, "run_multi_seed", False):
        print("\n>>> Running multi-seed stability experiment...")
        results["multi_seed"] = run_multi_seed(
            model_name=getattr(args, "multi_seed_model", "dino_vit_base"),
            seeds=(42, 123, 456),
        )

    if getattr(args, "run_low_synth_count", False):
        print("\n>>> Running low synthetic-count ablation (10, 20, 50, 100, 250, 500)...")
        from ablation import ablation_synth_count
        synth_csv = SYNTH_DIR / "synthetic_train.csv"
        if synth_csv.exists():
            results["low_synth_count"] = ablation_synth_count(
                model_name="efficientnetv2_rw_s",
                base_train_csv=SPLITS_DIR / args.train_csv,
                synth_csv=synth_csv,
                val_csv=SPLITS_DIR / args.val_csv,
                counts=(10, 20, 50, 100, 250, 500),
            )
        else:
            print(f"  ⚠ Synthetic CSV not found at {synth_csv} — skipping")
            results["low_synth_count"] = {"error": "synthetic_train.csv not found"}

    out_path = RESULTS_DIR / "reviewer_experiments.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReviewer experiments saved to {out_path}")

    return results


if __name__ == "__main__":
    # Standalone entry-point — exec into the GPU pod and run directly:
    #
    #   kubectl exec -it <pod-name> -- bash
    #   cd /opt/repo/gddpm
    #   python gastrovision/reviewer_experiments.py \
    #     --data_dir /data/gastrovision/data \
    #     --output_dir /data/gastrovision \
    #     --rare_threshold 30 --ultraRare_threshold 15 \
    #     --freeze_epochs 20 --fine_tune_epochs 100 \
    #     --batch_size 16 \
    #     --run_lora_rank_sweep \
    #     --run_multi_seed
    #
    # config.NUM_CLASSES is inferred automatically from the first checkpoint loaded.
    run_all_reviewer_experiments()

"""
ablation.py
===========
Ablation studies:
  1. Ensemble subset ablation     — which model combinations drive ensemble gains
  2. Sampling strategy ablation   — WeightedSampler vs shuffle vs class-balanced
  3. Loss function ablation       — FocalLoss vs WeightedCE vs CE
  4. Synthetic count ablation     — 100 / 250 / 500 samples per rare class
  5. LoRA rank ablation           — rank 16 / 32 / 64
  6. Rare-class threshold         — n<15 / n<30 / n<50
"""

import json
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support

import config as _config
from config import (
    args, DEVICE, CKPT_DIR, RESULTS_DIR, SPLITS_DIR, HPARAMS,
)
# _config.RARE_CLASSES, _config.NUM_CLASSES are populated after splits — access via _config.X
from dataset import GastroVisionDataset, get_weighted_sampler
from models import get_model, load_checkpoint
from ensemble import ConfidenceEnsemble, eval_ensemble
from losses import FocalLoss, WeightedCrossEntropy
from train import _eval_acc, _freeze, _unfreeze, train_classifier
from evaluate import evaluate_split


# ==============================================================================
# 1. Ensemble subset ablation
# ==============================================================================

def ablation_ensemble_subsets(split_csv=None, suffix: str = "") -> dict:
    """
    Evaluate all non-trivial subsets (size ≥ 2) of the ensemble models.
    Reports acc, mean F1, rare F1 for each subset.
    """
    if split_csv is None:
        split_csv = SPLITS_DIR / args.val_csv

    print(f"\n{'='*60}\nAblation: Ensemble subsets (suffix='{suffix}')\n{'='*60}")

    ds  = GastroVisionDataset(split_csv, "val")
    ldr = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=4, pin_memory=True)

    # Load all available models for this suffix
    available = []
    for name in args.models:
        ckpt = CKPT_DIR / f"sota_{name}{suffix}.pt"
        if ckpt.exists():
            available.append(name)

    results = {}
    for r in range(1, len(available) + 1):
        for subset in itertools.combinations(available, r):
            subset = list(subset)
            key    = "+".join(subset)
            try:
                ens = ConfidenceEnsemble(subset, suffix=suffix)
                acc, yt, yp, _ = eval_ensemble(ens, ldr, subset=subset)
                _, _, f1, _ = precision_recall_fscore_support(
                    yt, yp, labels=list(range(_config.NUM_CLASSES)), average=None, zero_division=0
                )
                rare_idx   = [c for c in _config.RARE_CLASSES if c < _config.NUM_CLASSES]
                f1_rare    = float(f1[rare_idx].mean()) if rare_idx else 0.0
                results[key] = {"acc": acc, "f1_mean": float(f1.mean()), "f1_rare": f1_rare,
                                "n_models": len(subset)}
                print(f"  {key:<60}  acc={acc:.4f}  f1={f1.mean():.4f}  rare_f1={f1_rare:.4f}")
            except Exception as e:
                print(f"  {key}: failed — {e}")

    out = RESULTS_DIR / f"ablation_ensemble{suffix}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out}")
    return results


# ==============================================================================
# 2. Sampling strategy ablation
# ==============================================================================

def ablation_sampling(model_name: str, train_csv, val_csv,
                      epochs: int = 10) -> dict:
    """
    Compare three sampling strategies for one model, holding all else equal.
    Strategies: (a) shuffle=True, (b) WeightedRandomSampler, (c) ClassBalancedBatch
    """
    from torch.utils.data import WeightedRandomSampler
    print(f"\n{'='*60}\nAblation: Sampling strategy — {model_name}\n{'='*60}")

    results = {}
    strategies = {
        "shuffle": {"sampler": None, "shuffle": True},
        "weighted_sampler": {"sampler": "weighted", "shuffle": False},
    }

    for strat_name, strat_cfg in strategies.items():
        model  = get_model(model_name)
        crit   = FocalLoss(gamma=HPARAMS[model_name]["gamma"])
        opt    = torch.optim.AdamW(model.parameters(), lr=HPARAMS[model_name]["lr"])
        ds     = GastroVisionDataset(train_csv, "train")

        if strat_cfg["sampler"] == "weighted":
            sampler = get_weighted_sampler(train_csv)
            tl = DataLoader(ds, batch_size=16, sampler=sampler, num_workers=2)
        else:
            tl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)

        val_ds  = GastroVisionDataset(val_csv, "val")
        vl      = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

        for ep in range(epochs):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()

        acc, yt, yp = _eval_acc(model, vl)
        _, _, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=list(range(_config.NUM_CLASSES)), average=None, zero_division=0
        )
        rare_idx = [c for c in _config.RARE_CLASSES if c < _config.NUM_CLASSES]
        results[strat_name] = {
            "acc": acc, "f1_mean": float(f1.mean()),
            "f1_rare": float(f1[rare_idx].mean()) if rare_idx else 0.0,
        }
        print(f"  {strat_name:<25}  acc={acc:.4f}  rare_f1={results[strat_name]['f1_rare']:.4f}")
        del model
        torch.cuda.empty_cache()

    out = RESULTS_DIR / f"ablation_sampling_{model_name}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ==============================================================================
# 3. Loss function ablation
# ==============================================================================

def ablation_loss_function(model_name: str, train_csv, val_csv,
                           epochs: int = 10) -> dict:
    """Compare FocalLoss vs WeightedCE vs standard CE."""
    from config import CLASS_COUNTS
    print(f"\n{'='*60}\nAblation: Loss function — {model_name}\n{'='*60}")

    counts  = [CLASS_COUNTS.get(i, 1) for i in range(_config.NUM_CLASSES)]
    losses  = {
        "focal":       FocalLoss(gamma=2.0),
        "weighted_ce": WeightedCrossEntropy(counts, DEVICE),
        "ce":          torch.nn.CrossEntropyLoss(),
    }
    results = {}
    val_ds  = GastroVisionDataset(val_csv, "val")
    vl      = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    for loss_name, crit in losses.items():
        model  = get_model(model_name)
        opt    = torch.optim.AdamW(model.parameters(), lr=HPARAMS[model_name]["lr"])
        ds     = GastroVisionDataset(train_csv, "train")
        sampler = get_weighted_sampler(train_csv)
        tl     = DataLoader(ds, batch_size=16, sampler=sampler, num_workers=2)

        for _ in range(epochs):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()

        acc, yt, yp = _eval_acc(model, vl)
        _, _, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=list(range(_config.NUM_CLASSES)), average=None, zero_division=0
        )
        rare_idx = [c for c in _config.RARE_CLASSES if c < _config.NUM_CLASSES]
        results[loss_name] = {
            "acc": acc, "f1_mean": float(f1.mean()),
            "f1_rare": float(f1[rare_idx].mean()) if rare_idx else 0.0,
        }
        print(f"  {loss_name:<15}  acc={acc:.4f}  rare_f1={results[loss_name]['f1_rare']:.4f}")
        del model
        torch.cuda.empty_cache()

    out = RESULTS_DIR / f"ablation_loss_{model_name}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ==============================================================================
# 4. Synthetic sample count ablation
# ==============================================================================

def ablation_synth_count(model_name: str, base_train_csv,
                         synth_csv, val_csv,
                         counts=(100, 250, 500)) -> dict:
    """
    Train with different numbers of synthetic images per rare class.
    Requires that the full synthetic CSV already exists.
    """
    print(f"\n{'='*60}\nAblation: Synthetic count — {model_name}\n{'='*60}")

    real_df  = pd.read_csv(base_train_csv)
    synth_df = pd.read_csv(synth_csv)
    results  = {}

    for count in counts:
        # Sample at most `count` images per rare class
        subset_rows = []
        for cls in synth_df["label"].unique():
            cls_rows = synth_df[synth_df["label"] == cls]
            subset_rows.append(cls_rows.head(count))
        subset_synth = pd.concat(subset_rows, ignore_index=True)

        combined = pd.concat([real_df, subset_synth], ignore_index=True)
        tmp_csv  = SPLITS_DIR / f"_ablation_synth_{count}.csv"
        combined.to_csv(tmp_csv, index=False)

        model = get_model(model_name)
        history = train_classifier(model_name, tmp_csv, val_csv, augmented=False)
        best_acc = max(history["val_acc"])

        # Load best checkpoint and evaluate
        ckpt = CKPT_DIR / f"sota_{model_name}.pt"
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        val_ds = GastroVisionDataset(val_csv, "val")
        vl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
        acc, yt, yp = _eval_acc(model, vl)
        _, _, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=list(range(_config.NUM_CLASSES)), average=None, zero_division=0
        )
        rare_idx = [c for c in _config.RARE_CLASSES if c < _config.NUM_CLASSES]
        results[count] = {
            "acc": acc, "f1_mean": float(f1.mean()),
            "f1_rare": float(f1[rare_idx].mean()) if rare_idx else 0.0,
            "n_synth": len(subset_synth),
        }
        print(f"  count={count:<5}  acc={acc:.4f}  rare_f1={results[count]['f1_rare']:.4f}")
        tmp_csv.unlink(missing_ok=True)
        del model
        torch.cuda.empty_cache()

    out = RESULTS_DIR / f"ablation_synth_count_{model_name}.json"
    with open(out, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    return results


# ==============================================================================
# 5. LoRA rank ablation (summary — runs domain_adapt_sd for each rank)
# ==============================================================================

def ablation_lora_rank(ranks=(16, 32, 64)) -> dict:
    """
    Train SD LoRA adapters at different ranks and record final training loss.
    Full retraining is expensive; this function records what's needed for Table 7.
    Each run requires calling domain_adapt_sd() with the specified rank.
    """
    print(f"\nLoRA rank ablation: ranks={ranks}")
    print("NOTE: This ablation requires retraining SD LoRA adapters.")
    print("Run manually with --lora_rank <value> for each rank and record FID/loss.")
    print("Results should be recorded in ablation_lora_rank.json manually.")

    placeholder = {str(r): {"lora_rank": r, "fid": None, "kid": None, "final_loss": None}
                   for r in ranks}
    out = RESULTS_DIR / "ablation_lora_rank.json"
    if not out.exists():
        with open(out, "w") as f:
            json.dump(placeholder, f, indent=2)
    return placeholder


# ==============================================================================
# Summary printer
# ==============================================================================

def print_ablation_summary():
    """Print a consolidated summary of all completed ablations."""
    files = {
        "Ensemble subsets": RESULTS_DIR / "ablation_ensemble.json",
        "Ensemble (aug)":   RESULTS_DIR / "ablation_ensemble_aug.json",
        "Sampling":         RESULTS_DIR / "ablation_sampling_efficientnetv2_rw_s.json",
        "Loss function":    RESULTS_DIR / "ablation_loss_efficientnetv2_rw_s.json",
        "Synth count":      RESULTS_DIR / "ablation_synth_count_efficientnetv2_rw_s.json",
        "LoRA rank":        RESULTS_DIR / "ablation_lora_rank.json",
    }
    print(f"\n{'='*70}\nAblation Summary\n{'='*70}")
    for label, path in files.items():
        if not path.exists():
            print(f"\n{label}: [not run yet]")
            continue
        with open(path) as f:
            data = json.load(f)
        print(f"\n{label}:")
        for k, v in data.items():
            if isinstance(v, dict):
                acc  = v.get("acc", "—")
                f1m  = v.get("f1_mean", "—")
                f1r  = v.get("f1_rare", "—")
                acc_s  = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
                f1m_s  = f"{f1m:.4f}" if isinstance(f1m, float) else str(f1m)
                f1r_s  = f"{f1r:.4f}" if isinstance(f1r, float) else str(f1r)
                print(f"  {k:<35}  acc={acc_s}  f1={f1m_s}  rare_f1={f1r_s}")

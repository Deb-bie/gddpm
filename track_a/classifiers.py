"""
track_a/classifiers.py
=======================
Classifier training wrapper for the n-grid x condition matrix (Phase 5).

DESIGN DECISION — per-class isolated curves, not one shared multiclass
sweep. This is the one substantive methodological call in this file that
wasn't already locked into the review doc, so it's documented here in full
rather than silently assumed:

  At each (dataset, target_class, n, condition, backbone) cell, ONLY
  target_class's training data is perturbed (subsampled to n, and/or mixed
  with synthetic images, and/or zeroed out for synth-only). Every OTHER
  class — including other sweep classes — stays at its full natural real
  training set, unchanged, in every single run.

  The alternative (subsample ALL sweep classes to the same n simultaneously
  in one shared model, training one classifier per n instead of one per
  class-per-n) is cheaper — one training run covers every class at once —
  but confounds the resulting curve: if Ulcer and Esophageal varices are
  both shrunk to n=16 at the same time, a drop in Ulcer's F1 could be
  partly an artifact of Esophageal varices' data ALSO having shrunk (shared
  feature-space competition, shifted decision boundaries), not purely an
  effect of Ulcer's own n. Since RQ1's crossover point n* is a per-class
  quantity, isolating the causal variable (one class's n) per run is the
  correct design, and it's a direct generalization of what
  reviewer_experiments.py's run_synthetic_only() ALREADY does for
  GastroVision (common classes full, one designated set zeroed + synthetic)
  — this just parameterizes that existing pattern down to a single class
  and a single n instead of hardcoding "all rare classes, real=0."

  Cost implication: this multiplies classifier-training runs by the number
  of sweep classes per dataset (up to 7 for GastroVision, 2 for HAM10000,
  2 for PathMNIST) rather than running once per n. Given each run is a
  cheap classifier fine-tune (not a diffusion training run), this is the
  right trade against the alternative's confound — but it IS a real cost,
  worth knowing about before kicking off the full matrix on the cluster.

Also duplicates (not imports) FocalLoss, EfficientNetV2-S, DINOv2Classifier,
the transform builder, and the freeze/unfreeze helpers from
gastrovision/{losses,models,dataset,train}.py — same reasoning as the rest
of track_a/: those modules sit behind gastrovision/config.py's import-time
argparse parser (models.py and train.py; losses.py itself has no config
dependency but is duplicated anyway for a single self-contained file), and
duplicating a few dependency-free classes is cheaper than making Track A's
import graph depend on gastrovision/'s specific sys.path assumptions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import timm

from sklearn.metrics import precision_recall_fscore_support


# ==============================================================================
# Loss
# ==============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ==============================================================================
# Backbones — only the two kept per Section 9 flag 7
# ==============================================================================

def get_effnetv2_s(num_classes: int) -> nn.Module:
    return timm.create_model("efficientnetv2_rw_s", pretrained=True, num_classes=num_classes)


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        feat_dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout), nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbones(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def get_features(self, x):
        return self.backbone(x)


MODEL_REGISTRY = {
    "efficientnetv2_rw_s": get_effnetv2_s,
    "dinov2":              lambda n: DINOv2Classifier(n),
}


def get_model(name: str, num_classes: int, device) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes).to(device)


def get_hparams(model_name: str, args) -> dict:
    """Per-backbone hyperparameters — DINOv2 needs a much lower fine-tune LR
    and a couple extra epochs to avoid catastrophic forgetting, same finding
    as gastrovision/config.py's HPARAMS comment for the same architecture."""
    base = {
        "lr": args.lr, "freeze_epochs": args.freeze_epochs,
        "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
        "gamma": args.gamma, "freeze_lr_mult": args.freeze_lr_mult,
        "weight_decay": args.weight_decay,
    }
    if model_name == "dinov2":
        base.update({
            "lr": args.lr * 0.05,
            "fine_tune_epochs": args.fine_tune_epochs + 8,
            "batch_size": min(args.batch_size, 16),
            "freeze_lr_mult": 5.0,
        })
    return base


def _freeze(model, model_name):
    if model_name == "dinov2":
        model.freeze_backbones()
    else:
        for p in model.parameters():
            p.requires_grad = False
        head = getattr(model, "head", None) or getattr(model, "classifier", None)
        for p in head.parameters():
            p.requires_grad = True


def _unfreeze(model, model_name):
    if model_name == "dinov2":
        model.unfreeze_all()
    else:
        for p in model.parameters():
            p.requires_grad = True


# ==============================================================================
# Transforms (copied from gastrovision/dataset.py's build_classifier_transform)
# ==============================================================================

_IMAGENET_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def build_classifier_transform(split: str, img_size: int, heavy: bool = False) -> T.Compose:
    if split == "train" and not heavy:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ToTensor(), _IMAGENET_NORM,
        ])
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), _IMAGENET_NORM])


# ==============================================================================
# Dataset — resolves image_path against a per-row `source` -> root mapping
# ==============================================================================

class TrackAClassifierDataset(Dataset):
    """
    df columns required: image_path, label, source.
    `source` indexes into `roots` (e.g. {"real": data_dir, "synth_sd": ...,
    "synth_dcgan": ...}) so real and synthetic images — which live under
    completely different directory trees — resolve correctly without any
    of GastroVisionDataset's "try each candidate root" guessing.
    """

    def __init__(self, df: pd.DataFrame, roots: dict, split: str, img_size: int, heavy: bool = False):
        self.df = df.reset_index(drop=True)
        self.roots = {k: Path(v) for k, v in roots.items()}
        self.transform = build_classifier_transform(split, img_size, heavy=heavy)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        root = self.roots[row["source"]]
        try:
            img = Image.open(root / row["image_path"]).convert("RGB")
            return self.transform(img), int(row["label"])
        except Exception as e:
            print(f"  Warning: {e}")
            size = self.transform.transforms[0].size[0]
            return torch.zeros(3, size, size), int(row["label"])


def get_weighted_sampler(df: pd.DataFrame, num_classes: int) -> WeightedRandomSampler:
    labels = df["label"].astype(int).tolist()
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = 1.0 / counts
    sw = [w[l] for l in labels]
    return WeightedRandomSampler(sw, len(sw), replacement=True)


# ==============================================================================
# Synthetic-label integrity check
# ==============================================================================

def validate_synthetic_labels(synth_df: pd.DataFrame, n_samples_per_class: int = 50,
                               seed: int = 42) -> None:
    """
    Samples up to n_samples_per_class random rows per class from a
    synthetic-image DataFrame and verifies the image_path's leading
    directory component matches the row's `label` column. Both
    generate_synthetic_for_classes() (SD/LoRA — paths like
    "{class_id}/synth_*.png") and generate_dcgan_images() (DCGAN — paths
    like "{class_id}/n{n}/dcgan_*.png") write the class id as the first
    path component, so this is a cheap, direct check that a label was
    never assigned from the wrong source.

    This checks the complementary direction to the bug Phase 5's dry-run
    test already caught in build_condition_df (a synthetic row missing
    its `source` tag entirely) — this instead confirms that a `label`
    value that IS present actually corresponds to the directory the image
    file lives in on disk. Raises AssertionError (not a warning): a label
    bug here would silently corrupt every downstream classifier run that
    consumes this pool, so it's deliberately loud rather than logged and
    ignored.
    """
    rng_seed = seed
    mismatches = []
    for cls, group in synth_df.groupby("label"):
        sample = group.sample(n=min(n_samples_per_class, len(group)), random_state=rng_seed)
        for _, row in sample.iterrows():
            parts = Path(row["image_path"]).parts
            leading = parts[0] if parts else None
            if leading != str(cls):
                mismatches.append((row["image_path"], cls, leading))

    if mismatches:
        detail = "\n".join(
            f"  path={p}  label={l}  path_leading_dir={d}" for p, l, d in mismatches[:20]
        )
        raise AssertionError(
            f"validate_synthetic_labels found {len(mismatches)} label/path "
            f"mismatch(es) out of {min(n_samples_per_class, len(synth_df))}"
            f"-per-class samples (showing up to 20):\n{detail}"
        )

    print(f"validate_synthetic_labels: OK — checked up to {n_samples_per_class} "
          f"images/class across {synth_df['label'].nunique()} classes, no mismatches")


# ==============================================================================
# Condition-dataframe construction — the n-grid x condition matrix itself
# ==============================================================================

def build_condition_df(all_train_df: pd.DataFrame, label_col: str, target_class,
                        n: int, condition: str, subsample_manifest: dict,
                        synth_pool_df: pd.DataFrame = None, synth_ratio: float = 1.0,
                        seed: int = 42):
    """
    Returns a combined DataFrame (columns: image_path, label, class_name,
    source) for ONE (target_class, n, condition) cell, or None if the cell
    is infeasible (real_only/sd_lora_synth/dcgan_synth at an n the class
    doesn't have enough real images for — synth_only has no such
    constraint since it uses zero real images for the target class).

    condition in {"real_only", "sd_lora_synth", "dcgan_synth", "synth_only"}.
    synth_pool_df: pre-filtered to the source generative model's images for
    this class (for sd_lora_synth/synth_only: the class's fixed pool from
    generate_synthetic_for_classes(); for dcgan_synth: the (class, n)
    -specific pool from generate_dcgan_images() — already sized to match
    n's real-data budget, so synth_ratio there is typically left at 1.0).
    """
    other_df = all_train_df[all_train_df[label_col] != target_class].copy()
    other_df["source"] = "real"
    rows = [other_df]

    if condition == "synth_only":
        if synth_pool_df is not None and len(synth_pool_df) > 0:
            target_synth = synth_pool_df[synth_pool_df[label_col] == target_class].copy()
            # Always stamp explicitly, same reasoning as the sd_lora_synth/
            # dcgan_synth branch below — do NOT special-case "only if
            # missing". generate_synthetic_for_classes() already writes a
            # `source` column into its CSV, but stamped "sd_lora_ema" (its
            # own internal provenance label), not "synth_sd" (the key
            # TrackAClassifierDataset's `roots` dict actually uses). Since
            # the column isn't missing, the old `if "source" not in
            # columns` guard silently let "sd_lora_ema" through instead of
            # correcting it, causing KeyError('sd_lora_ema') the first time
            # any synth_only cell reached the DataLoader. main.py's Step 8
            # always sources synth_only from sd_pool_main, never the DCGAN
            # pool, so "synth_sd" is always the correct target here.
            target_synth["source"] = "synth_sd"
            rows.append(target_synth)
        return pd.concat(rows, ignore_index=True)

    target_real = subsample_manifest.get(target_class, {}).get(n)
    if target_real is None:
        return None  # infeasible cell
    target_real = target_real.copy()
    target_real["source"] = "real"
    rows.append(target_real)

    if condition in ("sd_lora_synth", "dcgan_synth") and synth_pool_df is not None:
        n_wanted = max(1, round(n * synth_ratio))
        pool = synth_pool_df[synth_pool_df[label_col] == target_class]
        if len(pool) > 0:
            take = pool.sample(n=min(n_wanted, len(pool)), random_state=seed).copy()
            # Always stamp the source explicitly rather than trusting the
            # caller's pool to already carry the right label — a missing
            # `source` here means these rows silently fail to resolve to
            # a root in TrackAClassifierDataset (found the hard way: see
            # the Phase 5 dry-run test, where an untagged pool produced
            # rows with a null source that `value_counts()` just drops).
            take["source"] = "synth_sd" if condition == "sd_lora_synth" else "synth_dcgan"
            rows.append(take)

    return pd.concat(rows, ignore_index=True)


# ==============================================================================
# Evaluation
# ==============================================================================

def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == y_true).astype(float)
    bins        = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def _eval_loader(model, loader, device):
    model.eval()
    yt, yp, pr = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            with autocast():
                logits = model(xb.to(device))
                probs  = torch.softmax(logits, dim=1)
            yt.append(yb.numpy())
            yp.append(probs.argmax(1).cpu().numpy())
            pr.append(probs.cpu().numpy())
    yt, yp, pr = np.concatenate(yt), np.concatenate(yp), np.concatenate(pr)
    return float((yt == yp).mean()), yt, yp, pr


# ==============================================================================
# Training — two-phase frozen/fine-tune, same schedule as train.py
# ==============================================================================

def train_one_condition(backbone_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                         roots: dict, num_classes: int, target_class, args, device,
                         ckpt_path: Path, results_path: Path) -> dict:
    """
    Trains one classifier on one condition's train_df, evaluates on the
    dataset's UNCHANGED val_df (val/test splits are never perturbed — only
    the training set varies across cells), and returns/saves a result dict
    matching eval_results.json's schema (acc, f1 list, f1_mean, f1_rare
    -> here f1_target, since this run only has one perturbed class, ece).

    Skips training if results_path already exists (resumability, same
    convention as gastrovision/main.py).
    """
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    cfg = get_hparams(backbone_name, args)
    crit = FocalLoss(gamma=cfg["gamma"])
    scaler = GradScaler()
    model = get_model(backbone_name, num_classes, device)

    train_ds = TrackAClassifierDataset(train_df, roots, "train", args.img_size)
    val_ds   = TrackAClassifierDataset(val_df,   roots, "val",   args.img_size)
    sampler  = get_weighted_sampler(train_df, num_classes)
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                    num_workers=4, pin_memory=True)
    vl = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=4, pin_memory=True)

    # Phase 1 — frozen backbone
    _freeze(model, backbone_name)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["lr"] * cfg["freeze_lr_mult"])
    for ep in range(cfg["freeze_epochs"]):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with autocast():
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(opt); scaler.update()
        print(f"  [{backbone_name}] freeze ep {ep+1}/{cfg['freeze_epochs']}")

    # Phase 2 — full fine-tune, track best val acc
    _unfreeze(model, backbone_name)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["fine_tune_epochs"])
    best_acc, best_state = 0.0, None

    for ep in range(cfg["fine_tune_epochs"]):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with autocast():
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sch.step()
        acc, _, _, _ = _eval_loader(model, vl, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{backbone_name}] finetune ep {ep+1}/{cfg['fine_tune_epochs']}  val_acc={acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_path)

    acc, yt, yp, pr = _eval_loader(model, vl, device)
    _, _, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=list(range(num_classes)), average=None, zero_division=0
    )
    ece = compute_ece(pr, yt)

    result = {
        "acc": acc,
        "f1": f1.tolist(),
        "f1_mean": float(f1.mean()),
        "f1_target": float(f1[target_class]) if target_class < len(f1) else float("nan"),
        "target_class": int(target_class),
        "ece": ece,
        "n_train": len(train_df),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    # Persist raw val-set predictions alongside the JSON — needed for
    # track_a/analysis.py's bootstrap CIs on rare/target-class F1. The
    # aggregate f1_target above is a point estimate only; a legitimate CI
    # requires resampling over actual (y_true, y_pred) pairs, not just the
    # summary number. Compact int arrays, one file per cell — cheap even
    # across the full matrix.
    np.savez(results_path.with_suffix(".preds.npz"), y_true=yt, y_pred=yp)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ==============================================================================
# RQ7 / A7 — S3_pretrain_only: augmented head init, real-only fine-tune
# ==============================================================================

def train_pretrain_only_condition(backbone_name: str, stage1_df: pd.DataFrame, stage2_df: pd.DataFrame,
                                   val_df: pd.DataFrame, roots: dict, num_classes: int, target_class,
                                   args, device, ckpt_path: Path, results_path: Path) -> dict:
    """
    "s3_pretrain_only": stage 1 (frozen backbone, head-only) trains on
    stage1_df (typically the sd_lora_synth condition's DataFrame — real +
    synthetic), stage 2 (full fine-tune) SWITCHES to stage2_df (typically
    real_only) — model weights carry over between stages, nothing is
    reinitialized. Tests whether synthetic images help the classification
    head learn class-prototype structure even when they'd hurt the
    feature extractor if used throughout full fine-tuning too (Phase 5's
    build_condition_df/train_one_condition only support ONE dataset for
    both stages, so this needed its own function rather than a parameter
    tweak).

    Assertions (fail loudly, not silently): stage1_df and stage2_df must
    cover the same class set, since a class present in stage 1 but absent
    from stage 2 would leave that output neuron never updated in phase 2 —
    not a crash, just quietly wrong. val_df is shared and passed once, so
    there's no way for the two stages to see different val/test data by
    construction (no separate parameter to accidentally diverge).
    """
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    stage1_classes = set(stage1_df["label"].astype(int).unique())
    stage2_classes = set(stage2_df["label"].astype(int).unique())
    if not stage1_classes.issubset(stage2_classes) or not stage2_classes.issubset(stage1_classes):
        missing_from_2 = stage1_classes - stage2_classes
        missing_from_1 = stage2_classes - stage1_classes
        raise AssertionError(
            "train_pretrain_only_condition: stage1_df and stage2_df must cover "
            f"the same class set. Missing from stage2: {missing_from_2}. "
            f"Missing from stage1: {missing_from_1}. A class dropped between "
            "stages would leave its output neuron un-updated in phase 2 without "
            "raising any error — not something to discover after the fact."
        )

    cfg = get_hparams(backbone_name, args)
    crit = FocalLoss(gamma=cfg["gamma"])
    scaler = GradScaler()
    model = get_model(backbone_name, num_classes, device)

    stage1_ds = TrackAClassifierDataset(stage1_df, roots, "train", args.img_size)
    stage2_ds = TrackAClassifierDataset(stage2_df, roots, "train", args.img_size)
    val_ds    = TrackAClassifierDataset(val_df,    roots, "val",   args.img_size)
    sampler1  = get_weighted_sampler(stage1_df, num_classes)
    sampler2  = get_weighted_sampler(stage2_df, num_classes)
    tl1 = DataLoader(stage1_ds, batch_size=cfg["batch_size"], sampler=sampler1, num_workers=4, pin_memory=True)
    tl2 = DataLoader(stage2_ds, batch_size=cfg["batch_size"], sampler=sampler2, num_workers=4, pin_memory=True)
    vl  = DataLoader(val_ds,    batch_size=cfg["batch_size"], shuffle=False,   num_workers=4, pin_memory=True)

    # Phase 1 — frozen backbone, trained on STAGE 1 data (augmented)
    print(f"  [{backbone_name}] Stage 1 active dataset: augmented (n={len(stage1_df)})")
    _freeze(model, backbone_name)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["lr"] * cfg["freeze_lr_mult"])
    for ep in range(cfg["freeze_epochs"]):
        model.train()
        for xb, yb in tl1:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with autocast():
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(opt); scaler.update()
        print(f"  [{backbone_name}] stage1/freeze ep {ep+1}/{cfg['freeze_epochs']}")

    # Phase 2 — full fine-tune, SWITCHED to STAGE 2 data (real-only).
    # Model weights are NOT reinitialized between stages — same `model`
    # object, same optimizer state discarded (fresh AdamW) but same
    # parameters carried straight over from stage 1's frozen-head training.
    print(f"  [{backbone_name}] Stage 2 active dataset: real-only (n={len(stage2_df)})")
    _unfreeze(model, backbone_name)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["fine_tune_epochs"])
    best_acc, best_state = 0.0, None

    for ep in range(cfg["fine_tune_epochs"]):
        model.train()
        for xb, yb in tl2:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with autocast():
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sch.step()
        acc, _, _, _ = _eval_loader(model, vl, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{backbone_name}] stage2/finetune ep {ep+1}/{cfg['fine_tune_epochs']}  val_acc={acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_path)

    acc, yt, yp, pr = _eval_loader(model, vl, device)
    _, _, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=list(range(num_classes)), average=None, zero_division=0
    )
    ece = compute_ece(pr, yt)

    result = {
        "acc": acc,
        "f1": f1.tolist(),
        "f1_mean": float(f1.mean()),
        "f1_target": float(f1[target_class]) if target_class < len(f1) else float("nan"),
        "target_class": int(target_class),
        "ece": ece,
        "n_train_stage1": len(stage1_df),
        "n_train_stage2": len(stage2_df),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    np.savez(results_path.with_suffix(".preds.npz"), y_true=yt, y_pred=yp)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

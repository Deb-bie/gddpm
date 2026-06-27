"""
evaluate.py
===========
Evaluation functions:
  - evaluate_split()      : single model on val OR test split
  - evaluate_all()        : all models + ensemble on val (S1 / S3)
  - evaluate_heavy_aug()  : all models + ensemble on val (S2)
  - evaluate_test()       : final held-out test evaluation (call once, last)
  - compute_fid()         : pooled FID/KID for rare classes
  - compute_per_class_fid(): per-class FID/KID for each rare class
"""

import json
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
from sklearn.calibration import calibration_curve

import torchvision.transforms as T

import config as _config
from config import (
    args, DEVICE, SPLITS_DIR, RESULTS_DIR, CKPT_DIR, CALIB_DIR,
    IMAGE_ROOT_DIR, OUTPUT_DIR, CLASS_NAMES, FID_TRANSFORM,
)
# _config.NUM_CLASSES, _config.RARE_CLASSES, _config.ULTRA_RARE are populated after splits — access via _config.X
from dataset import GastroVisionDataset
from models import get_model, load_checkpoint
from ensemble import ConfidenceEnsemble, eval_ensemble
from train import _eval_acc


# ==============================================================================
# Single-model evaluation
# ==============================================================================

def evaluate_split(model, loader):
    """Returns (acc, y_true, y_pred, probs)."""
    model.eval()
    yt_list, yp_list, pr_list = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            with autocast():
                logits = model(xb.to(DEVICE))
                probs  = torch.softmax(logits, dim=1)
                preds  = probs.argmax(1)
            yt_list.append(yb.numpy())
            yp_list.append(preds.cpu().numpy())
            pr_list.append(probs.cpu().numpy())
    yt = np.concatenate(yt_list)
    yp = np.concatenate(yp_list)
    pr = np.concatenate(pr_list)
    return float((yt == yp).mean()), yt, yp, pr


# ==============================================================================
# Calibration (reliability diagram + ECE)
# ==============================================================================

def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == y_true).astype(float)
    bins        = np.linspace(0, 1, n_bins + 1)
    ece         = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc_bin  = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += mask.sum() / len(y_true) * abs(acc_bin - conf_bin)
    return float(ece)


def plot_calibration(probs: np.ndarray, y_true: np.ndarray,
                     model_name: str, suffix: str = ""):
    """Save reliability diagram for a model."""
    from sklearn.preprocessing import label_binarize
    n_cls   = probs.shape[1]
    y_bin   = label_binarize(y_true, classes=list(range(n_cls)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    mean_conf = probs.max(axis=1)
    mean_acc  = (probs.argmax(axis=1) == y_true).astype(float)
    frac_pos, mean_pred = calibration_curve(mean_acc, mean_conf, n_bins=15)
    ax.plot(mean_pred, frac_pos, "s-", label="Model")

    ece = compute_ece(probs, y_true)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of correct predictions")
    ax.set_title(f"Calibration — {model_name}{suffix}  (ECE={ece:.4f})")
    ax.legend()
    plt.tight_layout()
    path = CALIB_DIR / f"calibration_{model_name}{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return ece


# ==============================================================================
# GradCAM output
# ==============================================================================

def save_gradcam(model, model_name: str, loader, n_per_class: int = 5, suffix: str = ""):
    """
    Save GradCAM overlays for rare classes.
    Requires pytorch_grad_cam.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("  GradCAM not installed — skipping")
        return

    # Select target layer
    target_layer = None
    if hasattr(model, "cnn"):
        target_layer = list(model.cnn.children())[-1]
    elif hasattr(model, "features"):
        target_layer = list(model.features.children())[-1]

    if target_layer is None:
        print(f"  GradCAM: could not identify target layer for {model_name}")
        return

    cam     = GradCAM(model=model, target_layers=[target_layer])
    counts  = {c: 0 for c in _config.RARE_CLASSES}
    inv_norm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    model.eval()
    for xb, yb in loader:
        for i in range(len(yb)):
            cls = int(yb[i])
            if cls not in _config.RARE_CLASSES:
                continue
            if counts[cls] >= n_per_class:
                continue
            inp = xb[i:i+1].to(DEVICE)
            target = [ClassifierOutputTarget(cls)]
            grayscale_cam = cam(input_tensor=inp, targets=target)[0]
            rgb_img = inv_norm(xb[i]).permute(1, 2, 0).clamp(0, 1).numpy()
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            save_dir = RESULTS_DIR / "gradcam" / f"class_{cls:02d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image as PILImage
            PILImage.fromarray(visualization).save(
                save_dir / f"{model_name}{suffix}_{counts[cls]:02d}.png"
            )
            counts[cls] += 1
        if all(v >= n_per_class for v in counts.values()):
            break
    print(f"  GradCAM overlays saved → {RESULTS_DIR / 'gradcam'}")


# ==============================================================================
# FID / KID computation
# ==============================================================================

def _fid_features(df: pd.DataFrame, root_dir: Path, model, hook_list: list):
    feats = []
    for _, row in df.iterrows():
        try:
            img    = Image.open(root_dir / row["image_path"]).convert("RGB")
            tensor = FID_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            hook_list.clear()
            with torch.no_grad():
                _ = model(tensor)
            if hook_list:
                feats.append(hook_list[0].flatten())
        except Exception:
            continue
    return np.array(feats) if feats else None


def _frechet(r: np.ndarray, s: np.ndarray) -> float:
    mr, ms = r.mean(0), s.mean(0)
    sr     = np.cov(r, rowvar=False) + 1e-6 * np.eye(r.shape[1])
    ss     = np.cov(s, rowvar=False) + 1e-6 * np.eye(s.shape[1])
    d      = mr - ms
    cov    = sqrtm(sr @ ss)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(d @ d + np.trace(sr) + np.trace(ss) - 2 * np.trace(cov))


def _kid(r: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics.pairwise import polynomial_kernel
    n   = min(len(r), len(s), 500)
    rng = np.random.default_rng(args.seed)
    r   = r[rng.choice(len(r), n, replace=False)]
    s   = s[rng.choice(len(s), n, replace=False)]
    g   = 1.0 / r.shape[1]
    krr = polynomial_kernel(r, r, degree=3, gamma=g, coef0=1)
    kss = polynomial_kernel(s, s, degree=3, gamma=g, coef0=1)
    krs = polynomial_kernel(r, s, degree=3, gamma=g, coef0=1)
    np.fill_diagonal(krr, 0)
    np.fill_diagonal(kss, 0)
    return float((krr.sum()/(n*(n-1)) + kss.sum()/(n*(n-1)) - 2*krs.mean()) * 1000)


def _build_inception():
    inc = inception_v3(pretrained=True, aux_logits=True, transform_input=False).to(DEVICE)
    inc.fc = nn.Identity()
    inc.AuxLogits = None
    inc.eval()
    hook_list = []
    def hook(m, i, o):
        hook_list.append(o.detach().flatten(1).cpu().numpy())
    h = inc.avgpool.register_forward_hook(hook)
    return inc, hook_list, h


def compute_fid(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    """Pooled FID/KID across all rare classes."""
    print("\nComputing pooled FID / KID...")
    inc, hook_list, h = _build_inception()

    real_rare  = real_df[real_df["label"].isin(_config.RARE_CLASSES)]
    synth_rare = synth_df[synth_df["label"].isin(_config.RARE_CLASSES)]
    fr = _fid_features(real_rare,  IMAGE_ROOT_DIR, inc, hook_list)
    fs = _fid_features(synth_rare, OUTPUT_DIR,     inc, hook_list)

    h.remove()
    del inc
    torch.cuda.empty_cache()

    if fr is None or fs is None:
        print("  FID: insufficient features")
        return None, None

    fid = _frechet(fr, fs)
    kid = _kid(fr, fs)
    print(f"  Pooled FID      = {fid:.2f}  (n_real={len(fr)}, n_synth={len(fs)})")
    print(f"  Pooled KID×1000 = {kid:.3f}")
    return fid, kid


def compute_per_class_fid(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """
    Per-class FID and KID for every rare class.
    Returns dict: {class_idx: {"fid": float, "kid": float, "n_real": int, "n_synth": int}}
    """
    print("\nComputing per-class FID / KID...")
    inc, hook_list, h = _build_inception()
    results = {}

    for cls in _config.RARE_CLASSES:
        real_cls  = real_df[real_df["label"] == cls]
        synth_cls = synth_df[synth_df["label"] == cls]
        if len(real_cls) < 5 or len(synth_cls) < 5:
            print(f"  Class {cls}: insufficient samples (real={len(real_cls)}, synth={len(synth_cls)}) — skipping")
            continue
        fr = _fid_features(real_cls,  IMAGE_ROOT_DIR, inc, hook_list)
        fs = _fid_features(synth_cls, OUTPUT_DIR,     inc, hook_list)
        if fr is None or fs is None:
            continue
        fid = _frechet(fr, fs) if len(fr) >= 2 and len(fs) >= 2 else float("nan")
        kid = _kid(fr, fs)     if len(fr) >= 5 and len(fs) >= 5 else float("nan")
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
        print(f"  [{cls:2d}] {name:<40}  FID={fid:7.2f}  KID={kid:7.3f}")
        results[cls] = {
            "class_name": name, "fid": fid, "kid": kid,
            "n_real": len(fr), "n_synth": len(fs),
        }

    h.remove()
    del inc
    torch.cuda.empty_cache()

    # Save
    out = RESULTS_DIR / "fid_per_class.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Per-class FID saved → {out}")
    return results


# ==============================================================================
# Core evaluation runner
# ==============================================================================

def _run_evaluation(split_csv: Path, suffix: str, label: str) -> dict:
    """
    Shared evaluation logic for val and test splits.
    suffix: '' | '_aug' | '_heavy'
    label:  human-readable label for printing
    """
    ds  = GastroVisionDataset(split_csv, "val")   # no augmentation for eval
    ldr = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=4, pin_memory=True)
    results = {}

    for name in args.models:
        try:
            model = load_checkpoint(name, suffix)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}")
            continue

        acc, yt, yp, pr = evaluate_split(model, ldr)
        _, _, f1, _     = precision_recall_fscore_support(
            yt, yp, labels=list(range(_config.NUM_CLASSES)), average=None, zero_division=0
        )
        rare_idx = [c for c in _config.RARE_CLASSES if c < _config.NUM_CLASSES]
        ece      = plot_calibration(pr, yt, name, suffix)

        print(f"\n{name}{suffix}: acc={acc:.4f}  mean_f1={f1.mean():.4f}  ECE={ece:.4f}")
        print(classification_report(yt, yp, digits=4, zero_division=0))

        results[name] = {
            "acc":      acc,
            "f1":       f1.tolist(),
            "f1_mean":  float(f1.mean()),
            "f1_rare":  float(f1[rare_idx].mean()) if rare_idx else 0.0,
            "ece":      ece,
        }
        save_gradcam(model, name, ldr, n_per_class=5, suffix=suffix)
        del model
        torch.cuda.empty_cache()

    # Ensemble
    if len(results) >= 2:
        try:
            ensemble  = ConfidenceEnsemble(args.models, suffix=suffix)
            acc_e, yt_e, yp_e, pr_e = eval_ensemble(ensemble, ldr)
            _, _, f1_e, _ = precision_recall_fscore_support(
                yt_e, yp_e, labels=list(range(_config.NUM_CLASSES)), average=None, zero_division=0
            )
            rare_idx  = [c for c in _config.RARE_CLASSES if c < _config.NUM_CLASSES]
            ece_e     = plot_calibration(pr_e, yt_e, "ensemble", suffix)

            print(f"\nEnsemble ({len(ensemble.models)} models): "
                  f"acc={acc_e:.4f}  mean_f1={f1_e.mean():.4f}  ECE={ece_e:.4f}")
            print(classification_report(yt_e, yp_e, digits=4, zero_division=0))

            results["ensemble"] = {
                "acc":      acc_e,
                "f1":       f1_e.tolist(),
                "f1_mean":  float(f1_e.mean()),
                "f1_rare":  float(f1_e[rare_idx].mean()) if rare_idx else 0.0,
                "ece":      ece_e,
                "n_models": len(ensemble.models),
                "models":   list(ensemble.models.keys()),
            }

            # Confusion matrix
            cm = confusion_matrix(yt_e, yp_e)
            fig, ax = plt.subplots(figsize=(18, 16))
            sns.heatmap(
                cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8),
                annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES[:_config.NUM_CLASSES],
                yticklabels=CLASS_NAMES[:_config.NUM_CLASSES],
            )
            ax.set_title(f"Ensemble — normalised CM ({label})")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"confusion_matrix_ensemble{suffix}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()

            del ensemble
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Ensemble failed: {e}")

    return results


def evaluate_all(augmented: bool = False):
    """S1 (baseline) or S3 (SD synthetic augmented) evaluation on validation set."""
    suffix = "_aug" if augmented else ""
    label  = "S3: SD synthetic" if augmented else "S1: real only"
    print(f"\n{'='*65}\nEvaluation — {label}\n{'='*65}")

    results = _run_evaluation(SPLITS_DIR / args.val_csv, suffix, label)

    # FID for augmented
    if augmented:
        synth_csv = OUTPUT_DIR / args.synth_dir / "synthetic_train.csv"
        if synth_csv.exists():
            real_df  = pd.read_csv(SPLITS_DIR / args.train_csv)
            synth_df = pd.read_csv(synth_csv)
            fid, kid = compute_fid(real_df, synth_df)
            results["_fid_pooled"] = fid
            results["_kid_pooled"] = kid
            per_class = compute_per_class_fid(real_df, synth_df)
            results["_fid_per_class"] = per_class

    out = RESULTS_DIR / f"eval_results{suffix}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")
    return results


def evaluate_heavy_aug():
    """S2 (heavy augmentation) evaluation on validation set."""
    print(f"\n{'='*65}\nEvaluation — S2: heavy augmentation\n{'='*65}")
    results = _run_evaluation(SPLITS_DIR / args.val_csv, "_heavy", "S2: heavy aug")
    out = RESULTS_DIR / "eval_results_heavy.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")
    return results


def evaluate_test():
    """
    Final held-out test evaluation.
    Call this ONCE, only after all model selection is complete.
    Reports results for all three strategies (S1, S2, S3) on the test set.
    """
    print(f"\n{'='*65}")
    print("FINAL TEST SET EVALUATION")
    print("Call this only once — test set is the unseen held-out split.")
    print(f"{'='*65}")

    test_csv = SPLITS_DIR / args.test_csv
    if not test_csv.exists():
        print("  Test CSV not found — run create_splits() first")
        return {}

    all_results = {}
    for suffix, label in [("", "S1"), ("_heavy", "S2"), ("_aug", "S3")]:
        ckpts_exist = any(
            (CKPT_DIR / f"sota_{n}{suffix}.pt").exists() for n in args.models
        )
        if not ckpts_exist:
            print(f"  No checkpoints for {label} — skipping")
            continue
        res = _run_evaluation(test_csv, suffix, f"{label} TEST")
        all_results[label] = res

        out = RESULTS_DIR / f"test_results{suffix}.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Test results ({label}) saved → {out}")

    # Summary table
    print(f"\n{'Strategy':<10} {'Model':<33} {'Acc':>8}  {'Mean F1':>8}  {'Rare F1':>8}  {'ECE':>8}")
    print("-" * 80)
    for strat, res in all_results.items():
        for nm, v in res.items():
            if nm.startswith("_"):
                continue
            suffix_mark = " ◄" if nm == "ensemble" else ""
            print(f"  {strat:<8} {nm:<33} {v['acc']:>8.4f}  {v['f1_mean']:>8.4f}  "
                  f"{v['f1_rare']:>8.4f}  {v.get('ece', float('nan')):>8.4f}{suffix_mark}")
        print()

    return all_results

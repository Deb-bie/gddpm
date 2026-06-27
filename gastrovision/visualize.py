"""
visualize.py
============
All paper figures:
  - Figure 1:  Class distribution bar chart
  - Figure 3:  Real vs augmented vs synthetic comparison grid  (Q5)
  - Figure 5:  Per-class F1 grouped bar chart across strategies
  - Figure 8:  t-SNE of feature embeddings
  - Figure 9:  FID vs rare-class F1 scatter
  - Figure 10: Ablation bar charts
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageEnhance

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import config as _config
from config import (
    args, DEVICE, IMAGE_ROOT_DIR, OUTPUT_DIR, RESULTS_DIR, SPLITS_DIR,
    CLASS_NAMES, CLASS_COUNTS,
)
# _config.RARE_CLASSES, _config.ULTRA_RARE, _config.NUM_CLASSES are populated after splits — access via _config.X
from dataset import GastroVisionDataset, build_classifier_transform


# ==============================================================================
# Colour palette
# ==============================================================================

STRATEGY_COLORS = {
    "S1: Real only":    "#4878cf",
    "S2: Heavy aug":    "#d65f5f",
    "S3: SD synthetic": "#6acc65",
    "CLIP zero-shot":   "#f0a500",
    "ProtoNet":         "#b47cc7",
}

TIER_COLORS = {
    "common":      "#aec6e8",
    "rare":        "#f4a259",
    "ultra_rare":  "#c0392b",
}


# ==============================================================================
# Figure 1: Class distribution
# ==============================================================================

def plot_class_distribution(save_path=None):
    """
    Horizontal bar chart of class sizes, colour-coded by tier.
    Log scale on x-axis to show ultra-rare classes alongside common ones.
    """
    save_path = save_path or RESULTS_DIR / "fig1_class_distribution.png"

    names  = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
              for i in range(len(CLASS_COUNTS))]
    counts = [CLASS_COUNTS[i] for i in range(len(CLASS_COUNTS))]

    colors = []
    for i, c in enumerate(counts):
        if c < args.ultraRare_threshold:
            colors.append(TIER_COLORS["ultra_rare"])
        elif c < args.rare_threshold:
            colors.append(TIER_COLORS["rare"])
        else:
            colors.append(TIER_COLORS["common"])

    order  = np.argsort(counts)
    names  = [names[i]  for i in order]
    counts = [counts[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 12))
    bars = ax.barh(names, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of images (log scale)", fontsize=12)
    ax.set_title("GastroVision Class Distribution", fontsize=14, fontweight="bold")
    ax.axvline(args.ultraRare_threshold, color=TIER_COLORS["ultra_rare"],
               linestyle="--", alpha=0.7, label=f"Ultra-rare (<{args.ultraRare_threshold})")
    ax.axvline(args.rare_threshold, color=TIER_COLORS["rare"],
               linestyle="--", alpha=0.7, label=f"Rare (<{args.rare_threshold})")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure 1 saved → {save_path}")
    return save_path


# ==============================================================================
# Figure 3: Real vs Augmented vs Synthetic comparison grid  (Q5)
# ==============================================================================

def plot_comparison_grid(synth_dir=None, n_classes=None, n_cols=3, save_path=None):
    """
    Grid showing real / augmented / synthetic versions of the same class.
    Rows = rare classes; columns = [Real, S2 Augmented, S3 Synthetic].
    """
    save_path  = save_path  or RESULTS_DIR / "fig3_comparison_grid.png"
    synth_dir  = Path(synth_dir) if synth_dir else OUTPUT_DIR / args.synth_dir
    train_csv  = SPLITS_DIR / args.train_csv

    if not train_csv.exists():
        print("  train.csv not found — run create_splits() first")
        return

    df = pd.read_csv(train_csv)

    # Pick classes to display: prefer ultra-rare + a few rare
    display_classes = list(_config.ULTRA_RARE) + [c for c in _config.RARE_CLASSES if c not in _config.ULTRA_RARE]
    if n_classes:
        display_classes = display_classes[:n_classes]
    display_classes = [c for c in display_classes if len(df[df["label"] == c]) > 0]

    if not display_classes:
        print("  No rare classes found in training CSV")
        return

    n_rows = len(display_classes)
    fig    = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    gs     = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.15)

    heavy_transform = build_classifier_transform("train", heavy=True)
    inv_norm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    to_tensor = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    col_titles = ["Real", "S2: Heavy Augmented", "S3: SD Synthetic"]
    for col, title in enumerate(col_titles):
        fig.text(
            (col + 0.5) / n_cols, 1.01, title,
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )

    for row_idx, cls in enumerate(display_classes):
        cls_rows = df[df["label"] == cls]
        cls_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"

        # --- Column 0: Real image ---
        ax0 = fig.add_subplot(gs[row_idx, 0])
        real_img = None
        for _, r in cls_rows.iterrows():
            p = IMAGE_ROOT_DIR / r["image_path"]
            if p.exists():
                try:
                    real_img = Image.open(p).convert("RGB").resize(
                        (args.img_size, args.img_size)
                    )
                    break
                except Exception:
                    continue
        if real_img:
            ax0.imshow(real_img)
        else:
            ax0.imshow(np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8))
        ax0.set_ylabel(cls_name, fontsize=9, rotation=90, va="center", labelpad=5)
        ax0.axis("off")

        # --- Column 1: Heavy augmented (apply transform to the same real image) ---
        ax1 = fig.add_subplot(gs[row_idx, 1])
        if real_img:
            aug_tensor = heavy_transform(real_img)
            aug_img_np = inv_norm(aug_tensor).permute(1, 2, 0).clamp(0, 1).numpy()
            ax1.imshow(aug_img_np)
        else:
            ax1.imshow(np.zeros((args.img_size, args.img_size, 3)))
        ax1.axis("off")

        # --- Column 2: Synthetic ---
        ax2 = fig.add_subplot(gs[row_idx, 2])
        synth_cls_dir = synth_dir / str(cls)
        synth_imgs    = sorted(synth_cls_dir.glob("synth_*.png")) if synth_cls_dir.exists() else []
        if synth_imgs:
            try:
                synth_img = Image.open(synth_imgs[0]).convert("RGB").resize(
                    (args.img_size, args.img_size)
                )
                ax2.imshow(synth_img)
            except Exception:
                ax2.imshow(np.zeros((args.img_size, args.img_size, 3)))
        else:
            ax2.text(0.5, 0.5, "No synthetic\nimages found",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=8)
            ax2.imshow(np.ones((args.img_size, args.img_size, 3)))
        ax2.axis("off")

    fig.suptitle("Qualitative Comparison: Real vs Augmented vs Synthetic (Rare Classes)",
                 fontsize=14, fontweight="bold", y=1.04)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure 3 (comparison grid) saved → {save_path}")
    return save_path


# ==============================================================================
# Figure 5: Per-class F1 across strategies
# ==============================================================================

def plot_per_class_f1(save_path=None):
    """
    Grouped bar chart: per-class F1 for S1, S2, S3 ensemble.
    Rare classes highlighted with background shading.
    """
    save_path = save_path or RESULTS_DIR / "fig5_per_class_f1.png"

    strategy_files = {
        "S1: Real only":    RESULTS_DIR / "eval_results.json",
        "S2: Heavy aug":    RESULTS_DIR / "eval_results_heavy.json",
        "S3: SD synthetic": RESULTS_DIR / "eval_results_aug.json",
    }

    data = {}
    for label, path in strategy_files.items():
        if path.exists():
            with open(path) as f:
                res = json.load(f)
            if "ensemble" in res:
                data[label] = res["ensemble"]["f1"]

    if not data:
        print("  No eval results found — run evaluate_all() first")
        return

    n_cls   = max(len(v) for v in data.values())
    x       = np.arange(n_cls)
    n_strat = len(data)
    width   = 0.8 / n_strat

    fig, ax = plt.subplots(figsize=(max(18, n_cls * 0.7), 6))

    for i, (label, f1s) in enumerate(data.items()):
        f1s = f1s + [0] * (n_cls - len(f1s))
        ax.bar(x + i * width, f1s, width, label=label,
               color=STRATEGY_COLORS.get(label, f"C{i}"), alpha=0.85)

    # Shade rare classes
    for cls in _config.RARE_CLASSES:
        if cls < n_cls:
            ax.axvspan(cls - 0.45, cls + 0.45, alpha=0.08,
                       color="orange", zorder=0)
    for cls in _config.ULTRA_RARE:
        if cls < n_cls:
            ax.axvspan(cls - 0.45, cls + 0.45, alpha=0.12,
                       color="red", zorder=0)

    short_names = [n[:18] for n in CLASS_NAMES[:n_cls]]
    ax.set_xticks(x + width * (n_strat - 1) / 2)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-class F1 Score by Strategy (ensemble)\n"
                 "orange = rare, red = ultra-rare", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure 5 saved → {save_path}")
    return save_path


# ==============================================================================
# Figure 8: t-SNE of feature embeddings
# ==============================================================================

def plot_tsne(model, model_name: str, csv_path=None, save_path=None,
              n_samples: int = 1000):
    """
    t-SNE of penultimate-layer embeddings.
    Rare and ultra-rare class points are plotted with distinct markers.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  scikit-learn TSNE not available")
        return

    save_path = save_path or RESULTS_DIR / f"fig8_tsne_{model_name}.png"
    csv_path  = csv_path  or SPLITS_DIR / args.val_csv

    ds  = GastroVisionDataset(csv_path, "val")
    ldr = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    all_feats, all_labels = [], []
    model.eval()

    get_feats = getattr(model, "get_features", None)
    if get_feats is None:
        # Fallback: register hook on avgpool or norm
        feats_hook = []
        def hook_fn(m, i, o):
            feats_hook.append(o.detach().cpu())
        handle = None
        for name, mod in model.named_modules():
            if "avgpool" in name or "norm" in name:
                handle = mod.register_forward_hook(hook_fn)
                break

    with torch.no_grad():
        for xb, yb in ldr:
            if get_feats:
                f = get_feats(xb.to(DEVICE)).cpu()
            else:
                feats_hook.clear()
                _ = model(xb.to(DEVICE))
                f = feats_hook[0].flatten(1) if feats_hook else None
                if f is None:
                    continue
            all_feats.append(f)
            all_labels.append(yb)
            if sum(len(x) for x in all_feats) >= n_samples:
                break

    if handle:
        handle.remove()

    feats  = torch.cat(all_feats,  dim=0)[:n_samples].numpy()
    labels = torch.cat(all_labels, dim=0)[:n_samples].numpy()

    print(f"  Running t-SNE on {len(feats)} samples...")
    tsne    = TSNE(n_components=2, perplexity=30, random_state=args.seed, n_iter=1000)
    emb     = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap    = plt.cm.get_cmap("tab20", _config.NUM_CLASSES)

    for cls in range(_config.NUM_CLASSES):
        mask   = labels == cls
        if not mask.any():
            continue
        marker = "^" if cls in _config.ULTRA_RARE else ("s" if cls in _config.RARE_CLASSES else "o")
        size   = 80 if cls in _config.RARE_CLASSES else 30
        name   = CLASS_NAMES[cls][:15] if cls < len(CLASS_NAMES) else f"c{cls}"
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[cmap(cls)], marker=marker, s=size,
                   label=name, alpha=0.7, edgecolors="none")

    ax.set_title(f"t-SNE Feature Embeddings — {model_name}\n"
                 "▲ = ultra-rare  ■ = rare  ● = common", fontsize=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7,
              markerscale=1.5, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure 8 (t-SNE) saved → {save_path}")
    return save_path


# ==============================================================================
# Figure 9: FID vs rare-class F1 scatter
# ==============================================================================

def plot_fid_vs_f1(save_path=None):
    """
    Scatter: per-class FID (x) vs per-class F1 from S3 ensemble (y).
    Shows correlation between generation quality and classification improvement.
    """
    save_path = save_path or RESULTS_DIR / "fig9_fid_vs_f1.png"

    fid_path = RESULTS_DIR / "fid_per_class.json"
    aug_path = RESULTS_DIR / "eval_results_aug.json"

    if not fid_path.exists() or not aug_path.exists():
        print("  Per-class FID or aug results not found — run evaluate_all(augmented=True) first")
        return

    with open(fid_path)  as f: fid_data = json.load(f)
    with open(aug_path)  as f: aug_data = json.load(f)

    ens_f1 = aug_data.get("ensemble", {}).get("f1", [])
    if not ens_f1:
        print("  No ensemble F1 data in aug results")
        return

    fids, f1s, names = [], [], []
    for cls_str, v in fid_data.items():
        cls = int(cls_str)
        if cls < len(ens_f1) and not np.isnan(v["fid"]):
            fids.append(v["fid"])
            f1s.append(ens_f1[cls])
            names.append(v["class_name"][:20])

    if not fids:
        print("  Insufficient data for FID vs F1 scatter")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(fids, f1s, c=range(len(fids)), cmap="plasma", s=120, zorder=5)

    for i, name in enumerate(names):
        ax.annotate(name, (fids[i], f1s[i]),
                    fontsize=7, xytext=(5, 3), textcoords="offset points")

    # Trend line
    if len(fids) >= 3:
        z = np.polyfit(fids, f1s, 1)
        p = np.poly1d(z)
        xs = np.linspace(min(fids), max(fids), 100)
        ax.plot(xs, p(xs), "r--", alpha=0.5, label=f"Trend (slope={z[0]:.4f})")

    ax.set_xlabel("Per-class FID (lower = better generation quality)", fontsize=11)
    ax.set_ylabel("F1 Score — S3 Ensemble", fontsize=11)
    ax.set_title("FID vs Rare-Class F1: Does Generation Quality Predict Classification Gain?",
                 fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(sc, label="Class index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure 9 (FID vs F1) saved → {save_path}")
    return save_path


# ==============================================================================
# Figure 10: Ablation bar charts
# ==============================================================================

def plot_ablation_summary(save_path=None):
    """
    Side-by-side bar charts for sampling and loss function ablations.
    """
    save_path = save_path or RESULTS_DIR / "fig10_ablation.png"

    ablation_files = {
        "Sampling strategy":  RESULTS_DIR / "ablation_sampling_efficientnetv2_rw_s.json",
        "Loss function":      RESULTS_DIR / "ablation_loss_efficientnetv2_rw_s.json",
    }

    available = {k: v for k, v in ablation_files.items() if v.exists()}
    if not available:
        print("  No ablation results found — run ablation.py first")
        return

    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (title, path) in zip(axes, available.items()):
        with open(path) as f:
            data = json.load(f)
        keys    = list(data.keys())
        f1_rare = [data[k]["f1_rare"] for k in keys]
        f1_mean = [data[k]["f1_mean"] for k in keys]
        x       = np.arange(len(keys))
        ax.bar(x - 0.2, f1_mean, 0.35, label="Mean F1", color="#4878cf", alpha=0.85)
        ax.bar(x + 0.2, f1_rare, 0.35, label="Rare F1", color="#d65f5f", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=20, ha="right")
        ax.set_ylabel("F1 Score")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Ablation Study Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure 10 (ablation) saved → {save_path}")
    return save_path


# ==============================================================================
# Generate all paper figures
# ==============================================================================

def generate_all_figures(model=None, model_name: str = "efficientnetv2_rw_s"):
    """Call this once to regenerate all figures for the paper."""
    print("\nGenerating all paper figures...")
    plot_class_distribution()
    plot_comparison_grid()
    plot_per_class_f1()
    plot_fid_vs_f1()
    plot_ablation_summary()
    if model is not None:
        plot_tsne(model, model_name)
    print("\nAll figures saved to:", RESULTS_DIR)

"""
fewshot.py
==========
Few-shot learning track for ultra-rare classes (n < ultraRare_threshold).

Approaches:
  1. CLIP zero-shot      — no training required; ensemble of prompts
  2. PrototypicalNetwork — episodic training with EfficientNetV2 backbone (original)
  3. Backbone comparison — pretrained DINOv2 vs fine-tuned DINOv2 vs BiomedCLIP,
                           each with and without feature-space augmentation
  4. Frequency threshold sweep — where does ProtoNet break down as n → 1?

Key design: for (3) and (4) the backbone is completely frozen; we build
mean prototypes from support images and classify queries by nearest prototype.
No episodic training is needed for the comparison experiment — frozen backbone
quality is the variable under test.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path

import config as _config
from config import (
    args, DEVICE, RESULTS_DIR, SPLITS_DIR, IMAGE_ROOT_DIR,
    CLASS_NAMES, CLASS_PROMPTS,
)
# _config.RARE_CLASSES, _config.ULTRA_RARE, _config.NUM_CLASSES populated after splits
from dataset import GastroVisionDataset


# ==============================================================================
# Normalisation constants
# ==============================================================================

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_CLIP_MEAN     = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD      = [0.26862954, 0.26130258, 0.27577711]

_IMAGENET_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_CLIP_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(_CLIP_MEAN, _CLIP_STD),
])


# ==============================================================================
# 1. CLIP Zero-Shot
# ==============================================================================

CLIP_TEMPLATES = [
    "an endoscopy image of {}",
    "a gastroscopy photograph showing {}",
    "an endoscopic view of {}",
    "a colonoscopy image of {}",
]


def run_clip_zeroshot(split: str = "test") -> dict:
    """
    Run CLIP ViT-L/14 zero-shot classification on all rare classes.
    Uses ensemble of prompt templates to improve robustness.
    Returns per-class F1 and accuracy for rare and ultra-rare classes.
    """
    try:
        import clip
    except ImportError:
        print("  clip not installed. Run: pip install git+https://github.com/openai/CLIP.git")
        return {}

    print("\nRunning CLIP zero-shot inference...")
    model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    model.eval()

    # Build text embeddings for each class using multiple templates
    text_features_list = []
    for cls_idx in range(_config.NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
        prompts  = [t.format(cls_name) for t in CLIP_TEMPLATES]
        tokens   = clip.tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.mean(dim=0)
            feats = feats / feats.norm()
        text_features_list.append(feats)

    text_features = torch.stack(text_features_list, dim=0)  # (C, D)

    # Load split
    csv_path = SPLITS_DIR / (args.test_csv if split == "test" else args.val_csv)
    df       = pd.read_csv(csv_path)

    yt_all, yp_all = [], []
    for _, row in df.iterrows():
        try:
            img = Image.open(IMAGE_ROOT_DIR / row["image_path"]).convert("RGB")
        except Exception:
            continue
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feats = model.encode_image(img_tensor)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            sims      = (img_feats @ text_features.T).squeeze(0)
            pred      = sims.argmax().item()
        yt_all.append(int(row["label"]))
        yp_all.append(pred)

    yt = np.array(yt_all)
    yp = np.array(yp_all)

    from sklearn.metrics import f1_score, classification_report
    f1_per_class = f1_score(yt, yp, labels=list(range(_config.NUM_CLASSES)),
                            average=None, zero_division=0)
    acc          = float((yt == yp).mean())

    rare_idx  = [c for c in _config.RARE_CLASSES if c < _config.NUM_CLASSES]
    ultra_idx = [c for c in _config.ULTRA_RARE   if c < _config.NUM_CLASSES]

    results = {
        "acc":           acc,
        "f1":            f1_per_class.tolist(),
        "f1_mean":       float(f1_per_class.mean()),
        "f1_rare":       float(f1_per_class[rare_idx].mean()) if rare_idx else 0.0,
        "f1_ultra_rare": float(f1_per_class[ultra_idx].mean()) if ultra_idx else 0.0,
        "method":        "CLIP ViT-L/14 zero-shot",
        "split":         split,
    }

    print(f"\nCLIP zero-shot ({split}):  acc={acc:.4f}  "
          f"rare_f1={results['f1_rare']:.4f}  ultra_rare_f1={results['f1_ultra_rare']:.4f}")
    print(classification_report(yt, yp, zero_division=0))

    out = RESULTS_DIR / f"fewshot_clip_{split}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  CLIP results saved → {out}")

    del model
    torch.cuda.empty_cache()
    return results


# ==============================================================================
# 2. Backbone Factory
# ==============================================================================

def build_fewshot_backbone(backbone_type: str):
    """
    Returns (embed_fn, feat_dim, transform).

    embed_fn(x: Tensor [B,3,H,W]) -> Tensor [B, D]  (L2-normalised)

    backbone_type:
      "dinov2_pretrained" — ViT-B/14 from Meta, ImageNet pretrained only
      "dinov2_finetuned"  — ViT-B/14 extracted from sota_dinov2.pt (GastroVision fine-tuned)
      "biomedclip"        — BiomedCLIP ViT-B/16 from Microsoft (biomedical CLIP pretraining)
    """
    if backbone_type == "dinov2_pretrained":
        print("  Loading DINOv2 ViT-B/14 (pretrained)...")
        backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        ).to(DEVICE).eval()
        for p in backbone.parameters():
            p.requires_grad = False

        @torch.no_grad()
        def embed_fn_dino_pre(x):
            feats = backbone(x.to(DEVICE))
            return F.normalize(feats, dim=-1)

        return embed_fn_dino_pre, 768, _IMAGENET_TRANSFORM

    elif backbone_type == "dinov2_finetuned":
        ckpt_path = _config.CKPT_DIR / "sota_dinov2.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned DINOv2 checkpoint not found: {ckpt_path}. "
                "Complete the S1 training pipeline first."
            )
        print(f"  Loading fine-tuned DINOv2 from {ckpt_path.name}...")
        from models import DINOv2Classifier
        state = torch.load(ckpt_path, map_location=DEVICE)
        # Infer num_classes from classifier head weight
        _nc = None
        for _k in ("head.2.weight", "head.weight", "classifier.weight"):
            if _k in state and state[_k].ndim == 2:
                _nc = state[_k].shape[0]
                break
        if _nc is None:
            _nc = _config.NUM_CLASSES
        classifier = DINOv2Classifier(num_classes=_nc).to(DEVICE)
        classifier.load_state_dict(state)
        classifier.eval()
        backbone = classifier.backbone
        for p in backbone.parameters():
            p.requires_grad = False

        @torch.no_grad()
        def embed_fn_dino_ft(x):
            feats = backbone(x.to(DEVICE))
            return F.normalize(feats, dim=-1)

        return embed_fn_dino_ft, 768, _IMAGENET_TRANSFORM

    elif backbone_type == "biomedclip":
        print("  Loading BiomedCLIP ViT-B/16...")
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch not installed. Run: pip install open_clip_torch"
            )
        bc_model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        bc_model = bc_model.to(DEVICE).eval()
        for p in bc_model.parameters():
            p.requires_grad = False

        @torch.no_grad()
        def embed_fn_biomed(x):
            feats = bc_model.encode_image(x.to(DEVICE))
            return F.normalize(feats.float(), dim=-1)

        return embed_fn_biomed, 512, _CLIP_TRANSFORM

    else:
        raise ValueError(
            f"Unknown backbone_type '{backbone_type}'. "
            "Choose: 'dinov2_pretrained', 'dinov2_finetuned', 'biomedclip'"
        )


# ==============================================================================
# 3. Feature-Space Augmentation
# ==============================================================================

class FeatureAugmentor:
    """
    Augments support-set feature vectors to build richer class prototypes.

    Three complementary strategies applied on the unit hypersphere:
      1. Gaussian noise  — small additive perturbation (approximates intra-class
                           texture / appearance variation)
      2. Feature mixup   — linear interpolation between support examples within
                           the same class (fills the intra-class convex hull)
      3. Random scaling  — magnitude variation before re-normalisation (proxy
                           for lighting / contrast differences)

    All augmented vectors are re-normalised so distances remain meaningful
    for cosine/Euclidean prototype matching.
    """

    def __init__(
        self,
        noise_std:   float = 0.01,
        n_noise:     int   = 3,
        mixup_alpha: float = 0.3,
        n_mixup:     int   = 2,
        scale_lo:    float = 0.95,
        scale_hi:    float = 1.05,
        n_scale:     int   = 2,
    ):
        self.noise_std   = noise_std
        self.n_noise     = n_noise
        self.mixup_alpha = mixup_alpha
        self.n_mixup     = n_mixup
        self.scale_lo    = scale_lo
        self.scale_hi    = scale_hi
        self.n_scale     = n_scale

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, D] — support features for ONE class (L2-normalised)
        Returns:
            augmented: [N * (1 + n_noise + n_mixup + n_scale), D]
        """
        N     = features.shape[0]
        parts = [features]

        # 1. Gaussian noise
        for _ in range(self.n_noise):
            noise = torch.randn_like(features) * self.noise_std
            parts.append(F.normalize(features + noise, dim=-1))

        # 2. Feature mixup (only meaningful with > 1 support example)
        if N > 1:
            for _ in range(self.n_mixup):
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                idx = torch.randperm(N, device=features.device)
                mixed = lam * features + (1.0 - lam) * features[idx]
                parts.append(F.normalize(mixed, dim=-1))

        # 3. Random magnitude scaling
        for _ in range(self.n_scale):
            scale = self.scale_lo + (self.scale_hi - self.scale_lo) * torch.rand(
                N, 1, device=features.device
            )
            parts.append(F.normalize(features * scale, dim=-1))

        return torch.cat(parts, dim=0)


# ==============================================================================
# 4. Prototype Evaluation Helper  (frozen backbone — no training)
# ==============================================================================

def _load_class_images(df: pd.DataFrame, cls: int, transform) -> torch.Tensor | None:
    """Load and transform all images for a given class label from a DataFrame."""
    imgs = []
    for _, row in df[df["label"] == cls].iterrows():
        try:
            img = Image.open(IMAGE_ROOT_DIR / row["image_path"]).convert("RGB")
            imgs.append(transform(img))
        except Exception:
            continue
    return torch.stack(imgs) if imgs else None


def eval_backbone_on_rare(
    embed_fn,
    feat_dim:       int,
    transform,
    train_csv:      str,
    eval_csv:       str,
    augmentor=None,
    label:          str  = "",
    target_classes: list = None,
    n_other:        int  = 10,
) -> dict:
    """
    Build class prototypes from training support images and classify eval images
    by nearest prototype.  No episodic training — tests frozen backbone quality.

    target_classes: integer class indices to evaluate (default: _config.ULTRA_RARE)
    n_other:        number of distractor classes sampled to form the "other" prototype
    """
    if target_classes is None:
        target_classes = list(_config.ULTRA_RARE)

    train_df = pd.read_csv(train_csv)
    eval_df  = pd.read_csv(eval_csv)

    all_classes   = sorted(train_df["label"].unique().tolist())
    other_classes = [c for c in all_classes if c not in target_classes]
    if len(other_classes) > n_other:
        rng           = np.random.default_rng(42)
        other_classes = rng.choice(other_classes, size=n_other, replace=False).tolist()

    results = {}

    for cls in target_classes:
        support_imgs = _load_class_images(train_df, cls, transform)
        query_imgs   = _load_class_images(eval_df,  cls, transform)

        if support_imgs is None or query_imgs is None or len(query_imgs) == 0:
            print(f"    Class {cls}: no images — skipping")
            continue

        # Embed support → optional feature augmentation → mean prototype
        with torch.no_grad():
            s_feats = embed_fn(support_imgs.to(DEVICE))   # [N_s, D]
        if augmentor is not None:
            s_feats = augmentor(s_feats)
        target_proto = s_feats.mean(0, keepdim=True)      # [1, D]

        # Build distractor prototypes
        other_protos = []
        for oc in other_classes:
            oc_imgs = _load_class_images(train_df, oc, transform)
            if oc_imgs is None:
                continue
            with torch.no_grad():
                oc_feats = embed_fn(oc_imgs[:5].to(DEVICE))
            if augmentor is not None:
                oc_feats = augmentor(oc_feats)
            other_protos.append(oc_feats.mean(0))

        if not other_protos:
            continue

        # [0] = target class, [1..] = distractors
        prototypes = torch.cat(
            [target_proto, torch.stack(other_protos)], dim=0
        )  # [K+1, D]

        with torch.no_grad():
            q_feats = embed_fn(query_imgs.to(DEVICE))     # [N_q, D]
            dists   = torch.cdist(q_feats, prototypes)    # [N_q, K+1]
            preds   = dists.argmin(dim=-1)                # 0 = target class

        correct = int((preds == 0).sum().item())
        total   = len(query_imgs)
        recall  = correct / total

        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
        print(f"    [{cls:2d}] {name:<40}  recall={recall:.4f}  "
              f"({correct}/{total})  n_support={len(support_imgs)}")

        results[cls] = {
            "class_name": name,
            "recall":     recall,
            "n_support":  len(support_imgs),
            "n_query":    total,
            "correct":    correct,
        }

    recalls = [v["recall"] for v in results.values()]
    overall = float(np.mean(recalls)) if recalls else 0.0

    results["_meta"] = {
        "backbone":       label,
        "augmentation":   augmentor is not None,
        "overall_recall": overall,
        "n_classes":      len(recalls),
    }
    print(f"  [{label}]  overall recall = {overall:.4f}  over {len(recalls)} classes")
    return results


# ==============================================================================
# 5. Backbone Comparison Experiment
# ==============================================================================

def run_backbone_comparison(train_csv: str, val_csv: str) -> dict:
    """
    Compare three frozen backbones × two augmentation settings as ProtoNet
    feature extractors on ultra-rare GastroVision classes.

    Produces the core comparison table for the paper:
      | Backbone           | Feat-aug | Ultra-rare Recall |
      |--------------------|----------|-------------------|
      | DINOv2 pretrained  |    ✗     |       x.xx        |
      | DINOv2 pretrained  |    ✓     |       x.xx        |
      | DINOv2 fine-tuned  |    ✗     |       x.xx        |
      | DINOv2 fine-tuned  |    ✓     |       x.xx        |
      | BiomedCLIP         |    ✗     |       x.xx        |
      | BiomedCLIP         |    ✓     |       x.xx        |

    Saves: backbone_comparison.json
    """
    print("\n" + "="*65)
    print("Backbone Comparison Experiment")
    print("  DINOv2 pretrained  |  DINOv2 fine-tuned  |  BiomedCLIP")
    print("  × {no aug, feature-space aug}")
    print("="*65)

    BACKBONES = [
        ("dinov2_pretrained", "DINOv2 pretrained"),
        ("dinov2_finetuned",  "DINOv2 fine-tuned"),
        ("biomedclip",        "BiomedCLIP"),
    ]

    augmentor = FeatureAugmentor(
        noise_std=0.01, n_noise=3,
        mixup_alpha=0.3, n_mixup=2,
        n_scale=2,
    )

    all_results = {}

    for btype, bname in BACKBONES:
        try:
            embed_fn, feat_dim, transform = build_fewshot_backbone(btype)
        except (ImportError, FileNotFoundError) as e:
            print(f"\n  Skipping {bname}: {e}")
            continue

        for use_aug in (False, True):
            aug = augmentor if use_aug else None
            key = f"{bname} + feat-aug" if use_aug else bname
            print(f"\n--- {key} ---")
            res = eval_backbone_on_rare(
                embed_fn, feat_dim, transform,
                train_csv, val_csv,
                augmentor=aug,
                label=key,
            )
            all_results[key] = res

        del embed_fn
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "="*65)
    print(f"{'Backbone variant':<45}  {'Recall':>8}")
    print("-" * 57)
    for key, res in all_results.items():
        r = res.get("_meta", {}).get("overall_recall", 0.0)
        marker = " ◄ best" if r == max(
            v.get("_meta", {}).get("overall_recall", 0.0)
            for v in all_results.values()
        ) else ""
        print(f"  {key:<43}  {r:>8.4f}{marker}")
    print("="*65)

    # Serialise (int keys → str for JSON)
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {str(ck): cv for ck, cv in v.items()}

    out = RESULTS_DIR / "backbone_comparison.json"
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved → {out}")
    return all_results


# ==============================================================================
# 6. Frequency Threshold Sweep
# ==============================================================================

def run_frequency_threshold_sweep(
    train_csv:     str,
    val_csv:       str,
    thresholds:    tuple = (1, 2, 3, 5, 10, 15, 20),
    backbone_type: str   = "dinov2_finetuned",
) -> dict:
    """
    For each threshold n, evaluate ProtoNet recall on all classes with
    <= n training examples.  Reveals where few-shot performance degrades as
    support size shrinks — a direct answer to the question "at what point
    does the model fail on rare classes?"

    Uses fine-tuned DINOv2 + feature-space augmentation by default (expected
    best performer from run_backbone_comparison).

    Saves: frequency_threshold_sweep.json
    """
    print("\n" + "="*65)
    print(f"Frequency Threshold Sweep  (backbone={backbone_type})")
    print("="*65)

    try:
        embed_fn, feat_dim, transform = build_fewshot_backbone(backbone_type)
    except (ImportError, FileNotFoundError) as e:
        print(f"  Cannot run sweep: {e}")
        return {}

    augmentor    = FeatureAugmentor()
    train_df     = pd.read_csv(train_csv)
    class_counts = train_df.groupby("label").size().to_dict()

    results = {}
    for thresh in thresholds:
        target_classes = sorted(
            [int(cls) for cls, cnt in class_counts.items() if cnt <= thresh]
        )
        if not target_classes:
            print(f"  n<={thresh:2d}: no classes — skipping")
            continue

        names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls{c}"
                 for c in target_classes]
        print(f"\n  n<={thresh:2d}: {len(target_classes)} classes → {names}")

        res  = eval_backbone_on_rare(
            embed_fn, feat_dim, transform,
            train_csv, val_csv,
            augmentor=augmentor,
            label=f"n<={thresh}",
            target_classes=target_classes,
        )
        meta = res.get("_meta", {})
        results[thresh] = {
            "n_classes":     len(target_classes),
            "class_indices": target_classes,
            "class_names":   names,
            "recall":        meta.get("overall_recall", 0.0),
            "per_class":     {
                str(cls): res[cls]
                for cls in target_classes if cls in res
            },
        }
        print(f"  n<={thresh:2d}  recall = {results[thresh]['recall']:.4f}")

    del embed_fn
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*65)
    print(f"{'Threshold':<12}  {'# classes':>10}  {'Recall':>8}")
    print("-" * 34)
    for t in sorted(results.keys()):
        v = results[t]
        print(f"  n <= {t:<6}  {v['n_classes']:>10}  {v['recall']:>8.4f}")
    print("="*65)

    out = RESULTS_DIR / "frequency_threshold_sweep.json"
    with open(out, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"  Saved → {out}")
    return results


# ==============================================================================
# 7. Prototypical Network (original — kept for backward compatibility)
# ==============================================================================

class PrototypicalNetwork(nn.Module):
    """
    Prototypical network using a frozen pretrained backbone as feature extractor.
    At inference, class prototypes are the mean embedding of support images.
    """

    def __init__(self, backbone_name: str = "efficientnetv2_rw_s", feat_dim: int = 1408):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.feat_dim = self.backbone.num_features
        # Learnable projection head
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = True

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return F.normalize(self.proj(feats), dim=-1)

    def forward(self, support_imgs, support_labels, query_imgs):
        """
        support_imgs:   (S, C, H, W)
        support_labels: (S,)
        query_imgs:     (Q, C, H, W)
        Returns logits (Q, n_classes) as negative distance to prototypes.
        """
        s_emb = self.embed(support_imgs)
        q_emb = self.embed(query_imgs)

        classes = support_labels.unique()
        proto   = torch.stack([
            s_emb[support_labels == c].mean(0) for c in classes
        ])

        dists  = torch.cdist(q_emb, proto)
        logits = -dists
        return logits, classes


class EpisodicDataset(Dataset):
    """
    Samples N-way K-shot episodes from the training set.
    Used exclusively for prototypical network episodic training.
    """

    def __init__(self, csv_path: str, n_way: int = 10, k_shot: int = 5,
                 q_query: int = 10, n_episodes: int = 1000):
        self.df         = pd.read_csv(csv_path)
        self.n_way      = n_way
        self.k_shot     = k_shot
        self.q_query    = q_query
        self.n_episodes = n_episodes

        self.class_to_paths = {}
        for cls, grp in self.df.groupby("label"):
            self.class_to_paths[int(cls)] = grp["image_path"].tolist()

        self.valid_classes = [
            c for c, paths in self.class_to_paths.items()
            if len(paths) >= k_shot + q_query
        ]

        self.transform = T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def __len__(self):
        return self.n_episodes

    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(IMAGE_ROOT_DIR / path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        rng     = np.random.default_rng(idx)
        classes = rng.choice(
            self.valid_classes,
            size=min(self.n_way, len(self.valid_classes)),
            replace=False,
        )
        support_imgs, support_labels = [], []
        query_imgs,   query_labels   = [], []

        for local_idx, cls in enumerate(classes):
            paths  = self.class_to_paths[cls]
            chosen = rng.choice(paths, size=self.k_shot + self.q_query, replace=False)
            for p in chosen[:self.k_shot]:
                support_imgs.append(self._load(p))
                support_labels.append(local_idx)
            for p in chosen[self.k_shot:]:
                query_imgs.append(self._load(p))
                query_labels.append(local_idx)

        return (
            torch.stack(support_imgs),
            torch.tensor(support_labels),
            torch.stack(query_imgs),
            torch.tensor(query_labels),
        )


def train_prototypical(train_csv: str, val_csv: str,
                       n_episodes: int = 2000, n_way: int = 10,
                       k_shot: int = 5, q_query: int = 10,
                       backbone: str = "efficientnetv2_rw_s") -> PrototypicalNetwork:
    """Train the prototypical network with episodic training."""
    print(f"\nTraining Prototypical Network ({n_way}-way {k_shot}-shot)...")
    model = PrototypicalNetwork(backbone_name=backbone).to(DEVICE)
    opt   = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    dataset = EpisodicDataset(train_csv, n_way=n_way, k_shot=k_shot,
                              q_query=q_query, n_episodes=n_episodes)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    best_loss = float("inf")
    ckpt_path = RESULTS_DIR / "proto_net.pt"

    for ep, (s_imgs, s_lbls, q_imgs, q_lbls) in enumerate(loader):
        s_imgs = s_imgs.squeeze(0).to(DEVICE)
        s_lbls = s_lbls.squeeze(0).to(DEVICE)
        q_imgs = q_imgs.squeeze(0).to(DEVICE)
        q_lbls = q_lbls.squeeze(0).to(DEVICE)

        model.train()
        logits, _ = model(s_imgs, s_lbls, q_imgs)
        loss = F.cross_entropy(logits, q_lbls)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (ep + 1) % 200 == 0:
            acc = (logits.argmax(1) == q_lbls).float().mean().item()
            print(f"  Episode {ep+1}/{n_episodes}  loss={loss.item():.4f}  ep_acc={acc:.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), ckpt_path)

    print(f"  ProtoNet trained. Best loss={best_loss:.4f}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


def eval_prototypical_on_rare(proto_model: PrototypicalNetwork,
                              train_csv: str, test_csv: str) -> dict:
    """
    Evaluate trained ProtoNet on ultra-rare classes.
    Support = all training images; query = eval images.
    Saves to proto_eval_val.json (filename checked by main.py restart guard).
    """
    print("\nEvaluating Prototypical Network on ultra-rare classes...")

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    def load_class_images(df, cls):
        imgs = []
        for _, row in df[df["label"] == cls].iterrows():
            try:
                img = Image.open(IMAGE_ROOT_DIR / row["image_path"]).convert("RGB")
                imgs.append(transform(img))
            except Exception:
                continue
        return torch.stack(imgs) if imgs else None

    proto_model.eval()
    results = {}

    for cls in _config.ULTRA_RARE:
        support = load_class_images(train_df, cls)
        queries = load_class_images(test_df,  cls)
        if support is None or queries is None or len(queries) == 0:
            print(f"  Class {cls}: no support or query images")
            continue

        support_lbls = torch.zeros(len(support), dtype=torch.long)

        # Sample distractor classes
        other_imgs = []
        for other_cls in np.random.choice(
            [c for c in range(_config.NUM_CLASSES) if c != cls], size=5, replace=False
        ):
            other_support = load_class_images(train_df, other_cls)
            if other_support is not None:
                other_imgs.append(other_support[:3])

        if other_imgs:
            other_stack = torch.cat(other_imgs, dim=0)
            other_lbls  = torch.ones(len(other_stack), dtype=torch.long)
            all_support = torch.cat([support, other_stack], dim=0).to(DEVICE)
            all_lbls    = torch.cat([support_lbls, other_lbls]).to(DEVICE)
        else:
            all_support = support.to(DEVICE)
            all_lbls    = support_lbls.to(DEVICE)

        with torch.no_grad():
            logits, _ = proto_model(all_support, all_lbls, queries.to(DEVICE))
            preds     = logits.argmax(1)

        correct = (preds == 0).sum().item()
        total   = len(queries)
        recall  = correct / total

        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
        print(f"  [{cls:2d}] {name:<40}  recall={recall:.4f}  ({correct}/{total})")
        results[cls] = {
            "class_name": name,
            "recall":     recall,
            "n_support":  len(support),
            "n_query":    total,
        }

    # Save with the filename that main.py checks for on restart
    out = RESULTS_DIR / "proto_eval_val.json"
    with open(out, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"  Prototypical results saved → {out}")
    return results

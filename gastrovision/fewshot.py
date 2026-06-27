"""
fewshot.py
==========
Few-shot learning track for ultra-rare classes (n < ultraRare_threshold).

Two approaches:
  1. CLIP zero-shot — no training required, uses class prompts directly
  2. PrototypicalNetwork — episode-based training with frozen backbone features
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path

import config as _config
from config import (
    args, DEVICE, RESULTS_DIR, SPLITS_DIR, IMAGE_ROOT_DIR,
    CLASS_NAMES, CLASS_PROMPTS,
)
# _config.RARE_CLASSES, _config.ULTRA_RARE, _config.NUM_CLASSES are populated after splits — access via _config.X
from dataset import GastroVisionDataset


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

    rare_idx      = [c for c in _config.RARE_CLASSES  if c < _config.NUM_CLASSES]
    ultra_idx     = [c for c in _config.ULTRA_RARE    if c < _config.NUM_CLASSES]

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
# 2. Prototypical Network
# ==============================================================================

class PrototypicalNetwork(nn.Module):
    """
    Prototypical network using a frozen pretrained backbone as feature extractor.
    At inference, class prototypes are the mean embedding of support images.
    """

    def __init__(self, backbone_name: str = "efficientnetv2_rw_s", feat_dim: int = 1408):
        super().__init__()
        import timm
        self.backbone  = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.feat_dim  = self.backbone.num_features
        # Learnable projection head
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        # Freeze backbone, only train projection
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = True

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return F.normalize(self.proj(feats), dim=-1)

    def forward(self, support_imgs, support_labels, query_imgs):
        """
        support_imgs:   (S, C, H, W) — support set
        support_labels: (S,)         — class indices for each support image
        query_imgs:     (Q, C, H, W) — query images to classify

        Returns logits (Q, n_classes) based on negative distance to prototypes.
        """
        s_emb = self.embed(support_imgs)   # (S, D)
        q_emb = self.embed(query_imgs)     # (Q, D)

        classes   = support_labels.unique()
        n_classes = len(classes)
        proto     = torch.stack([
            s_emb[support_labels == c].mean(0) for c in classes
        ])  # (n_classes, D)

        dists  = torch.cdist(q_emb, proto)         # (Q, n_classes)
        logits = -dists                             # higher = closer = more likely
        return logits, classes


class EpisodicDataset(Dataset):
    """
    Samples N-way K-shot episodes from the training set.
    Used exclusively for prototypical network training.
    """

    def __init__(self, csv_path: str, n_way: int = 10, k_shot: int = 5,
                 q_query: int = 10, n_episodes: int = 1000):
        import torchvision.transforms as T
        self.df = pd.read_csv(csv_path)
        self.n_way     = n_way
        self.k_shot    = k_shot
        self.q_query   = q_query
        self.n_episodes = n_episodes

        # Group by label
        self.class_to_paths = {}
        for cls, grp in self.df.groupby("label"):
            self.class_to_paths[int(cls)] = grp["image_path"].tolist()

        # Only keep classes with enough images
        self.valid_classes = [
            c for c, paths in self.class_to_paths.items()
            if len(paths) >= k_shot + q_query
        ]

        self.transform = T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.n_episodes

    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(IMAGE_ROOT_DIR / path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        rng      = np.random.default_rng(idx)
        classes  = rng.choice(self.valid_classes,
                              size=min(self.n_way, len(self.valid_classes)),
                              replace=False)
        support_imgs, support_labels = [], []
        query_imgs,   query_labels   = [], []

        for local_idx, cls in enumerate(classes):
            paths   = self.class_to_paths[cls]
            chosen  = rng.choice(paths, size=self.k_shot + self.q_query, replace=False)
            s_paths = chosen[:self.k_shot]
            q_paths = chosen[self.k_shot:]
            for p in s_paths:
                support_imgs.append(self._load(p))
                support_labels.append(local_idx)
            for p in q_paths:
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
    model    = PrototypicalNetwork(backbone_name=backbone).to(DEVICE)
    opt      = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    dataset  = EpisodicDataset(train_csv, n_way=n_way, k_shot=k_shot,
                               q_query=q_query, n_episodes=n_episodes)
    loader   = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    best_loss = float("inf")
    ckpt_path = RESULTS_DIR / "proto_net.pt"

    for ep, (s_imgs, s_lbls, q_imgs, q_lbls) in enumerate(loader):
        # Squeeze batch dim (batch_size=1 for episodic)
        s_imgs  = s_imgs.squeeze(0).to(DEVICE)
        s_lbls  = s_lbls.squeeze(0).to(DEVICE)
        q_imgs  = q_imgs.squeeze(0).to(DEVICE)
        q_lbls  = q_lbls.squeeze(0).to(DEVICE)

        model.train()
        logits, classes = model(s_imgs, s_lbls, q_imgs)
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

    print(f"  Prototypical Network trained. Best loss={best_loss:.4f}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


def eval_prototypical_on_rare(proto_model: PrototypicalNetwork,
                              train_csv: str, test_csv: str) -> dict:
    """
    Evaluate prototypical network on ultra-rare classes.
    Support set = all training images for that class.
    Query set   = test images for that class.
    """
    print("\nEvaluating Prototypical Network on ultra-rare classes...")
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def load_class_images(df: pd.DataFrame, cls: int):
        imgs = []
        for _, row in df[df["label"] == cls].iterrows():
            try:
                img = Image.open(IMAGE_ROOT_DIR / row["image_path"]).convert("RGB")
                imgs.append(transform(img))
            except Exception:
                continue
        return torch.stack(imgs) if imgs else None

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    proto_model.eval()
    results = {}

    for cls in _config.ULTRA_RARE:
        support = load_class_images(train_df, cls)
        queries = load_class_images(test_df,  cls)
        if support is None or queries is None or len(queries) == 0:
            print(f"  Class {cls}: no support or query images")
            continue

        support_lbls = torch.zeros(len(support), dtype=torch.long)

        # Build "other" class prototype from remaining train classes
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

        # Class 0 = target class; count correct
        true_cls_preds = (preds == 0).sum().item()
        total          = len(queries)
        recall         = true_cls_preds / total

        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
        print(f"  [{cls:2d}] {name:<40}  recall={recall:.4f}  ({true_cls_preds}/{total})")
        results[cls] = {
            "class_name": name,
            "recall":     recall,
            "n_support":  len(support),
            "n_query":    total,
        }

    out = RESULTS_DIR / "fewshot_proto.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Prototypical results saved → {out}")
    return results

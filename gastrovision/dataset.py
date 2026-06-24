"""
dataset.py
==========
Dataset classes, data splits, and sampling utilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

from config import (
    args, IMAGE_ROOT_DIR, OUTPUT_DIR, SPLITS_DIR,
    CLASS_MAP, LABEL_MAP, REV_LABEL_MAP, NUM_CLASSES,
)


# ==============================================================================
# Split creation
# ==============================================================================

def create_splits():
    """Stratified 80/10/10 train/val/test split from raw folder structure."""
    import config  # to update globals

    raw_dir = IMAGE_ROOT_DIR
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for class_folder in sorted(raw_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        if class_name not in CLASS_MAP:
            print(f"  WARNING: {repr(class_name)} not in CLASS_MAP — skipping")
            continue
        original_label = CLASS_MAP[class_name]
        images = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
            images.extend(class_folder.glob(ext))
        for img_path in images:
            rows.append({
                "image_path": str(img_path.relative_to(raw_dir)),
                "original_label": original_label,
                "class_name": class_name,
            })
        print(f"  [{original_label:2d}] {class_name:<50} {len(images):>4} images")

    df = pd.DataFrame(rows)
    print(f"\nTotal: {len(df)} images, {df['original_label'].nunique()} classes")

    unique_labels = sorted(df["original_label"].unique())
    config.LABEL_MAP     = {orig: i for i, orig in enumerate(unique_labels)}
    config.REV_LABEL_MAP = {i: orig for orig, i in config.LABEL_MAP.items()}
    config.NUM_CLASSES   = len(unique_labels)
    df["label"] = df["original_label"].map(config.LABEL_MAP)

    train_rows, val_rows, test_rows = [], [], []
    for class_id, class_df in df.groupby("label"):
        class_df = class_df.sample(frac=1, random_state=args.seed)
        n = len(class_df)
        if n == 1:
            train_rows.append(class_df)
        elif n == 2:
            train_rows.append(class_df.iloc[[0]])
            val_rows.append(class_df.iloc[[1]])
        elif n < 10:
            n_train = max(1, int(0.6 * n))
            n_val   = max(1, int(0.2 * n))
            train_rows.append(class_df.iloc[:n_train])
            v = class_df.iloc[n_train:n_train + n_val]
            t = class_df.iloc[n_train + n_val:]
            if len(v) > 0: val_rows.append(v)
            if len(t) > 0: test_rows.append(t)
        else:
            tr, tmp = train_test_split(class_df, test_size=0.2, random_state=args.seed)
            v, t    = train_test_split(tmp,      test_size=0.5, random_state=args.seed)
            train_rows.append(tr)
            val_rows.append(v)
            test_rows.append(t)

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df   = pd.concat(val_rows,   ignore_index=True)
    test_df  = pd.concat(test_rows,  ignore_index=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR   / "val.csv",   index=False)
    test_df.to_csv(SPLITS_DIR  / "test.csv",  index=False)

    rare_orig = sorted([
        c for c in df["original_label"].unique()
        if len(df[df["original_label"] == c]) < args.rare_threshold
    ])
    ultra_rare_orig = sorted([
        c for c in df["original_label"].unique()
        if len(df[df["original_label"] == c]) < args.ultraRare_threshold
    ])

    config.RARE_CLASSES = sorted([config.LABEL_MAP[c] for c in rare_orig])
    config.ULTRA_RARE   = sorted([config.LABEL_MAP[c] for c in ultra_rare_orig])

    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")
    print(f"Rare classes (< {args.rare_threshold}):       {config.RARE_CLASSES}")
    print(f"Ultra-rare classes (< {args.ultraRare_threshold}): {config.ULTRA_RARE}")
    return train_df, val_df, test_df, rare_orig


# ==============================================================================
# Transforms
# ==============================================================================

_IMAGENET_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_DIFFUSION_NORM = T.Normalize([0.5] * 3, [0.5] * 3)


def build_classifier_transform(split: str, heavy: bool = False) -> T.Compose:
    if split == "train" and not heavy:
        return T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ToTensor(),
            _IMAGENET_NORM,
        ])
    if split == "train" and heavy:
        return T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.15),
            T.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
            T.RandomPerspective(distortion_scale=0.4, p=0.5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.ToTensor(),
            _IMAGENET_NORM,
        ])
    # val / test
    return T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        _IMAGENET_NORM,
    ])


def build_diffusion_transform(split: str) -> T.Compose:
    if split == "train":
        return T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            _DIFFUSION_NORM,
        ])
    return T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        _DIFFUSION_NORM,
    ])


# ==============================================================================
# Dataset classes
# ==============================================================================

class GastroVisionDataset(Dataset):
    """General classifier dataset that handles both real and synthetic images."""

    def __init__(self, csv_path, split="train", heavy=False, synth_dir_name=None):
        import config
        self.split = split
        df = pd.read_csv(csv_path)
        if "label" not in df.columns and "original_label" in df.columns:
            df["label"] = df["original_label"].map(config.LABEL_MAP)
        if {"image_path", "label"} - set(df.columns):
            raise ValueError("CSV missing image_path or label columns")

        self.imagepaths     = df["image_path"].tolist()
        self.labels         = df["label"].astype(int).tolist()
        self.class_names    = df["class_name"].tolist() if "class_name" in df.columns else None
        self.synth_dir_name = synth_dir_name or args.synth_dir
        self.transform      = build_classifier_transform(split, heavy=heavy)

    def __len__(self):
        return len(self.imagepaths)

    def _resolve_path(self, rel: str) -> Path:
        if rel.startswith(self.synth_dir_name + "/"):
            return OUTPUT_DIR / rel
        p = IMAGE_ROOT_DIR / rel
        if p.exists():
            return p
        alt = OUTPUT_DIR / rel
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Image not found: {rel}")

    def __getitem__(self, idx):
        try:
            path = self._resolve_path(self.imagepaths[idx])
            img  = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Warning: {e}")
            return torch.zeros(3, args.img_size, args.img_size), self.labels[idx]
        return self.transform(img), int(self.labels[idx])


class GastroVisionSDDataset(Dataset):
    """Dataset for Stable Diffusion LoRA fine-tuning."""

    def __init__(self, csv_path, tokenizer, size=512):
        import config
        self.df        = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        self.label_to_name = (
            dict(zip(self.df["label"].astype(int), self.df["class_name"]))
            if "class_name" in self.df.columns else {}
        )
        self._cfg = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from config import CLASS_PROMPTS, REV_LABEL_MAP
        row    = self.df.iloc[idx]
        label  = int(row["label"])
        pixel  = self.transform(
            Image.open(IMAGE_ROOT_DIR / row["image_path"]).convert("RGB")
        )
        prompt = CLASS_PROMPTS.get(
            REV_LABEL_MAP.get(label, label),
            "endoscopy photograph of gastrointestinal tissue"
        )
        tokens = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.squeeze(0)
        return {"pixel_values": pixel, "input_ids": tokens, "label": label}


# ==============================================================================
# Sampling utilities
# ==============================================================================

def get_weighted_sampler(csv_path) -> WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler that upsamples minority classes.
    Use this instead of shuffle=True in train DataLoader.
    """
    import config
    df     = pd.read_csv(csv_path)
    labels = df["label"].astype(int).tolist()
    counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w      = 1.0 / counts
    sw     = [w[l] for l in labels]
    return WeightedRandomSampler(sw, len(sw), replacement=True)

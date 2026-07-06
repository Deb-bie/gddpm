"""
track_a/generative/sd_dataset.py
=================================
Dataset-agnostic version of gastrovision/dataset.py's GastroVisionSDDataset —
takes an explicit data_dir, class_prompts dict, and domain_prefix instead of
importing them from a specific dataset's config module, so the same class
serves GastroVision, HAM10000, and PathMNIST.
"""

from pathlib import Path
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image


class DiffusionDataset(Dataset):
    """
    df must have columns [image_path, label] (image_path relative to data_dir).
    Builds the SD conditioning prompt as domain_prefix + class_prompts[label],
    falling back to a generic phrase if a label has no entry.
    """

    def __init__(self, df, data_dir, tokenizer, class_prompts: dict,
                 domain_prefix: str, size: int = 512, fallback_prompt: str = "medical image"):
        self.df            = df.reset_index(drop=True)
        self.data_dir       = Path(data_dir)
        self.tokenizer       = tokenizer
        self.class_prompts   = class_prompts
        self.domain_prefix   = domain_prefix
        self.fallback_prompt = fallback_prompt
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])
        img   = Image.open(self.data_dir / row["image_path"]).convert("RGB")
        pixel = self.transform(img)

        prompt = self.domain_prefix + self.class_prompts.get(label, self.fallback_prompt)
        tokens = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.squeeze(0)

        return {"pixel_values": pixel, "input_ids": tokens, "label": label}

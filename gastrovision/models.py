"""
models.py
=========
Model definitions: EfficientNetV2-S, Swin-V2-Base, DINOv2 ViT-B/14,
MobileNetV3 (legacy), HybridCNNTransformerV2, and model registry.
"""

import torch
import torch.nn as nn
import timm

from config import args, DEVICE, CKPT_DIR


# ==============================================================================
# Standard backbones
# ==============================================================================

def get_effnetv2_s(num_classes: int) -> nn.Module:
    return timm.create_model("efficientnetv2_rw_s", pretrained=True, num_classes=num_classes)


def get_swin_v2(num_classes: int) -> nn.Module:
    """Swin Transformer V2 — fixes resolution mismatch via log-continuous RPB."""
    return timm.create_model(
        "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        pretrained=True, num_classes=num_classes
    )


def get_mobilenetv3(num_classes: int) -> nn.Module:
    """Legacy — kept for ablation comparisons with original pipeline."""
    return timm.create_model("tf_mobilenetv3_large_minimal_100", pretrained=True, num_classes=num_classes)


class DINOv2Classifier(nn.Module):
    """
    DINOv2 ViT-B/14 with a classification head.
    Replaces MobileNetV3 as the lightweight diverse member of the ensemble.
    DINOv2 features are significantly more separable than MobileNet on medical data.
    """

    def __init__(self, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        )
        feat_dim = self.backbone.embed_dim  # 768 for ViT-B
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def freeze_backbones(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone embeddings (for few-shot / t-SNE use)."""
        return self.backbone(x)


# ==============================================================================
# Hybrid CNN-Transformer
# ==============================================================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim: int, tfm_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        d = min(cnn_dim, tfm_dim)
        self.cp  = nn.Linear(cnn_dim, d)
        self.tp  = nn.Linear(tfm_dim, d)
        self.ca1 = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.ca2 = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.n1  = nn.LayerNorm(d)
        self.n2  = nn.LayerNorm(d)
        self.out_dim = d * 2

    def forward(self, cf: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        cq = self.cp(cf).unsqueeze(1)
        tq = self.tp(tf).unsqueeze(1)
        a1, _ = self.ca1(cq, tq, tq)
        a1 = self.n1(a1.squeeze(1) + cq.squeeze(1))
        a2, _ = self.ca2(tq, cq, cq)
        a2 = self.n2(a2.squeeze(1) + tq.squeeze(1))
        return torch.cat([a1, a2], dim=-1)


class HybridCNNTransformerV2(nn.Module):
    """
    Sequential: EfficientNetV2-S feature map → Transformer encoder.
    Custom architecture contribution — foreground in ensemble ablation.
    """

    def __init__(self, num_classes: int, cnn_name: str = "efficientnetv2_rw_s",
                 transformer_dim: int = 512, depth: int = 4, heads: int = 8,
                 mlp_dim: int = 1024, dropout: float = 0.1, img_size: int = 224):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=True, features_only=True)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            last  = self.cnn(dummy)[-1]
            cout  = last.shape[1]
            self.n_tokens = last.shape[2] * last.shape[3]

        self.cnn_proj  = nn.Conv2d(cout, transformer_dim, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        self.cls_pos   = nn.Parameter(torch.randn(1, 1, transformer_dim))
        self.patch_pos = nn.Parameter(torch.randn(1, self.n_tokens, transformer_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(transformer_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B       = x.shape[0]
        proj    = self.cnn_proj(self.cnn(x)[-1])
        patches = proj.flatten(2).transpose(1, 2) + self.patch_pos
        cls     = self.cls_token.expand(B, -1, -1) + self.cls_pos
        tokens  = self.transformer(torch.cat([cls, patches], dim=1))
        return self.head(self.norm(tokens[:, 0]))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        B       = x.shape[0]
        proj    = self.cnn_proj(self.cnn(x)[-1])
        patches = proj.flatten(2).transpose(1, 2) + self.patch_pos
        cls     = self.cls_token.expand(B, -1, -1) + self.cls_pos
        tokens  = self.transformer(torch.cat([cls, patches], dim=1))
        return self.norm(tokens[:, 0])

    def freeze_backbones(self):
        for p in self.cnn.parameters():        p.requires_grad = False
        for p in self.cnn_proj.parameters():   p.requires_grad = False
        for p in self.transformer.parameters(): p.requires_grad = False
        for p in self.head.parameters():        p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ==============================================================================
# BiomedCLIP classifier
# ==============================================================================

class BiomedCLIPClassifier(nn.Module):
    """
    BiomedCLIP ViT-B/16 vision encoder with a linear classification head.

    BiomedCLIP (Microsoft, 2023) is pretrained on 15M biomedical image-text
    pairs from PubMed Central, making it a strong initialisation for medical
    imaging tasks.  Here we use it as a drop-in replacement for DINOv2 in the
    few-shot backbone comparison experiment.

    Feature dimension: 512  (BiomedCLIP image-embedding projection space)
    Requires: open_clip_torch  (pip install open_clip_torch)
    """

    def __init__(self, num_classes: int, dropout: float = 0.1):
        super().__init__()
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for BiomedCLIPClassifier. "
                "Run: pip install open_clip_torch"
            )
        self._model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        feat_dim = 512  # BiomedCLIP image-encoder output dimension
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._model.encode_image(x).float()
        return self.head(feats)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return image embeddings (for ProtoNet / t-SNE use)."""
        return self._model.encode_image(x).float()

    def freeze_backbones(self):
        for p in self._model.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ==============================================================================
# Registry
# ==============================================================================

def _make_swin_v2(n):
    try:
        return get_swin_v2(n)
    except Exception:
        # Fallback to original Swin-Base if V2 weights unavailable
        print("  Swin-V2 unavailable — falling back to Swin-Base")
        return timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=n)


MODEL_REGISTRY = {
    "efficientnetv2_rw_s":       get_effnetv2_s,
    "swin_v2":                   _make_swin_v2,
    "swin":                      lambda n: timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=n),
    "dinov2":                    lambda n: DINOv2Classifier(n),
    "biomedclip":                lambda n: BiomedCLIPClassifier(n),
    "mobile":                    get_mobilenetv3,
    "hybrid_cnn_transformer_v2": lambda n: HybridCNNTransformerV2(n),
}


def get_model(name: str) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY.keys())}")
    from config import NUM_CLASSES
    return MODEL_REGISTRY[name](NUM_CLASSES).to(DEVICE)


def load_checkpoint(model_name: str, suffix: str = "") -> nn.Module:
    """
    Load a saved checkpoint. suffix examples: '' (baseline), '_aug', '_heavy'.
    Infers NUM_CLASSES from .meta.json (preferred) or saved weights.
    """
    import json as _json
    import config
    path = CKPT_DIR / f"sota_{model_name}{suffix}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=DEVICE)

    # Infer NUM_CLASSES directly from the output layer weight in the checkpoint.
    # Check exact key names per architecture — avoids matching conv_head or
    # intermediate layers by accident.
    #   EfficientNetV2 / Swin-V2 / MobileNet : "classifier.weight"  [C, F]
    #   Swin-V2 (timm)                        : "head.weight"        [C, F]
    #   DINOv2 (LayerNorm→Drop→Linear)        : "head.2.weight"      [C, F]
    #   BiomedCLIP (LayerNorm→Drop→Linear)    : "head.2.weight"      [C, F]
    #   HybridCNNTransformerV2 (Drop→Linear)  : "head.1.weight"      [C, F]
    _nc = None
    for _key in ("classifier.weight", "head.weight", "head.1.weight", "head.2.weight"):
        if _key in state and state[_key].ndim == 2:
            _nc = state[_key].shape[0]
            break
    if _nc is not None:
        config.NUM_CLASSES = _nc
    elif config.NUM_CLASSES is None:
        raise RuntimeError(
            f"Cannot infer NUM_CLASSES from checkpoint {path.name}. "
            "Expected one of: classifier.weight, head.weight, head.1.weight, head.2.weight"
        )

    model = get_model(model_name)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {model_name}{suffix} (NUM_CLASSES={config.NUM_CLASSES}) from {path.name}")
    return model


def get_feature_extractor(model: nn.Module, model_name: str):
    """
    Returns a callable that extracts penultimate-layer features.
    Used by few-shot and t-SNE modules.
    """
    if hasattr(model, "get_features"):
        return model.get_features

    # Generic fallback: strip the classification head
    if model_name in ("efficientnetv2_rw_s",):
        def _extract(x):
            return model.forward_features(x).mean([-2, -1])
        return _extract

    if model_name in ("swin_v2", "swin"):
        def _extract(x):
            return model.forward_features(x).mean([-2, -1])
        return _extract

    # Default: forward minus head
    def _extract(x):
        with torch.no_grad():
            return model(x)  # returns logits; caller should use pre-head hooks if needed
    return _extract

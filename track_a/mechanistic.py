"""
track_a/mechanistic.py
========================
Phase 11 — quantitative mechanistic analyses replacing GradCAM-as-primary-
evidence, per your existing GastroVision paper's own framing (GradCAM there
is already "illustrative only, not causal evidence" — this keeps that
framing and adds two analyses that actually speak to dilution-vs-mismatch
directly):

  1. FeatureVarianceAnalysis — does S3 (augmented) training increase
     intra-class feature scatter relative to S1 (real-only)? More scatter
     = less coherent rare-class representations = consistent with either
     dilution or mismatch disrupting feature learning.
  2. NearestNeighbourPurity — for each synthetic image, does its nearest
     real-test-set neighbor (in feature space) share its class? Low purity
     = synthetic images are indistinguishable from the WRONG class =
     mismatch specifically (not just "some noise").

Feature extraction utilities here are the closest thing this project has
to the pasted spec's "core/features.py FeatureCache" — that module was
never part of this actual codebase (nothing in gastrovision/ or track_a/
implements it), so FeatureCache below is a fresh, minimal implementation
scoped to what this file actually needs: extract once, cache to .npz,
reuse. It is not a port of an existing class.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ==============================================================================
# FeatureCache — minimal extract-once-reuse wrapper (see module docstring)
# ==============================================================================

class FeatureCache:
    """
    Extracts penultimate-layer features for a DataFrame of images using a
    given model + its image root(s), caching to
    cache_dir/{cache_key}_features.npz (features, labels) so repeated
    analysis calls (variance at multiple n, purity at multiple ranks) don't
    re-run inference over images already seen.
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}_features.npz"

    def get_or_extract(self, cache_key: str, df: pd.DataFrame, root_dir, model,
                        get_features_fn, transform, device, force: bool = False):
        p = self._path(cache_key)
        if p.exists() and not force:
            data = np.load(p)
            return data["features"], data["labels"]

        import torch
        root_dir = Path(root_dir)
        feats, labels = [], []
        model.eval()
        with torch.no_grad():
            for _, row in df.iterrows():
                try:
                    img = Image.open(root_dir / row["image_path"]).convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(device)
                    f = get_features_fn(model, tensor).cpu().numpy().flatten()
                except Exception:
                    continue
                feats.append(f)
                labels.append(int(row["label"]))

        features = np.array(feats)
        labels = np.array(labels)
        np.savez(p, features=features, labels=labels)
        return features, labels

    def extract_synthetic_features(self, synth_df: pd.DataFrame, synth_root_dir, model,
                                    get_features_fn, transform, device, cache_key: str):
        """Same preprocessing/extraction path as real images — synthetic
        images get no special-cased transform, so purity comparisons aren't
        confounded by a preprocessing mismatch."""
        return self.get_or_extract(cache_key, synth_df, synth_root_dir, model,
                                    get_features_fn, transform, device)


# ==============================================================================
# 1. Intra-class feature variance
# ==============================================================================

class FeatureVarianceAnalysis:

    @staticmethod
    def compute_intraclass_variance(features: np.ndarray, labels: np.ndarray,
                                     class_ids: list) -> dict:
        """
        Mean pairwise cosine DISTANCE (1 - cosine similarity) among feature
        vectors sharing each label in class_ids. Returns {class_id: mean_distance}.
        Uses numpy (not torch.cdist) so this runs the same way whether or
        not a GPU is available — feature extraction is the GPU-bound step;
        this aggregation on already-extracted vectors is cheap regardless.
        """
        out = {}
        norm_feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        for cls in class_ids:
            mask = labels == cls
            n = mask.sum()
            if n < 2:
                out[cls] = float("nan")
                continue
            sub = norm_feats[mask]
            sim = sub @ sub.T
            dist = 1.0 - sim
            iu = np.triu_indices(n, k=1)
            out[cls] = float(dist[iu].mean())
        return out

    @classmethod
    def compare_strategies(cls, s1_features, s1_labels, s3_features, s3_labels,
                            rare_class_ids: list, epsilon: float = 0.01) -> pd.DataFrame:
        s1_var = cls.compute_intraclass_variance(s1_features, s1_labels, rare_class_ids)
        s3_var = cls.compute_intraclass_variance(s3_features, s3_labels, rare_class_ids)
        rows = []
        for c in rare_class_ids:
            v1, v3 = s1_var[c], s3_var[c]
            delta = (v3 - v1) if np.isfinite(v1) and np.isfinite(v3) else float("nan")
            interp = (
                "more scattered" if np.isfinite(delta) and delta > epsilon else
                "less scattered" if np.isfinite(delta) and delta < -epsilon else
                "similar"
            )
            rows.append({"class_id": c, "s1_variance": v1, "s3_variance": v3,
                         "delta_variance": delta, "interpretation": interp})
        return pd.DataFrame(rows)

    @staticmethod
    def variance_vs_n_curve(variance_by_n: dict) -> pd.DataFrame:
        """
        variance_by_n: {n: {"s1_variance": float, "s3_variance": float}} —
        already-computed per-n variances (from repeated compare_strategies
        calls at each n, one call site up); this just reshapes into the
        (n, s1_variance, s3_variance) table Figure 4 plots.
        """
        rows = [{"n": n, **vals} for n, vals in sorted(variance_by_n.items())]
        return pd.DataFrame(rows)


# ==============================================================================
# 2. Nearest-neighbour purity
# ==============================================================================

class NearestNeighbourPurity:

    @staticmethod
    def compute_purity(synthetic_features: np.ndarray, synthetic_labels: np.ndarray,
                        real_features: np.ndarray, real_labels: np.ndarray, k: int = 1) -> tuple:
        """
        For each synthetic feature vector, finds its k nearest REAL
        neighbors by cosine distance and checks label agreement.
        k=1: purity = fraction where the single nearest neighbor's label
        matches. k>1: purity = mean fraction of the k neighbors matching.
        Returns (overall_purity, {class_id: per_class_purity}).
        """
        sf = synthetic_features / (np.linalg.norm(synthetic_features, axis=1, keepdims=True) + 1e-8)
        rf = real_features / (np.linalg.norm(real_features, axis=1, keepdims=True) + 1e-8)
        sim = sf @ rf.T  # (n_synth, n_real)

        per_sample_purity = np.zeros(len(synthetic_features))
        for i in range(len(synthetic_features)):
            top_k_idx = np.argsort(-sim[i])[:k]
            matches = (real_labels[top_k_idx] == synthetic_labels[i]).mean()
            per_sample_purity[i] = matches

        overall = float(per_sample_purity.mean()) if len(per_sample_purity) else float("nan")
        per_class = {}
        for cls in np.unique(synthetic_labels):
            mask = synthetic_labels == cls
            per_class[int(cls)] = float(per_sample_purity[mask].mean()) if mask.sum() else float("nan")
        return overall, per_class

    @classmethod
    def purity_vs_rank(cls, synthetic_features_by_rank: dict, synthetic_labels_by_rank: dict,
                        real_features: np.ndarray, real_labels: np.ndarray, k: int = 1) -> pd.DataFrame:
        """synthetic_features_by_rank / _labels_by_rank: {rank: array}.
        Expected pattern: lower rank -> higher purity (less overfitting to
        the tiny reference set)."""
        rows = []
        for rank, feats in sorted(synthetic_features_by_rank.items()):
            labels = synthetic_labels_by_rank[rank]
            purity, _ = cls.compute_purity(feats, labels, real_features, real_labels, k=k)
            rows.append({"rank": rank, "purity": purity})
        return pd.DataFrame(rows)

    @classmethod
    def purity_vs_n(cls, synthetic_features_by_n: dict, synthetic_labels_by_n: dict,
                     real_features: np.ndarray, real_labels: np.ndarray, k: int = 1) -> pd.DataFrame:
        """Expected pattern: higher n (more LoRA reference images / more
        DCGAN training data) -> higher purity."""
        rows = []
        for n, feats in sorted(synthetic_features_by_n.items()):
            labels = synthetic_labels_by_n[n]
            purity, _ = cls.compute_purity(feats, labels, real_features, real_labels, k=k)
            rows.append({"n": n, "purity": purity})
        return pd.DataFrame(rows)


# ==============================================================================
# 3. GradCAM — restricted, illustrative use only (never primary evidence)
# ==============================================================================

def compute_gradcam_with_sanity_check(model, target_layer, image_tensor, target_class,
                                       device, run_sanity_check: bool = True):
    """
    WARNING: GradCAM is illustrative only. Always run with
    run_sanity_check=True and report the sanity maps alongside the main
    map — per Adebayo et al. 2018 ("Sanity Checks for Saliency Maps"),
    an unchecked saliency map can look plausible even when it is
    insensitive to the model's actual learned weights. This function
    should ONLY be used for illustrative figures (e.g. alongside the
    RQ2a synthetic-only analysis), never cited as evidence on its own for
    a mechanistic claim — that's what FeatureVarianceAnalysis and
    NearestNeighbourPurity above are for.

    Cascading randomization (Adebayo et al.): randomizes each layer's
    weights, from the OUTPUT layer back toward the input, one at a time,
    recomputing GradCAM after each randomization step. If the saliency map
    barely changes even after several layers are randomized, the original
    map was not actually sensitive to those layers' learned weights — a
    red flag that it was picking up on architecture/input structure alone.

    Returns (gradcam_map, sanity_maps_by_layer) — sanity_maps_by_layer is
    an ordered dict {layer_name: saliency_map_after_randomizing_up_to_here}.
    """
    import torch
    import copy

    def _single_gradcam(m, layer, inp, cls_idx):
        activations, gradients = [], []

        def fwd(_m, _i, o):
            activations.append(o.detach())

        def bwd(_m, _gi, go):
            gradients.append(go[0].detach())

        h_fwd = layer.register_forward_hook(fwd)
        h_bwd = layer.register_full_backward_hook(bwd)
        try:
            m.zero_grad()
            out = m(inp)
            out[0, cls_idx].backward()
        finally:
            h_fwd.remove(); h_bwd.remove()

        if not activations or not gradients:
            return None
        act, grad = activations[0], gradients[0]
        if act.ndim != 4:
            return None
        weights = grad.mean(dim=(-2, -1), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1)[0])
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()

    main_map = _single_gradcam(model, target_layer, image_tensor, target_class)
    sanity_maps = {}

    if run_sanity_check:
        # Work on a deep copy so the caller's actual model weights are
        # never touched by the randomization pass.
        model_copy = copy.deepcopy(model)
        named_modules = [
            (name, m) for name, m in model_copy.named_modules()
            if len(list(m.parameters(recurse=False))) > 0
        ]
        # Output -> input order, per Adebayo et al.
        for name, module in reversed(named_modules):
            with torch.no_grad():
                for p in module.parameters(recurse=False):
                    p.data.copy_(torch.randn_like(p) * p.data.std())
            sanity_maps[name] = _single_gradcam(model_copy, target_layer, image_tensor, target_class)

    return main_map, sanity_maps

"""
track_a/quality_metrics.py
============================
KID (Kernel Inception Distance) computation — the core measurement behind
RQ6's KID->DeltaF1 screening-correlation question.

REFERENCE SET — held-out TEST-set images, not the n-real training
subsample. This was revisited and changed after review: the original
design here compared synthetic images against the SAME n real images the
classifier trains on at that grid point (deliberately noisy at low n).
That's a defensible idea in isolation, but it conflicts with your original
GastroVision paper's Table III convention (test-set images only, "to avoid
data leakage") and with a resurfaced project spec that flagged this
explicitly. Confirmed with you: switch to test-set images as the fixed
reference, matching Table III.

Consequence worth knowing, not hiding: because the SD/LoRA arm generates
ONE fixed synthetic pool per class independent of n (domain-adapt-once +
prompt-conditioning — confirmed as the correct design, since it matches
the actual gastrovision/generate.py code), and the test set is also fixed
regardless of n, KID for the SD arm is now CONSTANT per class across the
whole n-grid by construction — it no longer varies with n at all. The
DCGAN arm still varies genuinely with n, since Phase 3 trains a fresh GAN
per (class, n) cell using only that cell's real images, so its generated
images differ from grid point to grid point even against the same fixed
test set. RQ6's correlation for the SD arm is therefore really testing
"does this class's overall synthetic-image quality predict whether/when
augmentation helps at each n" (one KID value per class, paired against a
curve of n-dependent DeltaF1 values) rather than "does KID track a
shrinking sample" — a different but still coherent question. This is
worth stating plainly in the paper rather than discovering post hoc.

Why KID, not FID: Binkowski et al. (arXiv:1801.01401) show FID is a biased
estimator whose bias is a function of sample size, so it isn't reliable at
this study's low-n grid points even with a fixed test-set reference — some
classes' test splits are themselves small. KID's polynomial-kernel MMD
estimator is unbiased at any sample size.

The KID formula (polynomial kernel, degree=3, gamma=1/dim, coef0=1,
diagonal zeroed, x1000) is copied from gastrovision/evaluate.py's existing
`_kid()`, reusing the exact formula already in the accepted pipeline.

Resumability: compute_kid_sweep() now saves to disk after EVERY (class, n)
cell and skips cells already present in the output file on a restart,
rather than computing the whole matrix in memory and writing once at the
end — KID computation over many datasets/classes/n/ranks is slow enough
that losing all progress to a killed job is a real risk, not a hypothetical
one.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import inception_v3
import torchvision.transforms as T

INCEPTION_TRANSFORM = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
])


def build_inception(device):
    inc = inception_v3(pretrained=True, aux_logits=True, transform_input=False).to(device)
    inc.fc = nn.Identity()
    inc.AuxLogits = None
    inc.eval()
    hook_list = []

    def hook(m, i, o):
        hook_list.append(o.detach().flatten(1).cpu().numpy())

    handle = inc.avgpool.register_forward_hook(hook)
    return inc, hook_list, handle


def extract_features(df: pd.DataFrame, root_dir, model, hook_list: list, device) -> np.ndarray | None:
    feats = []
    root_dir = Path(root_dir)
    for _, row in df.iterrows():
        try:
            img    = Image.open(root_dir / row["image_path"]).convert("RGB")
            tensor = INCEPTION_TRANSFORM(img).unsqueeze(0).to(device)
            hook_list.clear()
            with torch.no_grad():
                _ = model(tensor)
            if hook_list:
                feats.append(hook_list[0].flatten())
        except Exception:
            continue
    return np.array(feats) if feats else None


def kid_score(r: np.ndarray, s: np.ndarray, seed: int = 42, max_subsample: int = 500) -> float:
    """Unbiased polynomial-kernel KID (Binkowski et al.), identical formula
    to gastrovision/evaluate.py's `_kid()`."""
    from sklearn.metrics.pairwise import polynomial_kernel
    n = min(len(r), len(s), max_subsample)
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    r = r[rng.choice(len(r), n, replace=False)]
    s = s[rng.choice(len(s), n, replace=False)]
    g = 1.0 / r.shape[1]
    krr = polynomial_kernel(r, r, degree=3, gamma=g, coef0=1)
    kss = polynomial_kernel(s, s, degree=3, gamma=g, coef0=1)
    krs = polynomial_kernel(r, s, degree=3, gamma=g, coef0=1)
    np.fill_diagonal(krr, 0)
    np.fill_diagonal(kss, 0)
    return float((krr.sum() / (n * (n - 1)) + kss.sum() / (n * (n - 1)) - 2 * krs.mean()) * 1000)


def compute_kid_for_cell(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                          real_root_dir, synth_root_dir, inception_model, hook_list, device,
                          seed: int = 42) -> dict:
    """
    Generic single-cell KID between any two image sets. Callers in this
    pipeline (compute_kid_sweep below) always pass a fixed TEST-split
    real_df — this function itself doesn't enforce that (kept generic for
    reuse, e.g. by kid_screening.py's cross-validation), so enforcement
    lives at the driver level where it matters operationally.
    """
    fr = extract_features(real_df, real_root_dir, inception_model, hook_list, device)
    fs = extract_features(synth_df, synth_root_dir, inception_model, hook_list, device)
    if fr is None or fs is None:
        return {"kid": float("nan"), "n_real": 0, "n_synth": 0}
    return {"kid": kid_score(fr, fs, seed=seed), "n_real": len(fr), "n_synth": len(fs)}


def _load_existing(out_path: Path) -> dict:
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return {}


def _save(results: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(out_path)  # atomic-ish swap, avoids a half-written file on kill


def _cell_done(results: dict, cls_key: str, n_key: str) -> bool:
    cell = results.get(cls_key, {}).get(n_key)
    return cell is not None and np.isfinite(cell.get("kid", float("nan")))


def compute_kid_sweep(dataset_name: str, sweep_classes: list, test_df: pd.DataFrame, n_grid: list,
                       synth_df: pd.DataFrame, real_root_dir, synth_root_dir,
                       results_dir, device, class_names: dict = None,
                       seed: int = 42, gen_model: str = "sd_lora",
                       real_split_name: str = "test") -> dict:
    """
    Computes KID for every (class, n) cell using test_df (the dataset's
    held-out test split — SAME rows regardless of n) as the fixed real
    reference, for one generative-model arm (sd_lora or dcgan).

    `real_split_name` must be exactly "test" — this is a deliberate
    call-site attestation, not a formality: the whole point of this
    change is that the real reference must be the held-out split, never
    a training subsample, to avoid the data-leakage concern your original
    paper's Table III convention was designed around. A caller passing
    anything else raises rather than silently computing a leaky number.

    synth_df filtering:
      - sd_lora: ONE fixed pool per class (Phase 2 doesn't regenerate per
        n) — filtered by label only. KID is therefore constant per class
        across n for this arm (see module docstring).
      - dcgan: per-(class, n) pool — synth_df MUST carry an "n" column;
        rows filtered by (label == cls) & (n == n).

    Resumable: writes results_dir/kid_per_class_per_n_{gen_model}.json
    after every cell, and skips cells already present there on restart.
    """
    if real_split_name != "test":
        raise ValueError(
            f"compute_kid_sweep requires real_split_name='test' (got "
            f"{real_split_name!r}) — KID's real reference must be the "
            "held-out test split, never a training subsample. This is an "
            "explicit call-site attestation, not a default to override."
        )
    if gen_model == "dcgan" and "n" not in synth_df.columns:
        raise ValueError(
            "compute_kid_sweep(gen_model='dcgan') requires synth_df to have "
            "an 'n' column (one row block per (class, n) DCGAN checkpoint) — "
            "a DataFrame without it would silently pool every grid cell's "
            "images together into one meaningless mixed-n comparison."
        )

    out_path = Path(results_dir) / f"kid_per_class_per_n_{gen_model}.json"
    results = _load_existing(out_path)

    print(f"\nComputing KID sweep — {dataset_name} ({gen_model}), reference=test-set")
    inc = hook_list = handle = None  # lazily built only if there's actually work to do

    for cls in sweep_classes:
        cls_key = str(cls)
        results.setdefault(cls_key, {})
        real_cls_df = test_df[test_df["label"] == cls]
        real_feats_cache = None  # extracted once per class, reused across every n

        for n in n_grid:
            n_key = str(n)
            if _cell_done(results, cls_key, n_key):
                continue

            if gen_model == "dcgan":
                synth_cell = synth_df[(synth_df["label"] == cls) & (synth_df["n"] == n)]
            else:
                synth_cell = synth_df[synth_df["label"] == cls]

            if len(synth_cell) < 2 or len(real_cls_df) < 2:
                results[cls_key][n_key] = {
                    "kid": float("nan"), "n_real": len(real_cls_df), "n_synth": len(synth_cell),
                }
                _save(results, out_path)
                print(f"  Class {cls} n={n}: insufficient images "
                      f"(real_test={len(real_cls_df)}, synth={len(synth_cell)}) — recorded NaN")
                continue

            if inc is None:
                inc, hook_list, handle = build_inception(device)
            if real_feats_cache is None:
                real_feats_cache = extract_features(real_cls_df, real_root_dir, inc, hook_list, device)

            fs = extract_features(synth_cell, synth_root_dir, inc, hook_list, device)
            if real_feats_cache is None or fs is None:
                cell = {"kid": float("nan"), "n_real": 0, "n_synth": 0}
            else:
                cell = {
                    "kid": kid_score(real_feats_cache, fs, seed=seed),
                    "n_real": len(real_feats_cache), "n_synth": len(fs),
                }

            results[cls_key][n_key] = cell
            _save(results, out_path)

            name = class_names.get(cls, str(cls)) if class_names else str(cls)
            print(f"  [{cls:>3}] {name:<30} n={n:<4} "
                  f"KID={cell['kid']:.3f}  (n_real_test={cell['n_real']}, n_synth={cell['n_synth']})")

    if inc is not None:
        handle.remove()
        del inc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"KID sweep saved -> {out_path}")
    return results

"""
track_a/datasets/pathmnist.py
==============================
PathMNIST (histopathology) loader for Track A.

Source: NCT-CRC-HE-100K (Kather et al.), accessed via the `medmnist` package
at size=224 (native patch resolution — NOT the low-res 28x28 MedMNIST
default, which would be unrealistic for diffusion fine-tuning/generation).

9 tissue-type classes, none naturally rare (~9.5k-15.5k images each — see
CLASS_COUNTS below, confirmed against the published NCT-CRC-HE-100K numbers
in track_a_prior_work_review.docx Section 9 flag 2). Per that decision,
PathMNIST's role is the fully-controlled baseline: TUM (Colorectal
adenocarcinoma epithelium) and STR (Cancer-associated stroma) are the two
classes artificially subsampled through the full N_GRID, since there is no
natural scarcity to exploit. The other 7 classes are used at full size as
majority/bystander classes only — never swept.

medmnist ships its own official train/val/test split; that split is used
as-is (not re-split via track_a/split_utils) so PathMNIST results stay
comparable to any published MedMNIST v2 benchmark numbers.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = {
    0: "Adipose (ADI)",
    1: "Background (BACK)",
    2: "Debris (DEB)",
    3: "Lymphocytes (LYM)",
    4: "Mucus (MUC)",
    5: "Smooth muscle (MUS)",
    6: "Normal colon mucosa (NORM)",
    7: "Cancer-associated stroma (STR)",
    8: "Colorectal adenocarcinoma epithelium (TUM)",
}

# NCT-CRC-HE-100K published per-class counts (sum = 107,180).
CLASS_COUNTS = {
    0: 11745, 1: 11413, 2: 11851, 3: 12191, 4: 9931,
    5: 14128, 6: 9504, 7: 10867, 8: 15550,
}

DOMAIN_PREFIX = "H&E stained histopathology patch, colorectal tissue, 20x magnification: "

NEGATIVE_PROMPT = (
    "illustration, diagram, cartoon, drawing, text, watermark, "
    "endoscopy, dermoscopy, x-ray, mri, ct scan, "
    "blurry, low quality, overexposed, noisy, "
    "natural scene, person, face, outdoor"
)

CLASS_PROMPTS = {
    0: "adipose tissue, large clear vacuoles, thin cell membranes, H&E histology",
    1: "background, empty slide region, no tissue, H&E histology",
    2: "debris, necrotic tissue fragments, amorphous eosinophilic material, H&E histology",
    3: "lymphocytes, dense small dark round nuclei clusters, H&E histology",
    4: "mucus, pale pink acellular pools, H&E histology",
    5: "smooth muscle, elongated pink fibers, spindle-shaped nuclei, H&E histology",
    6: "normal colon mucosa, regular glandular architecture, uniform crypts, H&E histology",
    7: "cancer-associated stroma, dense fibrous spindle cells, disorganized architecture, H&E histology",
    8: "colorectal adenocarcinoma epithelium, irregular glands, hyperchromatic nuclei, H&E histology",
}

# No natural scarcity — TUM and STR are ARTIFICIALLY designated for the full
# n-grid sweep (Section 9 flag 2), not naturally rare.
NATURALLY_RARE = []
SWEEP_CLASSES = [8, 7]   # TUM, STR


def _is_valid_image(path: Path) -> bool:
    """Cheap integrity check (no full pixel decode) — catches the truncated/
    zero-byte files a killed process can leave behind mid-write."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _materialize_split(medmnist_ds, split_name: str, out_dir: Path) -> pd.DataFrame:
    """
    Writes each image in a medmnist split to PNG once (idempotent — skips
    files that already exist AND pass an integrity check), and returns the
    resulting image_path/label/class_name DataFrame. Materializing to disk
    keeps this loader's output schema identical to GastroVision's and
    HAM10000's (image_path column, consumed by the same downstream Dataset/
    generation code) rather than threading PIL Image objects or numpy
    arrays through a separate code path.

    Two failure modes this guards against, both observed in practice when a
    run got killed mid-materialization (OOM, node eviction, crash):
      1. A previous save() was interrupted partway through, leaving a
         corrupt/truncated file at the final path. Since existence alone
         used to be the "already done" check, that corrupt file would be
         skipped forever and fail every downstream read (PIL
         UnidentifiedImageError) on every future run. _is_valid_image()
         catches this and triggers a rewrite.
      2. THIS run gets killed mid-save. Writing to a temp path first and
         atomically renaming it into place (os.replace, same filesystem)
         means the final path only ever exists in a complete, valid state —
         a kill mid-write leaves an orphaned .tmp file, never a corrupt
         file at the real path.
    """
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    imgs   = medmnist_ds.imgs      # (N, H, W, 3) uint8
    labels = medmnist_ds.labels.squeeze(-1) if medmnist_ds.labels.ndim > 1 else medmnist_ds.labels

    rows = []
    n_repaired = 0
    for i in range(len(imgs)):
        label = int(labels[i])
        cls_dir = split_dir / str(label)
        cls_dir.mkdir(exist_ok=True)
        path = cls_dir / f"{split_name}_{i:06d}.png"

        needs_write = not path.exists()
        if not needs_write and not _is_valid_image(path):
            needs_write = True
            n_repaired += 1

        if needs_write:
            tmp_path = path.with_name(path.name + ".tmp")
            Image.fromarray(imgs[i]).save(tmp_path, format="PNG")
            tmp_path.replace(path)  # atomic on the same filesystem

        rows.append({
            "image_path": str(path.relative_to(out_dir.parent)),
            "label": label,
            "class_name": CLASS_NAMES[label],
        })

    if n_repaired:
        print(f"  {split_name}: repaired {n_repaired} corrupt/truncated image(s) from a previous interrupted run")
    return pd.DataFrame(rows)


def get_splits(data_dir, size: int = 224) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        from medmnist import PathMNIST
    except ImportError as e:
        raise ImportError(
            "medmnist package is required for the PathMNIST loader. "
            "Install with: pip install medmnist"
        ) from e

    data_dir = Path(data_dir)
    materialized_dir = data_dir / "pathmnist_materialized"

    # medmnist's own default root (~/.medmnist) auto-creates itself, but a
    # CUSTOM root (what we pass here, so downloads land on the PVC instead
    # of the ephemeral container home dir) is NOT auto-created by medmnist —
    # recent versions raise RuntimeError("Failed to setup the default `root`
    # directory...") if it doesn't already exist. Create it ourselves first.
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split_name in ("train", "val", "test"):
        ds = PathMNIST(split=split_name, download=True, size=size, root=str(data_dir))
        splits[split_name] = _materialize_split(ds, split_name, materialized_dir)
        print(f"PathMNIST {split_name}: {len(splits[split_name])} images")

    return splits["train"], splits["val"], splits["test"]

"""
track_a/datasets/ham10000.py
=============================
HAM10000 (dermoscopy) loader for Track A.

Expects the standard HAM10000 archive layout under data_dir:
    data_dir/HAM10000_metadata.csv
    data_dir/HAM10000_images_part_1/*.jpg
    data_dir/HAM10000_images_part_2/*.jpg

7 lesion classes (dx column): akiec, bcc, bkl, df, mel, nv, vasc.
df (dermatofibroma, ~115 images) and vasc (vascular lesions, ~142 images)
are HAM10000's two naturally rare classes and the only ones that can run
the full N_GRID {1,16,32,64,128,228} without hitting a feasibility wall —
see track_a_prior_work_review.docx Section 7 for the per-class arithmetic.

Splits ARE regenerated here (unlike GastroVision) via track_a/split_utils,
since there is no pre-existing HAM10000 pipeline output to stay consistent
with — this establishes the canonical Track A splits for this dataset.
"""

from pathlib import Path
import pandas as pd

from split_utils import stratified_split

# ==============================================================================
# Class map
# ==============================================================================

DX_TO_LABEL = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}

CLASS_NAMES = {
    0: "Actinic keratoses / intraepithelial carcinoma",
    1: "Basal cell carcinoma",
    2: "Benign keratosis-like lesions",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions",
}

# Reference counts from the published HAM10000 archive (ISIC 2018 / Tschandl
# et al. 2018). The loader recomputes actual counts from the metadata CSV at
# runtime — these are recorded here only as a sanity-check fallback and for
# feasibility discussion before the data is mounted.
CLASS_COUNTS = {0: 327, 1: 514, 2: 1099, 3: 115, 4: 1113, 5: 6705, 6: 142}

DOMAIN_PREFIX = "dermoscopy image, circular vignette, polarized light, skin lesion close-up: "

NEGATIVE_PROMPT = (
    "illustration, diagram, cartoon, drawing, text, watermark, "
    "endoscopy, x-ray, mri, ct scan, histology, microscopy, "
    "blurry, low quality, overexposed, noisy, "
    "natural scene, full face, outdoor"
)

CLASS_PROMPTS = {
    0: "actinic keratosis, rough scaly erythematous patch, sun-damaged skin, dermoscopy",
    1: "basal cell carcinoma, pearly nodule, arborizing telangiectasia, dermoscopy",
    2: "seborrheic keratosis, benign keratosis, waxy stuck-on lesion, dermoscopy",
    3: "dermatofibroma, firm brown-tan papule, central white scar-like patch, dermoscopy",
    4: "melanoma, irregular pigment network, asymmetric borders, multiple colors, dermoscopy",
    5: "melanocytic nevus, symmetric pigment network, uniform brown mole, dermoscopy",
    6: "vascular lesion, angioma, red-purple lacunes, dermoscopy",
}

NATURALLY_RARE = [3, 6]   # df, vasc
SWEEP_CLASSES = NATURALLY_RARE


def _find_image(image_id: str, data_dir: Path) -> Path | None:
    for sub in ("HAM10000_images_part_1", "HAM10000_images_part_2", "HAM10000_images"):
        p = data_dir / sub / f"{image_id}.jpg"
        if p.exists():
            return p
    # Fallback: search recursively (slower, only hit if the two-folder layout differs)
    matches = list(data_dir.rglob(f"{image_id}.jpg"))
    return matches[0] if matches else None


def get_splits(data_dir, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    meta = pd.read_csv(data_dir / "HAM10000_metadata.csv")

    rows = []
    missing = 0
    for _, r in meta.iterrows():
        img_path = _find_image(r["image_id"], data_dir)
        if img_path is None:
            missing += 1
            continue
        label = DX_TO_LABEL.get(r["dx"])
        if label is None:
            continue
        rows.append({
            "image_path": str(img_path.relative_to(data_dir)),
            "label": label,
            "class_name": CLASS_NAMES[label],
        })

    if missing:
        print(f"  WARNING: {missing} HAM10000 metadata rows had no matching image file")

    df = pd.DataFrame(rows)
    print(f"HAM10000: {len(df)} images across {df['label'].nunique()} classes")
    train_df, val_df, test_df = stratified_split(df, label_col="label", seed=seed)
    return train_df, val_df, test_df

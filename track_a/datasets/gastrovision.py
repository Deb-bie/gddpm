"""
track_a/datasets/gastrovision.py
=================================
GastroVision loader for Track A.

IMPORTANT — why this duplicates data instead of importing gastrovision.config:
gastrovision/config.py calls parse_args() at import time with its OWN
argparse.ArgumentParser(), which parses the full process sys.argv. If
track_a code imported that module while track_a/main.py was invoked with
track_a-specific flags (--datasets, --run_dcgan_comparison, etc.),
gastrovision's parser would choke on "unrecognized arguments" and crash on
import. Rather than patch the existing, already-validated pipeline's
argument parsing (risking behavior changes to a script that already
produced the results in gastrovision_results/), the class map, counts,
and prompts are duplicated here as plain data. They must be kept in sync
with gastrovision/config.py if that file's CLASS_MAP/CLASS_COUNTS ever
change; as of Section 9 flag 3 in the review doc, "Resection margins" is
confirmed staying a standalone class, so no sync issue currently exists.

Splits are NOT regenerated here. GastroVision's train/val/test CSVs already
exist (produced by gastrovision/dataset.py's create_splits(), consumed by
the existing eval_results*.json / test_results*.json results) and Track A
reuses them as-is, so GastroVision's real/S1/S2/S3 numbers stay directly
comparable to Track A's own results on the same splits.
"""

from pathlib import Path
import pandas as pd

# ==============================================================================
# Class map (duplicated from gastrovision/config.py — see module docstring)
# ==============================================================================

CLASS_MAP = {
    "Accessory tools": 0, "Angiectasia": 1, "Barrett's esophagus": 2,
    "Blood in lumen": 3, "Cecum": 4, "Colon diverticula": 5,
    "Colon polyps": 6, "Colorectal cancer": 7, "Duodenal bulb": 8,
    "Dyed-lifted-polyps": 9, "Dyed-resection-margins": 10, "Erythema": 11,
    "Esophageal varices": 12, "Esophagitis": 13, "Gastric polyps": 14,
    "Gastroesophageal_junction_normal z-line": 15, "Ileocecal valve": 16,
    "Mucosal inflammation large bowel": 17, "Normal esophagus": 18,
    "Normal mucosa and vascular pattern in the large bowel": 19,
    "Normal stomach": 20, "Pylorus": 21, "Resected polyps": 22,
    "Resection margins": 23, "Retroflex rectum": 24,
    "Small bowel_terminal ileum": 25, "Ulcer": 26,
}

CLASS_NAMES = {
    0: "Accessory tools", 1: "Angiectasia", 2: "Barrett's esophagus",
    3: "Blood in lumen", 4: "Cecum", 5: "Colon diverticula",
    6: "Colon polyps", 7: "Colorectal cancer", 8: "Duodenal bulb",
    9: "Dyed-lifted-polyps", 10: "Dyed-resection-margins", 11: "Erythema",
    12: "Esophageal varices", 13: "Esophagitis", 14: "Gastric polyps",
    15: "Gastroesophageal junction (z-line)", 16: "Ileocecal valve",
    17: "Mucosal inflammation LB", 18: "Normal esophagus",
    19: "Normal mucosa LB", 20: "Normal stomach", 21: "Pylorus",
    22: "Resected polyps", 23: "Resection margins", 24: "Retroflex rectum",
    25: "Small bowel/terminal ileum", 26: "Ulcer",
}

# Total (pre-split) per-class counts — from gastrovision/config.py CLASS_COUNTS.
CLASS_COUNTS = {
    0: 1266, 1: 17, 2: 95, 3: 171, 4: 113, 5: 29, 6: 820, 7: 139,
    8: 205, 9: 141, 10: 246, 11: 15, 12: 7, 13: 107, 14: 65,
    15: 330, 16: 200, 17: 29, 18: 140, 19: 1467, 20: 969, 21: 393,
    22: 92, 23: 25, 24: 67, 25: 846, 26: 6,
}

DOMAIN_PREFIX = "endoscopy photo, circular vignette, specular highlights, pink mucosa: "

NEGATIVE_PROMPT = (
    "illustration, diagram, cartoon, drawing, text, watermark, "
    "x-ray, mri, ct scan, histology, microscopy, "
    "blurry, low quality, overexposed, noisy, "
    "natural scene, person, face, outdoor"
)

CLASS_PROMPTS = {
    0:  "metal endoscopic tools, forceps or snare visible, gastroscopy",
    1:  "angiectasia, tortuous red vessels, salmon mucosa, capsule endoscopy",
    2:  "Barrett's esophagus, salmon irregular patches, lower esophagus",
    3:  "blood in lumen, dark red pooling, gastric cavity",
    4:  "cecum, pale pink mucosa, appendiceal orifice, haustral folds",
    5:  "colon diverticula, dark circular openings in colonic wall",
    6:  "colon polyp, sessile or pedunculated lesion, pink mucosa",
    7:  "colorectal cancer, irregular friable mass, ulceration, colon",
    8:  "duodenal bulb, pale smooth mucosa, circular folds",
    9:  "dyed lifted polyp, blue submucosal injection, raised lesion",
    10: "dyed resection margins, blue mucosal edges, post-polypectomy",
    11: "gastric erythema, diffuse reddish mucosal discoloration, stomach",
    12: "esophageal varices, bluish bulging veins, longitudinal, esophagus",
    13: "esophagitis, erythematous mucosa, linear erosions, esophagus",
    14: "gastric polyp, smooth rounded lesion, gastric wall",
    15: "gastroesophageal junction, z-line, squamocolumnar border",
    16: "ileocecal valve, two lips visible, cecal mucosa",
    17: "mucosal inflammation, granular friable reddish colon, lost vascular pattern",
    18: "normal esophagus, smooth pale pink mucosa, longitudinal folds",
    19: "normal colon, smooth pink mucosa, clear vascular pattern, haustrae",
    20: "normal stomach, rugal folds, pink gastric mucosa, gastric pool",
    21: "pylorus, circular orifice, antral folds, gastroscopy",
    22: "resected polyp, post-polypectomy scar, cauterized flat defect",
    23: "resection margins, cauterized edges, whitish fibrinous border",
    24: "retroflex rectum, retroflexed view, anorectal junction",
    25: "terminal ileum, pale villous mucosa, fine texture, small bowel",
    26: "gastric ulcer, mucosal crater, white fibrinous base, erythematous rim",
}

# Classes with real (not artificial) scarcity — n < 30, per the original
# rare_threshold used in gastrovision/config.py.
NATURALLY_RARE = sorted([c for c, n in CLASS_COUNTS.items() if n < 30])

# Which classes get swept across N_GRID: the naturally rare ones. Many will
# be infeasible at the higher grid points (see subsample.py) — that
# infeasibility IS the finding (Section 7 of the review doc), not a bug to
# route around.
SWEEP_CLASSES = NATURALLY_RARE


def get_splits(data_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads the EXISTING GastroVision splits produced by
    gastrovision/dataset.py's create_splits() — does not regenerate them.
    `data_dir` should point at the same OUTPUT_DIR the original pipeline
    used (i.e. /data/gastrovision on the Nautilus PVC), so
    data_dir/splits/{train,val,test}.csv are found directly.
    """
    splits_dir = Path(data_dir) / "splits"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df   = pd.read_csv(splits_dir / "val.csv")
    test_df  = pd.read_csv(splits_dir / "test.csv")

    for df in (train_df, val_df, test_df):
        if "label" not in df.columns and "original_label" in df.columns:
            df["label"] = df["original_label"]
        if "class_name" not in df.columns:
            df["class_name"] = df["label"].map(CLASS_NAMES)

    return train_df, val_df, test_df

"""
track_a/subsample.py
=====================
n-grid subsampling with per-class feasibility checking.

Two things this module guarantees that a naive `df.sample(n=n)` per grid
point would not:

1. Feasibility is explicit. If a class's real training pool has fewer than
   n images, that (class, n) cell is marked infeasible rather than silently
   over-sampling with replacement or crashing. This is what lets
   track_a/main.py skip GastroVision's ultra-rare classes at n=128/228
   without special-casing GastroVision anywhere else in the pipeline —
   see track_a_prior_work_review.docx Section 7 for why this matters
   (several GastroVision classes have fewer than 16 total images).

2. Subsamples are NESTED across the grid. Each class gets ONE fixed random
   shuffle of its available pool (seeded), and the n-sample is a prefix of
   that shuffle. So the n=16 subsample for a class is a strict subset of
   its n=32 subsample, which is a strict subset of its n=64 subsample, etc.
   Without this, differences in ΔF1 across the n-grid could be partly an
   artifact of *which* images happened to be drawn at each grid point
   rather than purely an effect of how many. Nesting removes that confound.
"""

import pandas as pd


def _class_pool(train_df: pd.DataFrame, label_col: str, class_id, seed: int) -> pd.DataFrame:
    """One fixed shuffle of a single class's available real training rows."""
    class_df = train_df[train_df[label_col] == class_id]
    return class_df.sample(frac=1, random_state=seed).reset_index(drop=True)


def subsample_at_n(train_df: pd.DataFrame, label_col: str, class_id, n: int, seed: int = 42):
    """
    Returns (subset_df, feasible, available).
    feasible is False when available < n — caller should skip that cell,
    not silently truncate to `available` and proceed as if nothing happened.
    """
    pool = _class_pool(train_df, label_col, class_id, seed)
    available = len(pool)
    feasible = available >= n
    subset = pool.iloc[:min(n, available)].copy()
    return subset, feasible, available


def build_subsample_manifest(train_df: pd.DataFrame, label_col: str, n_grid: list, seed: int = 42) -> dict:
    """
    Returns {class_id: {n: subset_df}}, containing only the (class, n) cells
    that are feasible. Every entry is a nested prefix of the same per-class
    shuffle, per the module docstring.
    """
    manifest = {}
    for cls in sorted(train_df[label_col].unique()):
        pool  = _class_pool(train_df, label_col, cls, seed)
        avail = len(pool)
        manifest[cls] = {n: pool.iloc[:n].copy() for n in n_grid if avail >= n}
    return manifest


def feasibility_table(class_counts: dict, n_grid: list) -> pd.DataFrame:
    """
    One row per class, one column per grid point, boolean feasible/infeasible.
    This is the machine-checkable version of the Section 7b bucket table —
    run it against a dataset's real CLASS_COUNTS before generation/training
    to catch grid mismatches early instead of discovering them mid-run.
    """
    rows = []
    for cls, avail in sorted(class_counts.items()):
        row = {"class": cls, "available": avail}
        for n in n_grid:
            row[f"n={n}"] = avail >= n
        rows.append(row)
    return pd.DataFrame(rows)


def print_feasibility_summary(dataset_name: str, class_counts: dict, n_grid: list,
                               class_names: dict | None = None) -> pd.DataFrame:
    table = feasibility_table(class_counts, n_grid)
    print(f"\nFeasibility — {dataset_name}  (n_grid={n_grid})")
    print("-" * (28 + 8 * len(n_grid)))
    for _, row in table.iterrows():
        name = class_names.get(row["class"], str(row["class"])) if class_names else str(row["class"])
        marks = "".join(f"{'✓' if row[f'n={n}'] else '·':>8}" for n in n_grid)
        print(f"  [{row['class']:>3}] {name:<24} (n_real={row['available']:>5}) {marks}")
    n_fully_feasible = int(table[[f"n={n}" for n in n_grid]].all(axis=1).sum())
    print(f"  {n_fully_feasible}/{len(table)} classes feasible across the full grid")
    return table

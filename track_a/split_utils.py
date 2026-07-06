"""
track_a/split_utils.py
=======================
Dataset-agnostic stratified train/val/test split, generalized from
gastrovision/dataset.py's create_splits() so all three Track A datasets
(GastroVision, HAM10000, PathMNIST) are split with identical logic. Using
one dataset's idiosyncratic split rule and a different rule for the others
would confound any cross-modality comparison (RQ1) with a splitting
artifact, so this is the single source of truth for splitting.

Same rule as the original GastroVision pipeline:
  - n == 1                 -> all 1 image to train
  - n == 2                 -> 1 train, 1 val
  - n < 10                  -> 60/20/20, at least 1 image per non-empty split
  - n >= 10                 -> 80/10/10 via sklearn train_test_split
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df: pd.DataFrame, label_col: str = "label", seed: int = 42):
    """
    df must have at least [image_path, label_col]. Any other columns
    (class_name, etc.) are carried through untouched.

    Returns (train_df, val_df, test_df), each with the same columns as df,
    row order shuffled deterministically by `seed`.
    """
    train_rows, val_rows, test_rows = [], [], []

    for _, class_df in df.groupby(label_col):
        class_df = class_df.sample(frac=1, random_state=seed)
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
            tr, tmp = train_test_split(class_df, test_size=0.2, random_state=seed)
            v, t    = train_test_split(tmp,      test_size=0.5, random_state=seed)
            train_rows.append(tr)
            val_rows.append(v)
            test_rows.append(t)

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else df.iloc[0:0].copy()
    val_df   = pd.concat(val_rows,   ignore_index=True) if val_rows   else df.iloc[0:0].copy()
    test_df  = pd.concat(test_rows,  ignore_index=True) if test_rows  else df.iloc[0:0].copy()

    return train_df, val_df, test_df


def class_counts(df: pd.DataFrame, label_col: str = "label") -> dict:
    """Per-class row counts, as a plain dict — used for feasibility checks."""
    return df[label_col].value_counts().to_dict()

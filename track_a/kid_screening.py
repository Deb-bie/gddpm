"""
track_a/kid_screening.py
==========================
Phase 10 (A8) — Contribution 3 of the paper: a practically usable KID
screening criterion. Fits the KID->DeltaF1 Spearman correlation (building
on analysis.py's per-arm version, but pooled across every dataset/backbone/
generative-model combination that has results so far) and a KID threshold
above which augmentation is more likely harmful than beneficial.

Data volume, honestly stated rather than assumed: with 3 datasets, up to
~7 sweep classes each, the 6-point n-grid, and 2 backbones, the maximum
possible number of (dataset, class, backbone, n) cells is nowhere near the
pasted spec's assumed ~400 (that count came from a 4-dataset x 5-class x
5-n x 4-rank design that no longer matches this study's actual scope).
fit_spearman/fit_threshold both report n_pairs explicitly rather than
asserting a specific sample size, and validate_threshold's docstring notes
when a leave-one-dataset-out fold is small enough that its precision/
recall estimate should be read as exploratory.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from analysis import kid_delta_f1_correlation


# ==============================================================================
# Building the merged (dataset, class, backbone, gen_model, n, KID, DeltaF1) table
# ==============================================================================

def build_merged_kid_delta_df(output_dir_root, dataset_modules: dict, backbones: list,
                               n_grid: list, gen_models=("sd_lora", "dcgan")) -> pd.DataFrame:
    """
    dataset_modules: {dataset_name: dataset_module} (track_a.datasets.get_dataset).
    Pools every available (dataset, backbone, gen_model)'s KID/DeltaF1 pairs
    (from analysis.kid_delta_f1_correlation, which already handles missing
    cells gracefully) into one long DataFrame for the screening fits below.
    """
    rows = []
    for dataset, mod in dataset_modules.items():
        results_dir = Path(output_dir_root) / dataset / "results"
        for backbone in backbones:
            for gen_model in gen_models:
                corr = kid_delta_f1_correlation(
                    results_dir, dataset, backbone, mod.SWEEP_CLASSES, n_grid, gen_model=gen_model
                )
                for pair in corr.get("pairs", []):
                    rows.append({
                        "dataset": dataset, "backbone": backbone, "gen_model": gen_model,
                        "class": pair["class"], "n": pair["n"],
                        "kid": pair["kid"], "delta_f1": pair["delta_f1"],
                    })
    return pd.DataFrame(rows)


# ==============================================================================
# Spearman fit, with optional grouping + Bonferroni correction
# ==============================================================================

def fit_spearman(df: pd.DataFrame, groupby: str = None) -> dict:
    """
    Spearman rho between kid and delta_f1. If groupby is given (e.g.
    "dataset" or "n"), computes rho separately per group and applies a
    Bonferroni correction (multiply each group's p-value by the number of
    groups tested, capped at 1.0) before flagging significance at 0.05.
    """
    from scipy.stats import spearmanr

    def _one(sub: pd.DataFrame) -> dict:
        d = sub.dropna(subset=["kid", "delta_f1"])
        if len(d) < 3:
            return {"rho": float("nan"), "p_value": float("nan"), "n_pairs": len(d)}
        rho, p = spearmanr(d["kid"], d["delta_f1"])
        return {"rho": float(rho), "p_value": float(p), "n_pairs": len(d)}

    if groupby is None:
        result = _one(df)
        result["significant"] = bool(np.isfinite(result["p_value"]) and result["p_value"] < 0.05)
        return result

    groups = {}
    n_groups = df[groupby].nunique()
    for key, sub in df.groupby(groupby):
        r = _one(sub)
        p_corrected = min(1.0, r["p_value"] * n_groups) if np.isfinite(r["p_value"]) else float("nan")
        r["p_value_bonferroni"] = p_corrected
        r["significant"] = bool(np.isfinite(p_corrected) and p_corrected < 0.05)
        if r["n_pairs"] < 25:
            r["note"] = "exploratory — fewer than 25 pairs in this group"
        groups[str(key)] = r
    return {"groupby": groupby, "n_groups": n_groups, "groups": groups}


# ==============================================================================
# Threshold fitting
# ==============================================================================

@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    n_harmful: int
    n_beneficial: int


def _precision_recall_at(df: pd.DataFrame, threshold: float) -> dict:
    harmful_actual = df["delta_f1"] < 0
    harmful_pred = df["kid"] > threshold
    tp = int((harmful_pred & harmful_actual).sum())
    fp = int((harmful_pred & ~harmful_actual).sum())
    fn = int((~harmful_pred & harmful_actual).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
          else float("nan"))
    return {"precision": precision, "recall": recall, "f1": f1}


def fit_threshold(df: pd.DataFrame, min_precision_target: float = 0.80) -> dict:
    """
    Sweeps every candidate KID threshold (each unique KID value in df) and
    finds (a) the threshold maximizing the binary harmful/beneficial
    classifier's F1, and (b) the threshold achieving >= min_precision_target
    precision with the highest recall among those that qualify (the
    "practically useful: if KID > this, augmentation hurts with 80%
    confidence" threshold from the spec). Returns both as ThresholdResult,
    plus the counts of harmful/beneficial examples in df overall.
    """
    d = df.dropna(subset=["kid", "delta_f1"])
    n_harmful = int((d["delta_f1"] < 0).sum())
    n_beneficial = int((d["delta_f1"] >= 0).sum())

    if len(d) < 4 or n_harmful == 0 or n_beneficial == 0:
        return {
            "error": "insufficient class balance to fit a threshold",
            "n_pairs": len(d), "n_harmful": n_harmful, "n_beneficial": n_beneficial,
        }

    candidates = sorted(d["kid"].unique())
    best_f1_result, best_f1 = None, -1.0
    best_precision_result, best_precision_recall = None, -1.0

    for t in candidates:
        pr = _precision_recall_at(d, t)
        if np.isfinite(pr["f1"]) and pr["f1"] > best_f1:
            best_f1 = pr["f1"]
            best_f1_result = ThresholdResult(float(t), pr["precision"], pr["recall"], pr["f1"], n_harmful, n_beneficial)
        if (np.isfinite(pr["precision"]) and pr["precision"] >= min_precision_target
                and np.isfinite(pr["recall"]) and pr["recall"] > best_precision_recall):
            best_precision_recall = pr["recall"]
            best_precision_result = ThresholdResult(float(t), pr["precision"], pr["recall"], pr["f1"], n_harmful, n_beneficial)

    return {
        "f1_maximizing": asdict(best_f1_result) if best_f1_result else None,
        f"precision_{int(min_precision_target*100)}pct": asdict(best_precision_result) if best_precision_result else None,
        "n_pairs": len(d), "n_harmful": n_harmful, "n_beneficial": n_beneficial,
    }


def validate_threshold(df: pd.DataFrame, cv_group_col: str = "dataset",
                        min_precision_target: float = 0.80) -> dict:
    """
    Leave-one-group-out CV (default: leave-one-dataset-out): for each held-
    out group, fits the F1-maximizing threshold on every OTHER group, then
    evaluates that fold-specific threshold against the held-out group's own
    data. Reports mean +/- std precision/recall across folds — a realistic
    generalization estimate, not the in-sample fit fit_threshold() alone
    would give. With only 3 datasets, this is a 3-fold CV — small enough
    that the resulting std should be read as a rough spread, not a
    precise variance estimate.
    """
    groups = df[cv_group_col].unique()
    if len(groups) < 2:
        return {"error": f"need >=2 groups in '{cv_group_col}' for CV, got {len(groups)}"}

    fold_precisions, fold_recalls, folds = [], [], []
    for held_out in groups:
        train = df[df[cv_group_col] != held_out]
        test  = df[df[cv_group_col] == held_out]
        fit = fit_threshold(train, min_precision_target=min_precision_target)
        if fit.get("f1_maximizing") is None:
            continue
        t = fit["f1_maximizing"]["threshold"]
        pr = _precision_recall_at(test.dropna(subset=["kid", "delta_f1"]), t)
        folds.append({"held_out": str(held_out), "threshold": t, **pr, "n_test": len(test)})
        if np.isfinite(pr["precision"]):
            fold_precisions.append(pr["precision"])
        if np.isfinite(pr["recall"]):
            fold_recalls.append(pr["recall"])

    return {
        "n_folds": len(folds),
        "folds": folds,
        "mean_precision": float(np.mean(fold_precisions)) if fold_precisions else float("nan"),
        "std_precision": float(np.std(fold_precisions)) if fold_precisions else float("nan"),
        "mean_recall": float(np.mean(fold_recalls)) if fold_recalls else float("nan"),
        "std_recall": float(np.std(fold_recalls)) if fold_recalls else float("nan"),
    }


# ==============================================================================
# Figures (basic implementations — figures/plot_all.py restyles these for
# publication; kept here so the analysis and its figure stay next to each
# other and can't drift apart on column names)
# ==============================================================================

def generate_screening_figure(df: pd.DataFrame, threshold: float, output_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=["kid", "delta_f1"])
    harmful = d["delta_f1"] < 0

    fig, (ax_main, ax_pr) = plt.subplots(1, 2, figsize=(11, 5))
    ax_main.scatter(d.loc[harmful, "kid"], d.loc[harmful, "delta_f1"], c="red", label="harmful", alpha=0.7)
    ax_main.scatter(d.loc[~harmful, "kid"], d.loc[~harmful, "delta_f1"], c="green", label="beneficial", alpha=0.7)
    ax_main.axvline(threshold, color="black", linestyle="--", label=f"screening threshold={threshold:.1f}")
    ax_main.axhline(0, color="grey", linestyle=":")
    ax_main.set_xlabel("KID"); ax_main.set_ylabel("Delta F1"); ax_main.legend()

    thresholds = sorted(d["kid"].unique())
    precisions = [_precision_recall_at(d, t)["precision"] for t in thresholds]
    recalls    = [_precision_recall_at(d, t)["recall"] for t in thresholds]
    ax_pr.plot(thresholds, precisions, label="precision")
    ax_pr.plot(thresholds, recalls, label="recall")
    ax_pr.axvline(threshold, color="black", linestyle="--")
    ax_pr.set_xlabel("KID threshold"); ax_pr.set_ylabel("Score"); ax_pr.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig


PRACTITIONER_FLOWCHART_MERMAID = """
flowchart TD
    A["Do you have fewer than ~30 real training images per rare class?"] -->|No| B["SD augmentation likely safe — still check KID before trusting it blindly"]
    A -->|Yes| C["Generate synthetic images and compute KID against held-out test-set images"]
    C --> D{{"Is KID above the screening threshold?"}}
    D -->|Yes| E["Do NOT use SD augmentation for this class — consider few-shot methods instead"]
    D -->|No| F["SD augmentation may help — monitor rare-class F1 carefully, don't assume it will"]
"""


def generate_practitioner_flowchart(threshold: float = None, output_path=None):
    """Returns the Mermaid source (always) and optionally writes a plain
    matplotlib rendering to output_path if given — Mermaid is the primary
    artifact since it renders correctly with zero layout guesswork."""
    mermaid = PRACTITIONER_FLOWCHART_MERMAID
    if threshold is not None:
        mermaid = mermaid.replace("the screening threshold", f"KID={threshold:.1f}")

    if output_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axis("off")
        steps = [
            "1. < ~30 real images per rare class?",
            "   NO -> augmentation likely safe (still check KID)",
            "   YES -> continue",
            "2. Compute KID vs. held-out test images",
            f"3. KID > threshold{'=' + format(threshold, '.1f') if threshold is not None else ''}?",
            "   YES -> do NOT use SD augmentation; consider few-shot methods",
            "   NO  -> augmentation may help; monitor rare-class F1 carefully",
        ]
        ax.text(0.02, 0.95, "\n".join(steps), va="top", fontsize=10, family="monospace")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return mermaid

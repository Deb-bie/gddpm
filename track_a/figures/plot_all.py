"""
track_a/figures/plot_all.py
==============================
Phase 13 — the 6 named figures the pasted spec's "Analysis and figures"
section asks for, all sharing figures/style.mplstyle:

  1. n-sweep curves        (F1 vs n, S1 vs S3, per class)            -> Fig 1
  2. ratio ablation curves (F1 vs synthetic:real ratio, at an anchor n) -> Fig 2
  3. KID screening scatter (KID vs DeltaF1 + threshold)               -> Fig 3
  4. feature variance vs n (S1 vs S3 intra-class scatter)             -> Fig 4
  5. t-SNE comparison       (real vs synthetic feature clouds)        -> Fig 5
  6. practitioner flowchart (KID decision rule)                       -> Fig 6

Every function takes already-computed data (DataFrames / dicts from
analysis.py, ratio_ablation.py / multiseed.py, kid_screening.py,
mechanistic.py) and an output_path — none of these functions run any
analysis, feature extraction, or training themselves. Same read/compute-
vs-plot split used throughout Track A: analysis.py (and friends) compute,
this module only draws. generate_all_figures() at the bottom accepts
whatever precomputed pieces are available and skips (with a printed
notice, not an error) any figure whose inputs weren't supplied — since
figures 4/5 specifically require real extracted features from a trained
checkpoint, which is a separate, expensive step this module deliberately
does not try to trigger itself.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

STYLE_PATH = Path(__file__).parent / "style.mplstyle"


def _use_style():
    plt.style.use(str(STYLE_PATH))


# ==============================================================================
# Figure 1 — n-sweep curves (S1 vs S3, per class)
# ==============================================================================

def plot_n_sweep_curves(crossover_rows: list, output_path=None, title: str = None):
    """
    crossover_rows: analysis.crossover_table()'s output — one row per class,
    each with a "curve" list of {n, f1_base, f1_aug, delta}. One subplot per
    class, F1 vs n (log-x) for S1 (real-only) and S3 (augmented), with a
    vertical marker at the class's crossover n* if one exists.
    """
    _use_style()
    n_classes = len(crossover_rows)
    if n_classes == 0:
        print("plot_n_sweep_curves: no rows given, skipping")
        return None

    ncols = min(3, n_classes)
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)

    for i, row in enumerate(crossover_rows):
        ax = axes[i // ncols][i % ncols]
        curve = row.get("curve") or []
        if not curve:
            ax.set_visible(False)
            continue
        ns = [c["n"] for c in curve]
        f1_base = [c["f1_base"] for c in curve]
        f1_aug = [c["f1_aug"] for c in curve]
        ax.plot(ns, f1_base, marker="o", label="S1 real-only")
        ax.plot(ns, f1_aug, marker="s", label="S3 augmented")
        if row.get("n_star") is not None:
            ax.axvline(row["n_star"], color="grey", linestyle=":", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("n (real images/class)")
        ax.set_ylabel("Target-class F1")
        ax.set_title(row.get("class_name", str(row.get("class", "?"))))
        ax.legend()

    for j in range(n_classes, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    plt.close(fig)
    return fig


# ==============================================================================
# Figure 2 — ratio ablation curves
# ==============================================================================

def plot_ratio_ablation_curves(curves_by_class: dict, output_path=None, n_anchor_label=None):
    """
    curves_by_class: {class_name: df}, df has a "ratio" column plus either
    "rare_f1" (single-seed, straight from ratio_ablation.RatioCurveFitter.df)
    or "mean_rare_f1"/"std_rare_f1" (multi-seed, from
    multiseed.aggregate_ratio_curves_across_seeds) — whichever is present
    is used, with error bars only in the multi-seed case.
    """
    _use_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    any_plotted = False
    for name, df in curves_by_class.items():
        if df is None or df.empty:
            continue
        any_plotted = True
        if "mean_rare_f1" in df.columns:
            yerr = df["std_rare_f1"] if "std_rare_f1" in df.columns else None
            ax.errorbar(df["ratio"], df["mean_rare_f1"], yerr=yerr, marker="o", capsize=3, label=name)
        else:
            ax.plot(df["ratio"], df["rare_f1"], marker="o", label=name)

    if not any_plotted:
        print("plot_ratio_ablation_curves: no non-empty curves given, skipping")
        plt.close(fig)
        return None

    ax.set_xlabel("Synthetic:real ratio")
    ax.set_ylabel("Target-class F1")
    title = "Synthetic:real ratio ablation"
    if n_anchor_label is not None:
        title += f" (n={n_anchor_label})"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    plt.close(fig)
    return fig


# ==============================================================================
# Figure 3 — KID screening scatter (restyled kid_screening.generate_screening_figure)
# ==============================================================================

def plot_kid_screening(df, threshold: float, output_path=None):
    """df: kid_screening.build_merged_kid_delta_df()'s output — columns
    kid, delta_f1 (plus dataset/backbone/gen_model/class/n metadata,
    ignored here)."""
    _use_style()
    d = df.dropna(subset=["kid", "delta_f1"])
    if len(d) == 0:
        print("plot_kid_screening: no valid (kid, delta_f1) pairs, skipping")
        return None
    harmful = d["delta_f1"] < 0

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(d.loc[harmful, "kid"], d.loc[harmful, "delta_f1"], label="harmful (ΔF1<0)", alpha=0.75)
    ax.scatter(d.loc[~harmful, "kid"], d.loc[~harmful, "delta_f1"], label="beneficial (ΔF1≥0)", alpha=0.75)
    ax.axvline(threshold, linestyle="--", color="black", label=f"screening threshold = {threshold:.2f}")
    ax.axhline(0, linestyle=":", color="grey")
    ax.set_xlabel("KID (×1000, vs. held-out test images)")
    ax.set_ylabel("ΔF1 (augmented − real-only)")
    ax.set_title("KID as a screening criterion for augmentation risk")
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    plt.close(fig)
    return fig


# ==============================================================================
# Figure 4 — feature variance vs n (S1 vs S3)
# ==============================================================================

def plot_feature_variance_vs_n(variance_df, output_path=None, title: str = None):
    """variance_df: mechanistic.FeatureVarianceAnalysis.variance_vs_n_curve()'s
    output — columns n, s1_variance, s3_variance."""
    _use_style()
    if variance_df is None or variance_df.empty:
        print("plot_feature_variance_vs_n: empty input, skipping")
        return None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(variance_df["n"], variance_df["s1_variance"], marker="o", label="S1 real-only")
    ax.plot(variance_df["n"], variance_df["s3_variance"], marker="s", label="S3 augmented")
    ax.set_xscale("log")
    ax.set_xlabel("n (real images/class)")
    ax.set_ylabel("Mean intra-class cosine distance")
    ax.set_title(title or "Feature scatter vs. n")
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    plt.close(fig)
    return fig


# ==============================================================================
# Figure 5 — t-SNE comparison (real vs. synthetic feature clouds)
# ==============================================================================

def plot_tsne_comparison(real_features, real_labels, synth_features, synth_labels,
                          class_names: dict = None, output_path=None,
                          perplexity: float = 30.0, seed: int = 42):
    """
    2D t-SNE projection of real vs. synthetic feature vectors, colored by
    class with real/synthetic distinguished by marker shape (circle=real,
    x=synthetic) — the qualitative counterpart to
    mechanistic.NearestNeighbourPurity's quantitative purity score.
    """
    from sklearn.manifold import TSNE

    _use_style()
    real_features = np.asarray(real_features)
    synth_features = np.asarray(synth_features)
    real_labels = np.asarray(real_labels)
    synth_labels = np.asarray(synth_labels)

    if len(real_features) == 0 or len(synth_features) == 0:
        print("plot_tsne_comparison: empty real or synthetic features, skipping")
        return None

    all_feats = np.concatenate([real_features, synth_features], axis=0)
    all_labels = np.concatenate([real_labels, synth_labels])
    is_synth = np.concatenate([
        np.zeros(len(real_features), dtype=bool),
        np.ones(len(synth_features), dtype=bool),
    ])

    n = len(all_feats)
    eff_perplexity = min(perplexity, max(5, (n - 1) // 3))
    proj = TSNE(n_components=2, perplexity=eff_perplexity, random_state=seed, init="pca").fit_transform(all_feats)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    classes = sorted(set(all_labels.tolist()))
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(classes):
        color = cmap(i % 10)
        name = (class_names or {}).get(cls, str(cls))
        mask_real = (all_labels == cls) & (~is_synth)
        mask_synth = (all_labels == cls) & is_synth
        ax.scatter(proj[mask_real, 0], proj[mask_real, 1], color=color, marker="o",
                   label=f"{name} (real)", alpha=0.7)
        ax.scatter(proj[mask_synth, 0], proj[mask_synth, 1], color=color, marker="x",
                   label=f"{name} (synthetic)", alpha=0.7)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Real vs. synthetic feature space")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    plt.close(fig)
    return fig


# ==============================================================================
# Figure 6 — practitioner flowchart
# ==============================================================================

def plot_practitioner_flowchart(threshold: float = None, output_path=None):
    """
    Publication-styled box-and-arrow rendering of
    kid_screening.PRACTITIONER_FLOWCHART_MERMAID's decision logic. The
    Mermaid source (kid_screening.generate_practitioner_flowchart) remains
    the primary, authoritative artifact for anyone re-deriving the logic;
    this is a standalone matplotlib rendering for direct inclusion as a
    paper figure without a Mermaid renderer in the loop.
    """
    _use_style()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    threshold_label = f"{threshold:.2f}" if threshold is not None else "τ"

    boxes = [
        (5, 9.3, "Fewer than ~30 real\nimages per rare class?"),
        (1.8, 6.7, "No: SD augmentation\nlikely safe—still\ncheck KID"),
        (7.8, 6.7, "Yes: generate synthetic\nimages, compute KID vs.\nheld-out test images"),
        (7.8, 4.2, f"KID > {threshold_label}?"),
        (9.3, 1.4, "Yes: do NOT use SD\naugmentation—consider\nfew-shot methods"),
        (5.5, 1.4, "No: augmentation may\nhelp—monitor rare-class\nF1 carefully"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#333333"))

    arrows = [
        ((5, 8.7), (1.8, 7.3)), ((5, 8.7), (7.8, 7.3)),
        ((7.8, 6.1), (7.8, 4.8)),
        ((7.8, 3.6), (9.3, 2.0)), ((7.8, 3.6), (5.5, 2.0)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#333333"))

    ax.set_title("Practitioner decision flowchart")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    plt.close(fig)
    return fig


# ==============================================================================
# Top-level driver
# ==============================================================================

def generate_all_figures(figures_out_dir, n_sweep_crossover_rows=None,
                          ratio_curves_by_class=None, ratio_n_anchor_label=None,
                          kid_delta_df=None, kid_threshold=None,
                          feature_variance_df=None,
                          tsne_data=None,
                          class_names=None) -> dict:
    """
    Generates whichever of the 6 figures have their inputs supplied,
    skipping (with a printed notice, not a raised error) any that don't —
    figures 4/5 in particular need real extracted features from a trained
    checkpoint, a separate expensive step this driver does not trigger
    itself. tsne_data, if given: dict with keys real_features, real_labels,
    synth_features, synth_labels. Returns {figure_name: output_path_or_None}.
    """
    out_dir = Path(figures_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if n_sweep_crossover_rows is not None:
        p = out_dir / "fig1_n_sweep_curves.png"
        results["n_sweep_curves"] = str(p) if plot_n_sweep_curves(n_sweep_crossover_rows, p) else None
    else:
        print("generate_all_figures: skipping Fig 1 (n-sweep curves) — no crossover_rows given")
        results["n_sweep_curves"] = None

    if ratio_curves_by_class is not None:
        p = out_dir / "fig2_ratio_ablation_curves.png"
        results["ratio_ablation_curves"] = str(p) if plot_ratio_ablation_curves(
            ratio_curves_by_class, p, n_anchor_label=ratio_n_anchor_label) else None
    else:
        print("generate_all_figures: skipping Fig 2 (ratio ablation) — no ratio_curves_by_class given")
        results["ratio_ablation_curves"] = None

    if kid_delta_df is not None and kid_threshold is not None:
        p = out_dir / "fig3_kid_screening.png"
        results["kid_screening"] = str(p) if plot_kid_screening(kid_delta_df, kid_threshold, p) else None
    else:
        print("generate_all_figures: skipping Fig 3 (KID screening) — no kid_delta_df/threshold given")
        results["kid_screening"] = None

    if feature_variance_df is not None:
        p = out_dir / "fig4_feature_variance_vs_n.png"
        results["feature_variance_vs_n"] = str(p) if plot_feature_variance_vs_n(feature_variance_df, p) else None
    else:
        print("generate_all_figures: skipping Fig 4 (feature variance vs n) — no feature_variance_df given")
        results["feature_variance_vs_n"] = None

    if tsne_data is not None:
        p = out_dir / "fig5_tsne_comparison.png"
        fig = plot_tsne_comparison(
            tsne_data["real_features"], tsne_data["real_labels"],
            tsne_data["synth_features"], tsne_data["synth_labels"],
            class_names=class_names, output_path=p,
        )
        results["tsne_comparison"] = str(p) if fig else None
    else:
        print("generate_all_figures: skipping Fig 5 (t-SNE) — no tsne_data given")
        results["tsne_comparison"] = None

    p = out_dir / "fig6_practitioner_flowchart.png"
    plot_practitioner_flowchart(kid_threshold, p)
    results["practitioner_flowchart"] = str(p)

    return results

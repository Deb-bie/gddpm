"""
track_a/analysis.py
=====================
Phase 7 — post-hoc analysis over the result cells written by
track_a/main.py: crossover point (n*) detection, bootstrap CIs on
target-class F1, the RQ6 KID->DeltaF1 screening correlation, and the
RQ1 (cross-modality n*) / RQ5 (SD+LoRA vs DCGAN) / RQ8 (backbone contrast)
comparison tables.

Does not train or generate anything — reads the JSON + .preds.npz files
train_one_condition() and compute_kid_sweep() already wrote, so this can
be re-run cheaply any time after (or during, on whatever's finished so
far) a track_a/main.py run.

Cell naming convention (must match track_a/main.py exactly):
  results_dir / f"{dataset}_{backbone}_class{cls}_n{n}_{condition}.json"
  results_dir / f"{dataset}_{backbone}_class{cls}_n{n}_{condition}.preds.npz"
"""

import json
from pathlib import Path

import numpy as np


# ==============================================================================
# Cell loading
# ==============================================================================

def _cell_path(results_dir, dataset, backbone, cls, n, condition) -> Path:
    return Path(results_dir) / f"{dataset}_{backbone}_class{cls}_n{n}_{condition}.json"


def load_cell(results_dir, dataset, backbone, cls, n, condition) -> dict | None:
    """Returns the result dict, or None if missing or explicitly skipped
    (infeasible n for this class — see track_a/main.py's skip handling)."""
    p = _cell_path(results_dir, dataset, backbone, cls, n, condition)
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return None if d.get("skipped") else d


def load_cell_preds(results_dir, dataset, backbone, cls, n, condition):
    """Returns (y_true, y_pred) int arrays, or None if not present."""
    p = _cell_path(results_dir, dataset, backbone, cls, n, condition).with_suffix(".preds.npz")
    if not p.exists():
        return None
    data = np.load(p)
    return data["y_true"], data["y_pred"]


# ==============================================================================
# Bootstrap CI on target-class F1
# ==============================================================================

def bootstrap_f1_ci(y_true: np.ndarray, y_pred: np.ndarray, target_class,
                     n_bootstrap: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """
    Percentile bootstrap CI for one class's F1, resampling (y_true, y_pred)
    pairs jointly with replacement. Resamples where the target class
    doesn't appear at all are dropped (F1 undefined) rather than treated
    as 0 — with as few as 1-4 target-class examples in a val split, this
    can meaningfully shrink the effective bootstrap count; that's reported
    (n_valid_bootstraps) rather than hidden.
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    point = float(f1_score(y_true, y_pred, labels=[target_class], average="macro", zero_division=0))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if (y_true[idx] == target_class).sum() == 0:
            continue
        boots.append(f1_score(y_true[idx], y_pred[idx], labels=[target_class], average="macro", zero_division=0))

    if not boots:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_valid_bootstraps": 0}

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_valid_bootstraps": len(boots)}


# ==============================================================================
# Chance-level test — the RQ2a diagnostic itself
# ==============================================================================

def chance_level(n_classes: int) -> float:
    """Expected F1 for a target class under random guessing over n_classes."""
    return 1.0 / n_classes


def above_chance_test(rare_f1: float, n_classes: int, ci_lower: float) -> bool:
    """
    True if we can conclude (at whatever confidence level ci_lower was
    built at — bootstrap_f1_ci defaults to 95%) that performance is above
    chance: the bootstrap CI's lower bound must clear chance_level, not
    just the point estimate. rare_f1 is accepted (not used in the
    comparison itself) so call sites can log point-estimate-vs-CI
    together without a second lookup.
    """
    return ci_lower > chance_level(n_classes)


def synthetic_only_diagnostic(results_dir, dataset, backbone, cls, n, num_classes,
                               aug_condition: str = "sd_lora_synth", n_bootstrap: int = 1000,
                               seed: int = 42) -> dict:
    """
    The single most diagnostic row for the RQ2a mechanism question: for one
    (class, n), lay out S1 (real-only), the augmented condition, and
    synth_only side by side against chance level. If synth_only's
    bootstrap CI lower bound clears chance, the synthetic images carry
    real class-discriminative signal even with zero real examples of that
    class in training — evidence against "the generator learned nothing
    class-specific." If synth_only sits at chance, the opposite.
    """
    s1         = load_cell(results_dir, dataset, backbone, cls, n, "real_only")
    s3         = load_cell(results_dir, dataset, backbone, cls, n, aug_condition)
    synth_only = load_cell(results_dir, dataset, backbone, cls, n, "synth_only")
    chance     = chance_level(num_classes)

    row = {
        "class": cls, "n": n,
        "s1_f1": s1["f1_target"] if s1 else None,
        "s3_f1": s3["f1_target"] if s3 else None,
        "synth_only_f1": synth_only["f1_target"] if synth_only else None,
        "chance_level": chance,
        "synth_only_ci_lower": None,
        "above_chance": None,
        "flag": "",
    }

    if synth_only is not None:
        preds = load_cell_preds(results_dir, dataset, backbone, cls, n, "synth_only")
        if preds is not None:
            yt, yp = preds
            ci = bootstrap_f1_ci(yt, yp, cls, n_bootstrap=n_bootstrap, seed=seed)
            row["synth_only_ci_lower"] = ci["ci_low"]
            row["above_chance"] = above_chance_test(row["synth_only_f1"], num_classes, ci["ci_low"])
            row["flag"] = "*" if row["above_chance"] else ""

    return row


def synthetic_only_report_table(results_dir, dataset, backbone, sweep_classes, n_grid,
                                 num_classes, aug_condition: str = "sd_lora_synth") -> list:
    """One row per (class, n): S1 F1 | S3 F1 | synth_only F1 | chance | flag."""
    return [
        synthetic_only_diagnostic(results_dir, dataset, backbone, cls, n, num_classes, aug_condition)
        for cls in sweep_classes for n in n_grid
    ]


def print_synthetic_only_table(rows: list, class_names: dict = None) -> None:
    header = f"{'Class':<28} {'n':>5} {'S1 F1':>8} {'S3 F1':>8} {'SynthOnly F1':>13} {'Chance':>8}  Flag"
    print(header)
    print("-" * len(header))
    for r in rows:
        name = (class_names or {}).get(r["class"], str(r["class"]))
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "  n/a"
        print(f"{name:<28} {r['n']:>5} {fmt(r['s1_f1']):>8} {fmt(r['s3_f1']):>8} "
              f"{fmt(r['synth_only_f1']):>13} {fmt(r['chance_level']):>8}  {r['flag']}")


# ==============================================================================
# Delta-F1 curves and crossover point (n*)
# ==============================================================================

def delta_f1_curve(results_dir, dataset, backbone, cls, n_grid,
                    aug_condition: str = "sd_lora_synth", base_condition: str = "real_only") -> list:
    """
    One row per feasible n: {n, f1_base, f1_aug, delta}. n's where either
    condition is missing/infeasible for this class are simply absent from
    the returned list — NOT filled with a placeholder value, since a
    missing cell (e.g. GastroVision's Ulcer at n=128) is a real
    data-availability fact the paper needs to report as missing, not
    interpolate over.
    """
    rows = []
    for n in n_grid:
        base = load_cell(results_dir, dataset, backbone, cls, n, base_condition)
        aug  = load_cell(results_dir, dataset, backbone, cls, n, aug_condition)
        if base is None or aug is None:
            continue
        rows.append({
            "n": n, "f1_base": base["f1_target"], "f1_aug": aug["f1_target"],
            "delta": aug["f1_target"] - base["f1_target"],
        })
    return sorted(rows, key=lambda r: r["n"])


def find_crossover(delta_rows: list):
    """
    Smallest n at which delta > 0 (augmentation starts helping rather than
    hurting). Returns None if augmentation never helps anywhere in the
    observed grid (a real, reportable finding — not an error) or if there
    are no feasible points at all.
    """
    for r in delta_rows:
        if r["delta"] > 0:
            return r["n"]
    return None


def crossover_table(results_dir, dataset, backbone, sweep_classes, n_grid,
                     class_names: dict = None, aug_condition: str = "sd_lora_synth") -> list:
    rows = []
    for cls in sweep_classes:
        curve = delta_f1_curve(results_dir, dataset, backbone, cls, n_grid, aug_condition=aug_condition)
        rows.append({
            "class": cls,
            "class_name": (class_names or {}).get(cls, str(cls)),
            "n_star": find_crossover(curve),
            "n_feasible_points": len(curve),
            "curve": curve,
        })
    return rows


# ==============================================================================
# RQ6 — KID -> DeltaF1 screening correlation
# ==============================================================================

def kid_delta_f1_correlation(results_dir, dataset, backbone, sweep_classes, n_grid,
                              gen_model: str = "sd_lora") -> dict:
    """
    Pools every feasible (class, n) cell's (KID, DeltaF1) pair for one
    dataset/backbone/generative-model arm and computes Spearman rho —
    RQ6's actual test: does KID, measured at the SAME n as the classifier
    condition (per track_a/quality_metrics.py's design), predict the
    resulting change in target-class F1 from adding that synthetic data?
    """
    from scipy.stats import spearmanr

    kid_path = Path(results_dir) / f"kid_per_class_per_n_{gen_model}.json"
    if not kid_path.exists():
        return {"error": f"{kid_path} not found"}
    with open(kid_path) as f:
        kid_data = json.load(f)

    condition = "sd_lora_synth" if gen_model == "sd_lora" else "dcgan_synth"
    pairs = []
    for cls in sweep_classes:
        cls_kid = kid_data.get(str(cls), {})
        for n in n_grid:
            cell = cls_kid.get(str(n))
            if cell is None or not np.isfinite(cell.get("kid", float("nan"))):
                continue
            base = load_cell(results_dir, dataset, backbone, cls, n, "real_only")
            aug  = load_cell(results_dir, dataset, backbone, cls, n, condition)
            if base is None or aug is None:
                continue
            pairs.append({
                "class": cls, "n": n, "kid": cell["kid"],
                "delta_f1": aug["f1_target"] - base["f1_target"],
            })

    if len(pairs) < 3:
        return {"error": "insufficient paired (KID, DeltaF1) points", "n_pairs": len(pairs)}

    kids   = [p["kid"] for p in pairs]
    deltas = [p["delta_f1"] for p in pairs]
    rho, pval = spearmanr(kids, deltas)
    return {"spearman_rho": float(rho), "p_value": float(pval), "n_pairs": len(pairs), "pairs": pairs}


# ==============================================================================
# RQ5 — SD+LoRA vs. DCGAN generative-model-generality comparison
# ==============================================================================

def rq5_generative_comparison(results_dir, dataset, backbone, sweep_classes, n_grid) -> list:
    """
    Per (class, n): DeltaF1 from SD+LoRA and from DCGAN, both relative to
    the SAME real_only baseline. If DCGAN's delta tracks SD+LoRA's delta
    closely, the low-n failure generalizes across generative-model
    families (not diffusion-specific); if they diverge, the failure is
    tied to something diffusion-specific (pretrained backbone, or the
    text-conditioning mechanism).
    """
    rows = []
    for cls in sweep_classes:
        for n in n_grid:
            base = load_cell(results_dir, dataset, backbone, cls, n, "real_only")
            if base is None:
                continue
            sd  = load_cell(results_dir, dataset, backbone, cls, n, "sd_lora_synth")
            gan = load_cell(results_dir, dataset, backbone, cls, n, "dcgan_synth")
            rows.append({
                "class": cls, "n": n,
                "delta_sd":    (sd["f1_target"] - base["f1_target"]) if sd  else None,
                "delta_dcgan": (gan["f1_target"] - base["f1_target"]) if gan else None,
            })
    return rows


# ==============================================================================
# HAM10000 external qualitative check — Sagers et al., NOT an "Akrout et al.
# replication" (that framing was dropped, see below)
# ==============================================================================

# Recorded once, from track_a_prior_work_review.docx Section 3/9 — Sagers et
# al. (arXiv:2308.12453) is the only paper in our reading list that actually
# ran a real n-sweep on a dermatology dataset (Fitzpatrick17k, Stanford DDI)
# with reported numbers. The pasted A1 task asked us to replicate "Akrout et
# al." on HAM10000 and hardcode a comparison against a "Traboulsi et al."
# result — but Akrout et al. never used HAM10000 or LoRA (proprietary
# dataset, textual inversion), and "Traboulsi et al." matches no paper in
# our reading list at all, so that comparison would have been printing a
# fabricated reference number next to a real one. This isn't an exact
# numeric replication target either (different dataset, different disease
# taxonomy) — it's a QUALITATIVE shape check: does our HAM10000 dose-response
# curve look like the pattern Sagers et al. actually reported (monotonic
# improvement with saturation as n grows), on the one prior paper that ran
# a comparable n-sweep on a comparable (dermatology) modality?
SAGERS_ET_AL_REFERENCE = {
    "citation": "Sagers et al. 2023, arXiv:2308.12453",
    "datasets": ["Fitzpatrick17k", "Stanford DDI"],
    "n_grid_used": [1, 16, 32, 64, 128, 228],
    "qualitative_finding": (
        "Dose-response-with-saturation: augmentation benefit increases "
        "monotonically with n up to a point, then saturates — augmentation "
        "helps more as more real reference data is available, not less."
    ),
}


def compare_ham10000_to_sagers(ham10000_crossover_rows: list) -> dict:
    """
    Qualitative (not numeric) comparison: does HAM10000's own crossover
    table (from crossover_table()) show the same MONOTONIC,
    saturating-benefit shape Sagers et al. reported, or does it diverge
    (e.g. benefit peaks then reverses, or never turns positive at all)?
    Returns a per-class classification, not a claim of matching numbers —
    HAM10000 and Sagers' datasets differ in disease taxonomy and are not
    directly numerically comparable.
    """
    out = {"reference": SAGERS_ET_AL_REFERENCE, "per_class": []}
    for row in ham10000_crossover_rows:
        curve = row["curve"]
        deltas = [c["delta"] for c in curve]
        if len(deltas) < 2:
            shape = "insufficient_points"
        elif all(d2 >= d1 - 1e-9 for d1, d2 in zip(deltas, deltas[1:])):
            shape = "monotonic_improving (matches Sagers' reported shape)"
        elif deltas[-1] > deltas[0]:
            shape = "non-monotonic_but_net_improving (partially consistent)"
        else:
            shape = "non-monotonic_or_declining (diverges from Sagers' shape)"
        out["per_class"].append({
            "class": row["class"], "class_name": row["class_name"],
            "n_star": row["n_star"], "shape": shape,
        })
    return out


# ==============================================================================
# RQ8 — backbone contrast (EfficientNetV2-S vs. DINOv2)
# ==============================================================================

def rq8_backbone_contrast(results_dir, dataset, sweep_classes, n_grid,
                           aug_condition: str = "sd_lora_synth") -> list:
    """Per class: crossover n* and full curve for both backbones side by side."""
    rows = []
    for cls in sweep_classes:
        eff  = delta_f1_curve(results_dir, dataset, "efficientnetv2_rw_s", cls, n_grid, aug_condition=aug_condition)
        dino = delta_f1_curve(results_dir, dataset, "dinov2", cls, n_grid, aug_condition=aug_condition)
        rows.append({
            "class": cls,
            "effnet_n_star":  find_crossover(eff),
            "dinov2_n_star":  find_crossover(dino),
            "effnet_curve":   eff,
            "dinov2_curve":   dino,
        })
    return rows


# ==============================================================================
# RQ1 — cross-modality consistency of n*
# ==============================================================================

def rq1_cross_modality_table(output_dir_root, dataset_modules: dict, backbone, n_grid,
                              aug_condition: str = "sd_lora_synth") -> dict:
    """
    dataset_modules: {dataset_name: dataset_module} (from
    track_a.datasets.get_dataset), used for SWEEP_CLASSES/CLASS_NAMES.
    output_dir_root: Track A's OUTPUT_DIR — results live at
    output_dir_root/{dataset}/results/.
    """
    out = {}
    for dataset, mod in dataset_modules.items():
        results_dir = Path(output_dir_root) / dataset / "results"
        out[dataset] = crossover_table(
            results_dir, dataset, backbone, mod.SWEEP_CLASSES, n_grid,
            class_names=mod.CLASS_NAMES, aug_condition=aug_condition,
        )
    return out


# ==============================================================================
# Top-level driver
# ==============================================================================

def run_full_analysis(output_dir_root, dataset_modules: dict, backbones: list, n_grid: list,
                       results_out_path=None) -> dict:
    """
    Runs every analysis above for every dataset x backbone and (for the
    default backbone) the cross-modality table, bundling everything into
    one JSON. Safe to call at any point — cells that don't exist yet just
    produce empty/None entries rather than raising.
    """
    report = {"rq1_cross_modality": {}, "rq5_generative_comparison": {},
              "rq6_kid_delta_f1": {}, "rq8_backbone_contrast": {}}

    default_backbone = backbones[0]
    report["rq1_cross_modality"] = rq1_cross_modality_table(
        output_dir_root, dataset_modules, default_backbone, n_grid
    )

    for dataset, mod in dataset_modules.items():
        results_dir = Path(output_dir_root) / dataset / "results"

        report["rq8_backbone_contrast"][dataset] = rq8_backbone_contrast(
            results_dir, dataset, mod.SWEEP_CLASSES, n_grid
        )

        report["rq5_generative_comparison"][dataset] = {}
        report["rq6_kid_delta_f1"][dataset] = {}
        for backbone in backbones:
            report["rq5_generative_comparison"][dataset][backbone] = rq5_generative_comparison(
                results_dir, dataset, backbone, mod.SWEEP_CLASSES, n_grid
            )
            report["rq6_kid_delta_f1"][dataset][backbone] = {
                "sd_lora": kid_delta_f1_correlation(results_dir, dataset, backbone, mod.SWEEP_CLASSES, n_grid, "sd_lora"),
                "dcgan":   kid_delta_f1_correlation(results_dir, dataset, backbone, mod.SWEEP_CLASSES, n_grid, "dcgan"),
            }

    if results_out_path is not None:
        with open(results_out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Full analysis report saved -> {results_out_path}")

    return report


if __name__ == "__main__":
    from config import args, OUTPUT_DIR
    from datasets import get_dataset

    mods = {name: get_dataset(name) for name in args.datasets}
    run_full_analysis(OUTPUT_DIR, mods, args.backbones, args.n_grid,
                       results_out_path=Path(OUTPUT_DIR) / "track_a_analysis_report.json")

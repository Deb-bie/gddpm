"""
track_a/multiseed.py
=======================
Phase 12 — seed-variance reporting for the 3 headline findings, per the
pasted spec's multiseed-analysis request. Scoped deliberately narrow: NOT
a full re-run of the entire n-grid x conditions x backbones matrix at
multiple seeds (cost-prohibitive — see config.py's MULTISEED_DEFAULT_SEEDS
docstring), just the three specific claims the paper leans on hardest:

  1. Crossover point n* (does the augmentation-starts-helping point move
     around under a different random subsample/training seed?)
  2. The synthetic:real ratio curve at n=16 (RQ_A5's most diagnostic anchor
     — is "more synthetic hurts" a stable shape or a single-seed artifact?)
  3. The KID -> DeltaF1 Spearman rho (RQ6's screening-criterion claim)

Design choice — no changes to analysis.py / ratio_ablation.py's cell
naming: every reader/writer function here builds a CONDITION STRING with a
"_seed{seed}" suffix (e.g. "sd_lora_synth_seed123") and passes that string
straight into the existing load_cell()/build_condition_df()/
train_one_condition() machinery, which just treats it as an opaque tag.
This means:
  - args.seed's own cells are read back AS-IS (no suffix) from whatever
    main.py's Step 8 / ratio_ablation.py already produced — never retrained.
  - Every OTHER seed's cells get their own suffixed files, so re-running
    with --run_multiseed can never clobber the base-seed results.
No core module needed to change to support this.
"""

import json
import itertools
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from analysis import load_cell, delta_f1_curve, find_crossover


# ==============================================================================
# SeedAggregator — generic mean/std/bootstrap-CI across a small set of
# per-seed point estimates
# ==============================================================================

class SeedAggregator:

    @staticmethod
    def aggregate(values: list, n_bootstrap: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
        """
        values: one point estimate per seed (e.g. one rho, one F1). None /
        non-finite entries are dropped. With as few as 3 seeds, a bootstrap
        CI over "which seeds got sampled" is necessarily crude — n_seeds is
        always reported alongside so this isn't mistaken for a
        well-powered CI. This is a SECOND, independent layer of uncertainty
        from analysis.bootstrap_f1_ci's within-cell resampling: that one
        asks "how uncertain is this F1 given this training run's val set,"
        this one asks "how much does the point estimate itself move if we'd
        used a different training seed."
        """
        arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
        n = len(arr)
        if n == 0:
            return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"),
                    "ci_high": float("nan"), "n_seeds": 0}
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0

        if n == 1:
            return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean,
                    "n_seeds": 1, "note": "single seed — no variance estimate possible"}

        rng = np.random.default_rng(seed)
        boots = [rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)]
        lo = float(np.percentile(boots, 100 * alpha / 2))
        hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
        out = {"mean": mean, "std": std, "ci_low": lo, "ci_high": hi, "n_seeds": n}
        if n < 5:
            out["note"] = f"only {n} seeds — bootstrap CI is illustrative, not a reliable interval"
        return out


def seed_variance_report(name: str, values_by_seed: dict, **agg_kwargs) -> dict:
    """values_by_seed: {seed: value_or_None}. Bundles the per-seed listing
    with SeedAggregator's summary under one named report entry."""
    agg = SeedAggregator.aggregate(list(values_by_seed.values()), **agg_kwargs)
    return {"name": name, "values_by_seed": values_by_seed, **agg}


# ==============================================================================
# 1. Crossover point (n*) across seeds
# ==============================================================================

def read_multiseed_crossover_curves(results_dir, dataset, backbone, cls, n_grid, seeds,
                                     base_seed: int = 42, aug_condition: str = "sd_lora_synth",
                                     base_condition: str = "real_only") -> dict:
    """{seed: delta_f1_curve rows} — reads back args.seed's curve unmodified
    and every other seed's curve from its _seed{seed}-suffixed cells."""
    curves = {}
    for seed in seeds:
        aug_cond  = aug_condition  if seed == base_seed else f"{aug_condition}_seed{seed}"
        base_cond = base_condition if seed == base_seed else f"{base_condition}_seed{seed}"
        curves[seed] = delta_f1_curve(results_dir, dataset, backbone, cls, n_grid,
                                       aug_condition=aug_cond, base_condition=base_cond)
    return curves


class CrossoverCIWithSeeds:
    """
    n* is an ordinal grid value (one of N_GRID), not a continuous quantity
    — reporting a bootstrapped mean+CI on it would suggest "n*=47.3" is
    meaningful when only {1,16,32,64,128,228} are observable outcomes.
    Instead this reports the empirical distribution directly: how often a
    crossover exists at all across seeds, and which n-value is the modal
    (most frequent) crossover point among the seeds where one exists.
    """

    @staticmethod
    def from_curves(curves_by_seed: dict) -> dict:
        n_stars = {seed: find_crossover(curve) for seed, curve in curves_by_seed.items()}
        finite = [v for v in n_stars.values() if v is not None]
        n_seeds = len(n_stars)
        frac_with_crossover = (len(finite) / n_seeds) if n_seeds else float("nan")

        if finite:
            counts = Counter(finite)
            modal_n_star, modal_count = counts.most_common(1)[0]
            modal_agreement_fraction = modal_count / len(finite)
        else:
            modal_n_star, modal_agreement_fraction = None, float("nan")

        return {
            "n_star_by_seed": n_stars,
            "n_seeds": n_seeds,
            "fraction_seeds_with_crossover": frac_with_crossover,
            "modal_n_star": modal_n_star,
            "modal_agreement_fraction": modal_agreement_fraction,
        }


def run_multiseed_crossover_training(dataset_name, ds_module, dirs, real_root, train_df, val_df,
                                      num_classes, args, device, seeds,
                                      sd_pool_main=None, backbones=None, n_grid=None,
                                      conditions=("real_only", "sd_lora_synth")) -> list:
    """
    Retrains real_only + sd_lora_synth cells for every seed OTHER than
    args.seed across the full n_grid (args.seed's own cells already exist
    from main.py's Step 8 and are never re-touched here). Each extra seed
    gets its own subsample manifest too (not just a different training
    seed) — a real independent re-run would redraw the random n-subset as
    well as re-initialize the classifier, so this captures both sources of
    variance rather than training-noise alone.
    """
    from subsample import build_subsample_manifest
    from classifiers import build_condition_df, train_one_condition

    backbones = backbones or args.backbones
    n_grid = n_grid or args.n_grid
    base_seed = args.seed
    touched = []

    for seed in seeds:
        if seed == base_seed:
            continue
        seed_manifest = build_subsample_manifest(train_df, "label", n_grid, seed=seed)

        for backbone, cls, n, condition in itertools.product(
            backbones, ds_module.SWEEP_CLASSES, n_grid, conditions
        ):
            if condition == "sd_lora_synth" and sd_pool_main is None:
                continue

            cell_tag = f"{dataset_name}_{backbone}_class{cls}_n{n}_{condition}_seed{seed}"
            ckpt_path    = dirs["checkpoints"] / f"{cell_tag}.pt"
            results_path = dirs["results"] / f"{cell_tag}.json"

            synth_pool = sd_pool_main if condition == "sd_lora_synth" else None
            cond_df = build_condition_df(
                train_df, "label", cls, n, condition, seed_manifest,
                synth_pool_df=synth_pool, synth_ratio=args.synth_ratio, seed=seed,
            )
            if cond_df is None:
                if not results_path.exists():
                    with open(results_path, "w") as f:
                        json.dump({"skipped": True, "reason": "infeasible_n_for_class"}, f, indent=2)
                continue

            roots = {"real": real_root, "synth_sd": dirs["synth_sd"], "synth_dcgan": dirs["synth_dcgan"]}
            train_one_condition(
                backbone, cond_df, val_df, roots, num_classes, cls, args, device,
                ckpt_path, results_path,
            )
            touched.append(cell_tag)

    return touched


# ==============================================================================
# 2. Ratio ablation curve (at one n anchor, default 16) across seeds
# ==============================================================================

def run_multiseed_ratio_training(dataset_name, ds_module, dirs, real_root, train_df, val_df,
                                  num_classes, args, device, seeds, sd_pool,
                                  n: int = None, ratios=None) -> list:
    """
    Parallels ratio_ablation.run_ratio_ablation, restricted to ONE n anchor
    (default the smaller of ABLATION_N_ANCHORS — n=16 is where the ratio
    curve's shape is most diagnostic per the pasted spec) and iterated
    across seeds. For seed == args.seed, ratio=1 is read back from the
    existing sd_lora_synth cell (already trained in Step 8) same as
    ratio_ablation.py does; every other ratio at every OTHER seed is a
    fresh, seed-suffixed training run — including ratio=1, since a
    different seed draws a different real/synthetic subsample even there.
    """
    from subsample import build_subsample_manifest
    from classifiers import build_condition_df, train_one_condition
    from ratio_ablation import _ratio_tag, RATIO_SET
    from config import ABLATION_N_ANCHORS

    n = n if n is not None else ABLATION_N_ANCHORS[0]
    ratios = ratios or RATIO_SET
    base_seed = args.seed
    touched = []

    for seed in seeds:
        seed_manifest = (
            build_subsample_manifest(train_df, "label", args.n_grid, seed=seed)
        )
        for backbone in args.backbones:
            for cls in ds_module.SWEEP_CLASSES:
                if n not in seed_manifest.get(cls, {}):
                    continue
                for ratio in ratios:
                    if ratio == 1 and seed == base_seed:
                        # Already trained by main.py Step 8 — read back, don't retrain.
                        touched.append((seed, f"{dataset_name}_{backbone}_class{cls}_n{n}_sd_lora_synth"))
                        continue

                    tag = _ratio_tag(dataset_name, backbone, cls, n, ratio)
                    if seed != base_seed:
                        tag = f"{tag}_seed{seed}"
                    ckpt_path    = dirs["checkpoints"] / f"{tag}.pt"
                    results_path = dirs["results"] / f"{tag}.json"

                    cond_df = build_condition_df(
                        train_df, "label", cls, n, "sd_lora_synth", seed_manifest,
                        synth_pool_df=sd_pool, synth_ratio=ratio, seed=seed,
                    )
                    if cond_df is None:
                        continue

                    roots = {"real": real_root, "synth_sd": dirs["synth_sd"], "synth_dcgan": dirs["synth_dcgan"]}
                    train_one_condition(
                        backbone, cond_df, val_df, roots, num_classes, cls, args, device,
                        ckpt_path, results_path,
                    )
                    touched.append((seed, tag))

    return touched


def read_multiseed_ratio_fitters(results_dir, dataset, backbone, cls, n, seeds,
                                  base_seed: int = 42, ratios=None) -> dict:
    """{seed: RatioCurveFitter} for one (dataset, backbone, class, n)."""
    from ratio_ablation import RatioCurveFitter, RATIO_SET

    ratios = ratios or RATIO_SET
    fitters = {}
    for seed in seeds:
        base_cond = "real_only" if seed == base_seed else f"real_only_seed{seed}"
        base = load_cell(results_dir, dataset, backbone, cls, n, base_cond)
        rows = []
        for ratio in ratios:
            if ratio == 1:
                cond = "sd_lora_synth" if seed == base_seed else f"sd_lora_synth_seed{seed}"
            else:
                cond = f"ratio{ratio:g}" if seed == base_seed else f"ratio{ratio:g}_seed{seed}"
            cell = load_cell(results_dir, dataset, backbone, cls, n, cond)
            if cell is None:
                continue
            rows.append({
                "ratio": ratio, "synthetic_count": ratio * n,
                "rare_f1": cell["f1_target"],
                "s1_baseline_f1": base["f1_target"] if base else None,
            })
        fitters[seed] = RatioCurveFitter(pd.DataFrame(rows), n)
    return fitters


def aggregate_ratio_curves_across_seeds(fitters_by_seed: dict) -> pd.DataFrame:
    """Mean +/- std rare_f1 per ratio, pooled across seeds' fitters — the
    table Figure 3 (ratio ablation, mean-with-error-bars variant) plots."""
    frames = []
    for seed, fitter in fitters_by_seed.items():
        df = fitter.df.copy()
        df["seed"] = seed
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ratio", "mean_rare_f1", "std_rare_f1", "n_seeds"])
    all_df = pd.concat(frames, ignore_index=True)
    grouped = all_df.groupby("ratio")["rare_f1"].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = ["ratio", "mean_rare_f1", "std_rare_f1", "n_seeds"]
    return grouped.sort_values("ratio").reset_index(drop=True)


# ==============================================================================
# 3. KID -> DeltaF1 Spearman rho across seeds
# ==============================================================================

def read_multiseed_kid_delta_f1_rhos(results_dir, dataset, backbone, sweep_classes, n_grid, seeds,
                                      base_seed: int = 42, gen_model: str = "sd_lora") -> dict:
    """
    KID doesn't need recomputing per seed — it's a property of the fixed
    synthetic pool vs. the fixed held-out test set (quality_metrics.py's
    design), independent of which seed trained the classifier. What DOES
    vary per seed is DeltaF1, so this re-pairs the one set of KID values
    against each seed's own DeltaF1 and refits Spearman rho per seed.
    Returns {seed: rho_or_None}.
    """
    from scipy.stats import spearmanr

    kid_path = Path(results_dir) / f"kid_per_class_per_n_{gen_model}.json"
    if not kid_path.exists():
        return {"error": f"{kid_path} not found"}
    with open(kid_path) as f:
        kid_data = json.load(f)

    condition = "sd_lora_synth" if gen_model == "sd_lora" else "dcgan_synth"
    rho_by_seed = {}
    for seed in seeds:
        aug_cond  = condition   if seed == base_seed else f"{condition}_seed{seed}"
        base_cond = "real_only" if seed == base_seed else f"real_only_seed{seed}"

        pairs = []
        for cls in sweep_classes:
            cls_kid = kid_data.get(str(cls), {})
            for n in n_grid:
                cell = cls_kid.get(str(n))
                if cell is None or not np.isfinite(cell.get("kid", float("nan"))):
                    continue
                base = load_cell(results_dir, dataset, backbone, cls, n, base_cond)
                aug  = load_cell(results_dir, dataset, backbone, cls, n, aug_cond)
                if base is None or aug is None:
                    continue
                pairs.append((cell["kid"], aug["f1_target"] - base["f1_target"]))

        if len(pairs) < 3:
            rho_by_seed[seed] = None
            continue
        kids, deltas = zip(*pairs)
        rho, _p = spearmanr(kids, deltas)
        rho_by_seed[seed] = float(rho)

    return rho_by_seed


# ==============================================================================
# Top-level bundled report
# ==============================================================================

def multiseed_report(output_dir_root, dataset_name, ds_module, backbone, n_grid, seeds,
                      base_seed: int = 42, ratio_n_anchor: int = 16,
                      results_out_path=None) -> dict:
    """
    Bundles all 3 headline-finding seed-variance analyses for one
    (dataset, backbone). Assumes run_multiseed_crossover_training and
    run_multiseed_ratio_training have already been run (this function only
    reads results back — it never trains anything itself, same
    read-vs-write split as analysis.py vs. main.py).
    """
    results_dir = Path(output_dir_root) / dataset_name / "results"
    report = {"dataset": dataset_name, "backbone": backbone, "seeds": seeds, "base_seed": base_seed,
              "crossover": {}, "ratio_at_n": {}, "kid_delta_f1_rho": {}}

    for cls in ds_module.SWEEP_CLASSES:
        curves = read_multiseed_crossover_curves(results_dir, dataset_name, backbone, cls, n_grid, seeds, base_seed)
        report["crossover"][cls] = CrossoverCIWithSeeds.from_curves(curves)

        fitters = read_multiseed_ratio_fitters(results_dir, dataset_name, backbone, cls, ratio_n_anchor, seeds, base_seed)
        agg_curve = aggregate_ratio_curves_across_seeds(fitters)
        report["ratio_at_n"][cls] = agg_curve.to_dict(orient="records")

    rho_by_seed = read_multiseed_kid_delta_f1_rhos(results_dir, dataset_name, backbone, ds_module.SWEEP_CLASSES, n_grid, seeds, base_seed)
    if "error" not in rho_by_seed:
        report["kid_delta_f1_rho"] = seed_variance_report("kid_delta_f1_rho", rho_by_seed)
    else:
        report["kid_delta_f1_rho"] = rho_by_seed

    if results_out_path is not None:
        with open(results_out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Multiseed report saved -> {results_out_path}")

    return report

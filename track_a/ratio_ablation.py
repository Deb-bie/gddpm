"""
track_a/ratio_ablation.py
===========================
Phase 9 (A5) — synthetic:real ratio ablation, decoupling two candidate
failure mechanisms:

  DILUTION:  too many synthetic images relative to real ones swamps the
             real signal, regardless of synthetic quality.
  MISMATCH:  synthetic images are out-of-distribution regardless of ratio
             — even a 1:1 mix hurts.

Run at ABLATION_N_ANCHORS = {16, 128} (track_a/config.py) — NOT the pasted
spec's {15, 100}. Those two numbers were chosen so the rank ablation, the
ratio ablation, and the main n-grid all share one consistent set of
reference n-values throughout the paper (see config.py's module docstring
and track_a_prior_work_review.docx Section 9 flag 4) — reusing them here
rather than introducing a third pair of anchor points.

Ratio set default is {1, 3, 5, 10}, deliberately more conservative than
the pasted spec's {1,5,10,33,67} / {1,5,10,30,100}. Those ratios assumed
per-(class, n) LoRA fine-tuning with a fresh, cheap-to-regenerate
synthetic bank at each n. We confirmed the opposite design (domain-adapt-
once + prompt-conditioning, matching the actual working gastrovision/
pipeline) — which means the SD/LoRA synthetic pool is ONE fixed, shared
per-class pool (Phase 2), and covering a ratio like 100:1 at n=128 would
require 12,800 pre-generated images per class just for this one ablation.
{1,3,5,10} keeps the largest requirement (10:1 at n=128 -> 1,280
images/class) within reach of a bumped-up samples_per_class rather than
an order of magnitude beyond it. Widen RATIO_SET below if you want the
more extreme points and are willing to pay for the extra generation.

The ratio=1.0 condition is NEVER retrained here — it's identical to the
main n-grid's "sd_lora_synth" condition at that n (already computed in
main.py's Step 8), so this module reuses that existing result file
instead of duplicating a training run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import ABLATION_N_ANCHORS
from generative.lora import generate_synthetic_for_classes
from classifiers import build_condition_df, train_one_condition
from analysis import load_cell

RATIO_SET = (1, 3, 5, 10)


def _ratio_tag(dataset_name, backbone, cls, n, ratio) -> str:
    return f"{dataset_name}_{backbone}_class{cls}_n{n}_ratio{ratio:g}"


def ensure_synthetic_bank(dataset_name: str, ds_module, dirs, args, device,
                           ratios=RATIO_SET, n_anchors=None, rank: int = None):
    """
    Extends the existing per-class SD/LoRA pool (Phase 2) up to
    max(n * ratio) images per class, if it isn't already that large.
    generate_synthetic_for_classes() is already resumable (only generates
    the DIFFERENCE between the current pool size and the requested target),
    so calling it again with a bigger samples_per_class just tops up the
    existing pool rather than regenerating it from scratch.
    """
    n_anchors = n_anchors or ABLATION_N_ANCHORS
    needed_max = max(n * r for n in n_anchors for r in ratios)
    print(f"Ratio ablation synthetic bank — {dataset_name}: need up to "
          f"{needed_max} images/class (largest of n_anchors x ratios)")
    rank = rank if rank is not None else args.lora_rank
    generate_synthetic_for_classes(
        dataset_name, dirs["checkpoints"], dirs["synth_sd"],
        ds_module.SWEEP_CLASSES, ds_module.CLASS_PROMPTS, ds_module.DOMAIN_PREFIX,
        ds_module.NEGATIVE_PROMPT, ds_module.CLASS_NAMES, needed_max, args, device, rank=rank,
    )
    csv_path = dirs["synth_sd"] / f"synthetic_r{rank}.csv"
    return pd.read_csv(csv_path) if csv_path.exists() else None


def run_ratio_ablation(dataset_name: str, ds_module, dirs, real_root, train_df, val_df,
                        manifest: dict, sd_pool: pd.DataFrame, num_classes: int, args, device,
                        ratios=RATIO_SET, n_anchors=None) -> list:
    """
    Runs every (backbone, class, n_anchor, ratio) cell not already covered
    by the main n-grid matrix (ratio=1.0 is skipped and read back from the
    existing sd_lora_synth cell instead). Returns the list of cell tags
    touched, for bookkeeping.
    """
    n_anchors = n_anchors or ABLATION_N_ANCHORS
    touched = []

    for backbone in args.backbones:
        for cls in ds_module.SWEEP_CLASSES:
            for n in n_anchors:
                if n not in manifest.get(cls, {}):
                    continue  # infeasible cell — same rule as the main matrix
                for ratio in ratios:
                    if ratio == 1:
                        # Identical to the main grid's sd_lora_synth cell at
                        # this n — do not retrain, just note the tag so
                        # RatioCurveFitter can look it up under its real name.
                        touched.append(f"{dataset_name}_{backbone}_class{cls}_n{n}_sd_lora_synth")
                        continue

                    tag = _ratio_tag(dataset_name, backbone, cls, n, ratio)
                    ckpt_path    = dirs["checkpoints"] / f"{tag}.pt"
                    results_path = dirs["results"] / f"{tag}.json"

                    cond_df = build_condition_df(
                        train_df, "label", cls, n, "sd_lora_synth", manifest,
                        synth_pool_df=sd_pool, synth_ratio=ratio, seed=args.seed,
                    )
                    if cond_df is None:
                        continue

                    roots = {"real": real_root, "synth_sd": dirs["synth_sd"], "synth_dcgan": dirs["synth_dcgan"]}
                    train_one_condition(
                        backbone, cond_df, val_df, roots, num_classes, cls, args, device,
                        ckpt_path, results_path,
                    )
                    touched.append(tag)

    return touched


# ==============================================================================
# RatioCurveFitter — reads back the cells run above
# ==============================================================================

def _load_ratio_cell(results_dir, dataset, backbone, cls, n, ratio):
    if ratio == 1:
        return load_cell(results_dir, dataset, backbone, cls, n, "sd_lora_synth")
    from pathlib import Path as _P
    import json as _json
    p = _P(results_dir) / f"{_ratio_tag(dataset, backbone, cls, n, ratio)}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = _json.load(f)
    return None if d.get("skipped") else d


class RatioCurveFitter:
    """
    Operates on one (dataset, backbone, class, n) ratio curve at a time —
    build with `from_results()`, which reads back every ratio in `ratios`
    for that cell and assembles the DataFrame the pasted spec describes
    (synthetic_count, rare_f1, plus the derived ratio column).
    """

    def __init__(self, df: pd.DataFrame, n_real: int):
        self.df = df.sort_values("ratio").reset_index(drop=True)
        self.n_real = n_real

    @classmethod
    def from_results(cls, results_dir, dataset, backbone, target_class, n_real,
                      ratios=RATIO_SET, base_condition: str = "real_only"):
        base = load_cell(results_dir, dataset, backbone, target_class, n_real, base_condition)
        rows = []
        for ratio in ratios:
            cell = _load_ratio_cell(results_dir, dataset, backbone, target_class, n_real, ratio)
            if cell is None:
                continue
            rows.append({
                "ratio": ratio,
                "synthetic_count": ratio * n_real,
                "rare_f1": cell["f1_target"],
                "s1_baseline_f1": base["f1_target"] if base else None,
            })
        return cls(pd.DataFrame(rows), n_real)

    def is_monotone_decreasing(self, epsilon: float = 0.01) -> bool:
        """
        True if F1 never increases by more than epsilon between consecutive
        ratio points — small non-monotonicities from noise don't count as
        evidence against monotone dilution.
        """
        f1s = self.df["rare_f1"].tolist()
        return all(f1s[i + 1] <= f1s[i] + epsilon for i in range(len(f1s) - 1))

    def find_optimal_ratio(self):
        """Ratio at peak F1, or None if the curve is monotone decreasing
        (in which case there IS no interior optimum — 1:1, or less, is best)."""
        if self.is_monotone_decreasing():
            return None
        best_idx = self.df["rare_f1"].idxmax()
        return float(self.df.loc[best_idx, "ratio"])

    def classify_mechanism(self, other_n_fitter: "RatioCurveFitter" = None) -> str:
        """
        "dilution"  — monotone decreasing at this n (and at the other
                       anchor n, if provided): more synthetic data always
                       hurts more, regardless of quality.
        "mismatch"  — F1 at ratio=1 is already far below the S1 baseline,
                       especially at low n: the synthetic images are
                       harmful even at a gentle mix.
        "mixed"     — neither pattern cleanly holds.
        """
        row_1 = self.df[self.df["ratio"] == 1]
        mono_here = self.is_monotone_decreasing()
        mono_other = other_n_fitter.is_monotone_decreasing() if other_n_fitter is not None else None

        mismatch_at_1 = False
        if len(row_1) and row_1.iloc[0]["s1_baseline_f1"] is not None:
            gap = row_1.iloc[0]["s1_baseline_f1"] - row_1.iloc[0]["rare_f1"]
            mismatch_at_1 = gap > 0.05  # more than 5pp worse than S1 even at the gentlest mix

        if mono_here and (mono_other is None or mono_other):
            return "dilution"
        if mismatch_at_1:
            return "mismatch"
        return "mixed"

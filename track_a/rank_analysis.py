"""
track_a/rank_analysis.py
==========================
Phase 9 (A6) — analysis over the LoRA-rank ablation cells main.py's Step 9
already trains (ranks in config.LORA_RANKS_TO_SWEEP = {4,8,16,32,128}, run
at n in config.ABLATION_N_ANCHORS = {16,128}) and the per-rank KID sweep
Step 9 also now computes (kid_per_class_per_n_sd_lora_r{rank}.json).

Alpha-scaling note (documented once here, not re-derived at each call
site): generative/lora.py's domain_adapt_sd() now defaults alpha to
2 x rank whenever main.py calls it without an explicit alpha (which is
every call site) — so effective LoRA scale (alpha/rank) stays fixed at
2.0 across every rank in {4,8,16,32,128}, rather than silently jumping to
16.0 at rank 4 if alpha had stayed pinned at the old fixed default of 64.
This was fixed directly in generative/lora.py rather than left as a
caller responsibility, since getting it wrong would have meant every
low-rank synthetic pool was trained under a different effective scale
than intended, without any visible symptom short of noticing the numbers
looked off.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import load_cell


def _rank_cell_path(results_dir, dataset, backbone, cls, n, rank) -> Path:
    return Path(results_dir) / f"{dataset}_{backbone}_class{cls}_n{n}_sd_lora_r{rank}.json"


def load_rank_cell(results_dir, dataset, backbone, cls, n, rank) -> dict:
    p = _rank_cell_path(results_dir, dataset, backbone, cls, n, rank)
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return None if d.get("skipped") else d


def load_rank_kid(results_dir, dataset, cls, n, rank):
    p = Path(results_dir) / f"kid_per_class_per_n_sd_lora_r{rank}.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    cell = data.get(str(cls), {}).get(str(n))
    if cell is None or not np.isfinite(cell.get("kid", float("nan"))):
        return None
    return cell["kid"]


class RankAnalysis:
    def __init__(self, results_dir, dataset: str, backbone: str, sweep_classes: list,
                 n_anchors: list, ranks: list):
        self.results_dir = Path(results_dir)
        self.dataset = dataset
        self.backbone = backbone
        self.sweep_classes = sweep_classes
        self.n_anchors = n_anchors
        self.ranks = ranks
        self.df = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        rows = []
        for cls in self.sweep_classes:
            for n in self.n_anchors:
                base = load_cell(self.results_dir, self.dataset, self.backbone, cls, n, "real_only")
                for rank in self.ranks:
                    cell = load_rank_cell(self.results_dir, self.dataset, self.backbone, cls, n, rank)
                    kid  = load_rank_kid(self.results_dir, self.dataset, cls, n, rank)
                    if cell is None:
                        continue
                    rows.append({
                        "class": cls, "n": n, "rank": rank,
                        "rare_f1": cell["f1_target"],
                        "delta_f1": (cell["f1_target"] - base["f1_target"]) if base else None,
                        "kid": kid,
                    })
        return pd.DataFrame(rows)

    def compute_kid_f1_correlation(self) -> dict:
        """Spearman rho between KID and rare_f1 (not DeltaF1 — this is A6's
        own question: does generation quality at a given rank predict
        absolute downstream performance, distinct from A8/RQ6's KID-vs-
        DeltaF1 screening test in kid_screening.py)."""
        from scipy.stats import spearmanr
        d = self.df.dropna(subset=["kid", "rare_f1"])
        if len(d) < 3:
            return {"error": "insufficient paired (kid, rare_f1) points", "n_pairs": len(d)}
        rho, pval = spearmanr(d["kid"], d["rare_f1"])
        return {"spearman_rho": float(rho), "p_value": float(pval), "n_pairs": len(d)}

    def find_optimal_rank_per_n(self) -> dict:
        """{n: rank achieving the highest MEAN rare_f1 across sweep classes
        at that n} — pooled across classes since the question is about the
        rank hyperparameter's general behavior, not one class's quirk."""
        out = {}
        for n in self.n_anchors:
            sub = self.df[self.df["n"] == n]
            if sub.empty:
                out[n] = None
                continue
            means = sub.groupby("rank")["rare_f1"].mean()
            out[n] = int(means.idxmax()) if not means.empty else None
        return out

    def is_rank_monotone(self, n: int, epsilon: float = 0.01) -> dict:
        """Does mean rare_f1 increase monotonically as rank DECREASES at
        this n (the low-rank-avoids-overfitting hypothesis)? Returns
        {"monotone": bool, "direction": "lower_rank_better"|"higher_rank_better"|"non_monotone"}."""
        sub = self.df[self.df["n"] == n]
        if sub.empty:
            return {"monotone": False, "direction": "no_data"}
        means = sub.groupby("rank")["rare_f1"].mean().sort_index()  # ascending rank
        vals = means.tolist()
        increasing_with_rank = all(vals[i + 1] >= vals[i] - epsilon for i in range(len(vals) - 1))
        decreasing_with_rank = all(vals[i + 1] <= vals[i] + epsilon for i in range(len(vals) - 1))
        if decreasing_with_rank and not increasing_with_rank:
            return {"monotone": True, "direction": "lower_rank_better"}
        if increasing_with_rank and not decreasing_with_rank:
            return {"monotone": True, "direction": "higher_rank_better"}
        return {"monotone": False, "direction": "non_monotone"}

    def generate_report(self) -> str:
        lines = [f"Rank ablation report — {self.dataset} / {self.backbone}"]
        for n in self.n_anchors:
            sub = self.df[self.df["n"] == n]
            if sub.empty:
                continue
            kid_row = "  n={:<4} ".format(n) + "  ".join(
                f"rank{r} KID={sub[sub['rank']==r]['kid'].mean():.2f}" for r in self.ranks
                if not sub[sub["rank"] == r].empty
            )
            f1_row = "  n={:<4} ".format(n) + "  ".join(
                f"rank{r} F1={sub[sub['rank']==r]['rare_f1'].mean():.4f}" for r in self.ranks
                if not sub[sub["rank"] == r].empty
            )
            lines.append(kid_row)
            lines.append(f1_row)
        corr = self.compute_kid_f1_correlation()
        lines.append(f"  KID-F1 Spearman rho: {corr}")
        report = "\n".join(lines)
        print(report)
        return report

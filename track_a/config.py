"""
track_a/config.py
==================
Shared argument parsing, global constants, and path conventions for the
Track A multi-dataset study ("When and Why Diffusion-Based Augmentation
Fails for Rare-Class Medical Image Classification").

Unlike gastrovision/config.py, this module is dataset-agnostic — per-dataset
class maps, prompts, and natural class counts live in track_a/datasets/*.py.
This file only holds constants and paths shared across all three datasets
(GastroVision, HAM10000, PathMNIST).

Design decisions encoded here (see track_a_prior_work_review.docx, Section 9,
for the full reasoning behind each):
  - N_GRID = Sagers et al.'s exact sweep {1, 16, 32, 64, 128, 228} (flag 4)
  - The LoRA-rank ablation and the synthetic:real ratio ablation are each run
    at TWO representative n-values, ABLATION_N_ANCHORS = {16, 128}, rather
    than at every grid point — these anchors reuse the same two numbers as
    N_GRID so there's a single set of reference n-values throughout the
    paper (flag 4). Do not confuse this with the rank VALUES being swept —
    those are LORA_RANKS_TO_SWEEP below, an independent set of numbers that
    happen to include 16 but are not "the same 16."
  - BACKBONES narrowed to EfficientNetV2-S + DINOv2 (flag 7)
  - GEN_MODELS is a 2-way comparison: SD+LoRA vs. DCGAN — ControlNet dropped,
    DDPM replaced by DCGAN (flags 5, 8)
  - DATASETS = GastroVision, HAM10000, PathMNIST — CheXpert dropped (flag 1)
"""

import argparse
from pathlib import Path
import numpy as np
import torch


# ==============================================================================
# Study-wide constants (see Section 9 of the review doc for justification)
# ==============================================================================

# Main n-grid, adopted directly from Sagers et al. (arXiv:2308.12453).
# Doubling 16→32→64→128, extreme floor at 1, data-availability ceiling at 228.
N_GRID = [1, 16, 32, 64, 128, 228]

# n-values at which the LoRA-rank and synth:real-ratio ablations are run
# (a subset of N_GRID, not every grid point — those two ablations are run
# per-rank/per-ratio, which multiplies cost fast).
ABLATION_N_ANCHORS = [16, 128]

# Rank values swept in the LoRA-rank ablation (RQ4). Carries over the
# existing reviewer_experiments.py convention (ranks 8/16/32) rather than
# inventing new values; 128 is added as a high-rank point since Track A's
# rank ablation is a first-class RQ, not a reviewer-response afterthought.
# 4 is also included (both extremes tested) to cover the low-rank
# underfitting hypothesis alongside the high-rank overfitting one.
LORA_RANKS_TO_SWEEP = [4, 8, 16, 32, 128]

# Default LoRA rank used for the main (non-ablation) generation runs.
DEFAULT_LORA_RANK = 32

# Extra training seeds used ONLY by the opt-in multiseed re-run
# (track_a/multiseed.py, main.py Step 12) — kept separate from ABLATION_N_ANCHORS'
# n-values entirely; this is a seed COUNT/list, not an n-value. Deliberately
# small (3 total including args.seed) since multiseed retraining is restricted
# to the 3 headline findings (crossover n*, ratio curve at n=16, KID-DeltaF1
# rho), not the full grid — even 3x the cost of those narrow slices is far
# cheaper than 3x the full n-grid x conditions x backbones matrix.
MULTISEED_DEFAULT_SEEDS = [42, 123, 7]

BACKBONES = ["efficientnetv2_rw_s", "dinov2"]

# RQ5 generative-model-generality comparison. 2-way: diffusion (pretrained
# backbone + LoRA) vs. GAN (trained from scratch, per class, unconditional).
GEN_MODELS = ["sd_lora", "dcgan"]

DATASETS = ["gastrovision", "ham10000", "pathmnist"]

# Training conditions run at every (dataset, class, n, backbone) combination
# that is feasible for that class (see subsample.py for feasibility logic).
CONDITIONS = [
    "real_only",       # S1 equivalent — no synthetic data
    "sd_lora_synth",   # real + SD/LoRA-generated synthetic
    "dcgan_synth",     # real + DCGAN-generated synthetic (RQ5 arm)
    "synth_only",      # RQ2a mechanism experiment — no real images for the class
]


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Track A: multi-dataset diffusion-augmentation failure study"
    )

    # Paths — Track A's OWN data/output tree (for HAM10000, PathMNIST, and
    # all of Track A's own synthetic/checkpoint/results output)
    p.add_argument("--data_dir",   default="/data")
    p.add_argument("--output_dir", default="/output/track_a")

    # GastroVision is a special case: it reuses the EXISTING gastrovision/
    # pipeline's splits and image root rather than regenerating them (see
    # track_a/datasets/gastrovision.py's module docstring), so it needs its
    # own paths matching job.yaml's conventions for that pipeline, which are
    # NOT under --data_dir/--output_dir above.
    p.add_argument("--gastrovision_data_dir",   default="/data/gastrovision/data",
                   help="Matches gastrovision job.yaml's --data_dir")
    p.add_argument("--gastrovision_output_dir", default="/data/gastrovision",
                   help="Matches gastrovision job.yaml's --output_dir — "
                        "this is where splits/train.csv etc. already live")
    p.add_argument("--gastrovision_image_root", default="gastrovision_raw/Gastrovision",
                   help="Matches gastrovision job.yaml's --image_root, relative "
                        "to --gastrovision_data_dir")

    # Scope
    p.add_argument("--datasets",  nargs="+", default=list(DATASETS))
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    p.add_argument("--n_grid",    type=int, nargs="+", default=list(N_GRID))

    # Stable Diffusion / LoRA
    p.add_argument("--sd_model_id",        default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora_rank",          type=int,   default=DEFAULT_LORA_RANK)
    p.add_argument("--lora_alpha",         type=int,   default=64)
    p.add_argument("--lora_dropout",       type=float, default=0.1)
    p.add_argument("--domain_adapt_steps", type=int,   default=15000)
    p.add_argument("--sd_batch_size",      type=int,   default=1)
    p.add_argument("--sd_grad_accum",      type=int,   default=16)
    p.add_argument("--sd_lr",              type=float, default=1e-4)
    p.add_argument("--ema_decay",          type=float, default=0.9999)
    p.add_argument("--ema_warmup_steps",   type=int,   default=100)
    p.add_argument("--gen_steps",          type=int,   default=50)
    p.add_argument("--guidance_scale",     type=float, default=7.5)
    p.add_argument("--gen_batch_size",     type=int,   default=1)
    p.add_argument("--samples_per_class",  type=int,   default=250,
                   help="Fixed per-class SD/LoRA synthetic pool size — must "
                        "exceed max(N_GRID)=228 so every grid point's "
                        "synth_ratio=1.0 draw is satisfiable from one pool.")
    p.add_argument("--synth_ratio",        type=float, default=1.0,
                   help="Synthetic:real ratio at each grid point — 1.0 means "
                        "n synthetic images are mixed with n real images.")

    # DCGAN (RQ5 comparison arm)
    p.add_argument("--dcgan_latent_dim",   type=int,   default=100)
    p.add_argument("--dcgan_native_res",   type=int,   default=64,
                   help="DCGAN's own architecture operates at this resolution "
                        "(standard for transposed-conv generators — 224 isn't "
                        "reachable via clean doubling). Generated images are "
                        "resized to --img_size after generation, same as the "
                        "SD/LoRA pipeline's 512→img_size resize, so both arms "
                        "feed the classifier at identical resolution.")
    p.add_argument("--dcgan_epochs",       type=int,   default=2000,
                   help="DCGANs trained per-class-per-n need many epochs at low n; "
                        "epoch count, not step count, since datasets are tiny.")
    p.add_argument("--dcgan_batch_size",   type=int,   default=8)
    p.add_argument("--dcgan_lr",           type=float, default=2e-4)

    # Classifier
    p.add_argument("--img_size",         type=int,   default=224)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--freeze_epochs",    type=int,   default=16)
    p.add_argument("--fine_tune_epochs", type=int,   default=24)
    p.add_argument("--gamma",            type=float, default=2.0)
    p.add_argument("--freeze_lr_mult",   type=float, default=10.0)

    # Ablations / comparisons
    p.add_argument("--run_lora_rank_ablation", action="store_true",
                   help=f"Sweep LoRA rank in {LORA_RANKS_TO_SWEEP} at n in {ABLATION_N_ANCHORS}")
    p.add_argument("--run_dcgan_comparison",   action="store_true",
                   help="Run the RQ5 SD+LoRA vs. DCGAN comparison arm")
    p.add_argument("--run_synthetic_only",     action="store_true",
                   help="Run the RQ2a synthetic-only mechanism condition")
    p.add_argument("--run_pretrain_only",      action="store_true",
                   help="Run the RQ7/A7 s3_pretrain_only condition (stage 1 on "
                        "augmented data, stage 2 switched to real-only) at "
                        "ABLATION_N_ANCHORS — a secondary diagnostic, not part "
                        "of the main n-grid, so it's opt-in like the rank/ratio "
                        "ablations rather than always-on.")
    p.add_argument("--run_ratio_ablation",     action="store_true",
                   help="Run the RQ_A5 synthetic:real ratio ablation at "
                        "ABLATION_N_ANCHORS (track_a/ratio_ablation.py)")
    p.add_argument("--run_multiseed",          action="store_true",
                   help="Retrain the 3 headline findings (crossover n*, ratio "
                        "curve at n=16, KID-DeltaF1 rho) at extra seeds beyond "
                        "--seed, for seed-variance reporting (track_a/multiseed.py)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Extra seeds for --run_multiseed; defaults to "
                        "MULTISEED_DEFAULT_SEEDS if not given. --seed's own "
                        "value is always included automatically.")

    # Evaluation
    p.add_argument("--n_bootstrap",      type=int, default=1000,
                   help="Bootstrap resamples for rare-class F1 CIs")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--min_free_disk_gb", type=float, default=20.0)

    # Execution modes (mirrors gastrovision/config.py's flags)
    p.add_argument("--skip_domain_adapt", action="store_true")
    p.add_argument("--skip_generation",   action="store_true")
    p.add_argument("--skip_training",     action="store_true")
    p.add_argument("--evaluate_only",     action="store_true")

    return p.parse_args()


# ==============================================================================
# Global state
# ==============================================================================

args = parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DATA_DIR   = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)


def dataset_dirs(dataset_name: str) -> dict:
    """
    Per-dataset output directory tree, namespaced under OUTPUT_DIR/<dataset>/
    so the three datasets never collide with each other or with the original
    gastrovision/ pipeline's own /output layout.
    """
    root = OUTPUT_DIR / dataset_name
    d = {
        "root":        root,
        "splits":      root / "splits",
        "synth_sd":    root / "synthetic" / "sd_lora",
        "synth_dcgan": root / "synthetic" / "dcgan",
        "checkpoints": root / "checkpoints",
        "results":     root / "results",
        "logs":        root / "logs",
    }
    for v in d.values():
        v.mkdir(parents=True, exist_ok=True)
    return d

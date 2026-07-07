"""
track_a/main.py
================
Orchestrator — thin glue tying Phases 1-5 together into one resumable run
per dataset. Mirrors gastrovision/main.py's "skip if output already exists"
convention throughout, so a killed/restarted Nautilus job resumes instead
of re-running finished work.

Execution order per dataset:
  1. Load splits (real train/val/test — reused as-is for GastroVision,
     freshly split for HAM10000/PathMNIST)
  2. Feasibility check — print + record which (class, n) cells are
     constructible at all, before spending any compute on infeasible ones
  3. Build the n-grid subsample manifest for this dataset's SWEEP_CLASSES
  4. SD/LoRA domain adaptation (main rank, + rank-ablation ranks if enabled)
  5. SD/LoRA synthetic generation (one fixed pool per class, all ranks run)
  6. DCGAN training + generation, per (class, n) — only if
     --run_dcgan_comparison
  7. KID computation — sd_lora arm always; dcgan arm if comparison enabled
  8. Classifier training matrix — every (backbone, class, n, condition)
     cell, using Phase 5's isolated-per-class design
  9. LoRA-rank ablation classifier runs, at ABLATION_N_ANCHORS only — if
     --run_lora_rank_ablation
  10. A7 s3_pretrain_only, at ABLATION_N_ANCHORS only — if --run_pretrain_only
  11. A5 synthetic:real ratio ablation, at ABLATION_N_ANCHORS only — if
      --run_ratio_ablation
  12. Multiseed retraining of the 3 headline findings only (crossover n*,
      ratio curve at n=16, KID-DeltaF1 rho) — if --run_multiseed

This file does NOT run any analysis (crossover point, KID-F1 correlation,
RQ1/RQ5/RQ8 tables) — that's Phase 7 (track_a/analysis.py), run separately
once the matrix above has produced results.
"""

import json
import itertools
from pathlib import Path

import pandas as pd

from config import args, DEVICE, DATA_DIR, dataset_dirs, ABLATION_N_ANCHORS, LORA_RANKS_TO_SWEEP, MULTISEED_DEFAULT_SEEDS
from datasets import get_dataset
from subsample import build_subsample_manifest, print_feasibility_summary
from generative.lora import domain_adapt_sd, generate_synthetic_for_classes
from generative.dcgan import train_dcgan_for_class, generate_dcgan_images
from quality_metrics import compute_kid_sweep
from classifiers import build_condition_df, train_one_condition, validate_synthetic_labels


def _load_synth_pool(dirs, dataset_name, rank):
    csv_path = dirs["synth_sd"] / f"synthetic_r{rank}.csv"
    return pd.read_csv(csv_path) if csv_path.exists() else None


def run_dataset(dataset_name: str):
    print(f"\n{'='*70}\nTRACK A — {dataset_name}\n{'='*70}")
    ds = get_dataset(dataset_name)
    dirs = dataset_dirs(dataset_name)

    # ------------------------------------------------------------------
    # Step 1: Splits
    # ------------------------------------------------------------------
    if dataset_name == "gastrovision":
        # Reuses the EXISTING gastrovision/ pipeline's splits and image
        # root — a completely different path tree from Track A's own
        # --data_dir/--output_dir (see config.py's --gastrovision_* args
        # and track_a/datasets/gastrovision.py's module docstring).
        splits_source = Path(args.gastrovision_output_dir)
        image_root    = Path(args.gastrovision_data_dir) / args.gastrovision_image_root
        train_df, val_df, test_df = ds.get_splits(splits_source)
    else:
        # HAM10000 / PathMNIST: raw data (or medmnist's download target)
        # lives under Track A's own DATA_DIR, NOT under OUTPUT_DIR — keeps
        # input dataset storage separate from generated checkpoints/results.
        # Per-dataset overrides (--ham10000_data_dir / --pathmnist_data_dir)
        # take precedence when the PVC layout doesn't follow the
        # DATA_DIR/<dataset_name> convention (e.g. HAM10000 nested under
        # gastrovision's own data tree).
        override = {"ham10000": args.ham10000_data_dir, "pathmnist": args.pathmnist_data_dir}.get(dataset_name)
        image_root = Path(override) if override else DATA_DIR / dataset_name
        train_df, val_df, test_df = ds.get_splits(image_root)

    # val/test are never perturbed with synthetic data — always real images
    # only (see train_one_condition's docstring) — but none of the three
    # dataset loaders' get_splits() stamp a `source` column (only
    # build_condition_df does, and only for the training set it builds per
    # cell). TrackAClassifierDataset requires `source` on every df it
    # wraps, so without this val/test crash with KeyError('source') the
    # first time anything actually evaluates against them.
    val_df = val_df.copy();   val_df["source"] = "real"
    test_df = test_df.copy(); test_df["source"] = "real"

    num_classes = len(ds.CLASS_NAMES)
    print(f"  {dataset_name}: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"classes={num_classes}  sweep_classes={ds.SWEEP_CLASSES}")

    real_roots = {"real": image_root}

    # ------------------------------------------------------------------
    # Step 2: Feasibility
    # ------------------------------------------------------------------
    print_feasibility_summary(dataset_name, ds.CLASS_COUNTS, args.n_grid, ds.CLASS_NAMES)

    # ------------------------------------------------------------------
    # Step 3: Subsample manifest (all classes — sweep classes are what
    # main.py actually iterates, but building it for every class costs
    # nothing and keeps the manifest usable for diagnostics/analysis too).
    # ------------------------------------------------------------------
    manifest = build_subsample_manifest(train_df, "label", args.n_grid, seed=args.seed)

    # ------------------------------------------------------------------
    # Step 4: SD/LoRA domain adaptation
    # ------------------------------------------------------------------
    ranks_to_adapt = [args.lora_rank]
    if args.run_lora_rank_ablation:
        ranks_to_adapt = sorted(set(ranks_to_adapt) | set(LORA_RANKS_TO_SWEEP))

    if not args.skip_domain_adapt:
        for rank in ranks_to_adapt:
            print(f"\n--- Step 4: domain adaptation, {dataset_name}, rank={rank} ---")
            domain_adapt_sd(
                dataset_name, train_df, real_roots["real"], dirs["checkpoints"], dirs["results"],
                ds.CLASS_PROMPTS, ds.DOMAIN_PREFIX, args, DEVICE, rank=rank,
            )

    # ------------------------------------------------------------------
    # Step 5: SD/LoRA synthetic generation (fixed per-class pool)
    # ------------------------------------------------------------------
    if not args.skip_generation:
        for rank in ranks_to_adapt:
            print(f"\n--- Step 5: SD/LoRA generation, {dataset_name}, rank={rank} ---")
            generate_synthetic_for_classes(
                dataset_name, dirs["checkpoints"], dirs["synth_sd"],
                ds.SWEEP_CLASSES, ds.CLASS_PROMPTS, ds.DOMAIN_PREFIX, ds.NEGATIVE_PROMPT,
                ds.CLASS_NAMES, args.samples_per_class, args, DEVICE, rank=rank,
            )

    sd_pool_main = _load_synth_pool(dirs, dataset_name, args.lora_rank)
    if sd_pool_main is not None:
        validate_synthetic_labels(sd_pool_main)

    # ------------------------------------------------------------------
    # Step 6: DCGAN training + generation (RQ5 arm)
    # ------------------------------------------------------------------
    dcgan_rows = []
    if args.run_dcgan_comparison:
        print(f"\n--- Step 6: DCGAN training + generation, {dataset_name} ---")
        for cls in ds.SWEEP_CLASSES:
            for n, real_subset in manifest.get(cls, {}).items():
                train_dcgan_for_class(
                    dataset_name, cls, n, real_subset["image_path"].tolist(),
                    real_roots["real"], dirs["checkpoints"], args, DEVICE,
                )
                num_synth = max(1, round(n * args.synth_ratio))
                cell_df = generate_dcgan_images(
                    dataset_name, cls, n, dirs["checkpoints"], dirs["synth_dcgan"],
                    num_synth, ds.CLASS_NAMES.get(cls, str(cls)), args, DEVICE,
                )
                dcgan_rows.append(cell_df)
    dcgan_pool = pd.concat(dcgan_rows, ignore_index=True) if dcgan_rows else None
    if dcgan_pool is not None:
        validate_synthetic_labels(dcgan_pool)

    # ------------------------------------------------------------------
    # Step 7: KID — reference is always the held-out TEST split (fixed,
    # independent of n), never the n-real training subsample. See
    # quality_metrics.py's module docstring for why this changed and what
    # it means for the SD arm's KID becoming n-invariant per class.
    # ------------------------------------------------------------------
    if sd_pool_main is not None:
        print(f"\n--- Step 7a: KID sweep (sd_lora), {dataset_name} ---")
        compute_kid_sweep(
            dataset_name, ds.SWEEP_CLASSES, test_df, args.n_grid, sd_pool_main,
            real_roots["real"], dirs["synth_sd"], dirs["results"], DEVICE,
            class_names=ds.CLASS_NAMES, seed=args.seed, gen_model="sd_lora",
            real_split_name="test",
        )
    if dcgan_pool is not None:
        print(f"\n--- Step 7b: KID sweep (dcgan), {dataset_name} ---")
        compute_kid_sweep(
            dataset_name, ds.SWEEP_CLASSES, test_df, args.n_grid, dcgan_pool,
            real_roots["real"], dirs["synth_dcgan"], dirs["results"], DEVICE,
            class_names=ds.CLASS_NAMES, seed=args.seed, gen_model="dcgan",
            real_split_name="test",
        )

    # ------------------------------------------------------------------
    # Step 8: Classifier training matrix — main n-grid
    # ------------------------------------------------------------------
    if not args.skip_training:
        print(f"\n--- Step 8: classifier matrix, {dataset_name} ---")
        conditions = ["real_only", "synth_only"]
        if sd_pool_main is not None:
            conditions.append("sd_lora_synth")
        if dcgan_pool is not None:
            conditions.append("dcgan_synth")

        for backbone, cls, n, condition in itertools.product(
            args.backbones, ds.SWEEP_CLASSES, args.n_grid, conditions
        ):
            cell_tag = f"{dataset_name}_{backbone}_class{cls}_n{n}_{condition}"
            ckpt_path    = dirs["checkpoints"] / f"{cell_tag}.pt"
            results_path = dirs["results"] / f"{cell_tag}.json"

            if condition == "dcgan_synth":
                synth_pool = dcgan_pool[dcgan_pool["n"] == n] if dcgan_pool is not None else None
            elif condition in ("sd_lora_synth", "synth_only"):
                synth_pool = sd_pool_main
            else:
                synth_pool = None

            cond_df = build_condition_df(
                train_df, "label", cls, n, condition, manifest,
                synth_pool_df=synth_pool, synth_ratio=args.synth_ratio, seed=args.seed,
            )
            if cond_df is None:
                # Infeasible cell (not enough real images for this class at
                # this n) — record why, rather than leaving a silent gap
                # the analysis phase would have to rediscover from absence.
                if not results_path.exists():
                    with open(results_path, "w") as f:
                        json.dump({"skipped": True, "reason": "infeasible_n_for_class"}, f, indent=2)
                continue

            roots = dict(real_roots)
            roots["synth_sd"]    = dirs["synth_sd"]
            roots["synth_dcgan"] = dirs["synth_dcgan"]

            train_one_condition(
                backbone, cond_df, val_df, roots, num_classes, cls, args, DEVICE,
                ckpt_path, results_path,
            )

    # ------------------------------------------------------------------
    # Step 9: LoRA-rank ablation classifier runs — ABLATION_N_ANCHORS only
    # ------------------------------------------------------------------
    if args.run_lora_rank_ablation and not args.skip_training:
        print(f"\n--- Step 9: LoRA-rank ablation, {dataset_name} ---")
        for rank in LORA_RANKS_TO_SWEEP:
            rank_pool = _load_synth_pool(dirs, dataset_name, rank)
            if rank_pool is None:
                print(f"  rank={rank}: no synthetic pool found — skipping")
                continue
            validate_synthetic_labels(rank_pool)

            # KID per rank (needed by RankAnalysis's KID-F1 correlation —
            # the Step 7 KID sweep above only covers the MAIN rank's pool).
            # gen_model="sd_lora_r{rank}" reuses compute_kid_sweep's
            # non-dcgan (label-only) filtering path, just with a
            # rank-specific output filename.
            compute_kid_sweep(
                dataset_name, ds.SWEEP_CLASSES, test_df, ABLATION_N_ANCHORS, rank_pool,
                real_roots["real"], dirs["synth_sd"], dirs["results"], DEVICE,
                class_names=ds.CLASS_NAMES, seed=args.seed, gen_model=f"sd_lora_r{rank}",
                real_split_name="test",
            )

            for backbone, cls, n in itertools.product(
                args.backbones, ds.SWEEP_CLASSES, ABLATION_N_ANCHORS
            ):
                cell_tag = f"{dataset_name}_{backbone}_class{cls}_n{n}_sd_lora_r{rank}"
                ckpt_path    = dirs["checkpoints"] / f"{cell_tag}.pt"
                results_path = dirs["results"] / f"{cell_tag}.json"

                cond_df = build_condition_df(
                    train_df, "label", cls, n, "sd_lora_synth", manifest,
                    synth_pool_df=rank_pool, synth_ratio=args.synth_ratio, seed=args.seed,
                )
                if cond_df is None:
                    if not results_path.exists():
                        with open(results_path, "w") as f:
                            json.dump({"skipped": True, "reason": "infeasible_n_for_class"}, f, indent=2)
                    continue

                roots = dict(real_roots)
                roots["synth_sd"]    = dirs["synth_sd"]
                roots["synth_dcgan"] = dirs["synth_dcgan"]

                train_one_condition(
                    backbone, cond_df, val_df, roots, num_classes, cls, args, DEVICE,
                    ckpt_path, results_path,
                )

    # ------------------------------------------------------------------
    # Step 10: A7 s3_pretrain_only — ABLATION_N_ANCHORS only, opt-in
    # ------------------------------------------------------------------
    if args.run_pretrain_only and not args.skip_training and sd_pool_main is not None:
        from classifiers import train_pretrain_only_condition
        print(f"\n--- Step 10: s3_pretrain_only, {dataset_name} ---")
        for backbone, cls, n in itertools.product(args.backbones, ds.SWEEP_CLASSES, ABLATION_N_ANCHORS):
            stage1_df = build_condition_df(
                train_df, "label", cls, n, "sd_lora_synth", manifest,
                synth_pool_df=sd_pool_main, synth_ratio=args.synth_ratio, seed=args.seed,
            )
            stage2_df = build_condition_df(
                train_df, "label", cls, n, "real_only", manifest, seed=args.seed,
            )
            cell_tag = f"{dataset_name}_{backbone}_class{cls}_n{n}_s3_pretrain_only"
            ckpt_path    = dirs["checkpoints"] / f"{cell_tag}.pt"
            results_path = dirs["results"] / f"{cell_tag}.json"

            if stage1_df is None or stage2_df is None:
                if not results_path.exists():
                    with open(results_path, "w") as f:
                        json.dump({"skipped": True, "reason": "infeasible_n_for_class"}, f, indent=2)
                continue

            roots = dict(real_roots)
            roots["synth_sd"]    = dirs["synth_sd"]
            roots["synth_dcgan"] = dirs["synth_dcgan"]

            train_pretrain_only_condition(
                backbone, stage1_df, stage2_df, val_df, roots, num_classes, cls, args, DEVICE,
                ckpt_path, results_path,
            )

    # ------------------------------------------------------------------
    # Step 11: A5 ratio ablation — ABLATION_N_ANCHORS only, opt-in
    # ------------------------------------------------------------------
    if args.run_ratio_ablation and not args.skip_training and sd_pool_main is not None:
        from ratio_ablation import ensure_synthetic_bank, run_ratio_ablation
        print(f"\n--- Step 11: ratio ablation, {dataset_name} ---")
        bank = ensure_synthetic_bank(dataset_name, ds, dirs, args, DEVICE)
        if bank is not None:
            run_ratio_ablation(
                dataset_name, ds, dirs, real_roots["real"], train_df, val_df,
                manifest, bank, num_classes, args, DEVICE,
            )

    # ------------------------------------------------------------------
    # Step 12: multiseed retraining — 3 headline findings only, opt-in
    # ------------------------------------------------------------------
    if args.run_multiseed and not args.skip_training:
        from multiseed import run_multiseed_crossover_training, run_multiseed_ratio_training
        seeds = sorted(set((args.seeds or MULTISEED_DEFAULT_SEEDS) + [args.seed]))
        print(f"\n--- Step 12: multiseed retraining, {dataset_name}, seeds={seeds} ---")

        run_multiseed_crossover_training(
            dataset_name, ds, dirs, real_roots["real"], train_df, val_df,
            num_classes, args, DEVICE, seeds, sd_pool_main=sd_pool_main,
        )

        if sd_pool_main is not None:
            from ratio_ablation import ensure_synthetic_bank
            # Reuses the same top-up helper Step 11 uses, restricted to the
            # single ratio_n_anchor (default ABLATION_N_ANCHORS[0]=16) this
            # module operates on, so the bank is guaranteed big enough even
            # if samples_per_class was left at a value smaller than
            # n=16 x max(RATIO_SET)=160.
            bank_for_ratio = ensure_synthetic_bank(
                dataset_name, ds, dirs, args, DEVICE, n_anchors=[ABLATION_N_ANCHORS[0]],
            )
            if bank_for_ratio is not None:
                run_multiseed_ratio_training(
                    dataset_name, ds, dirs, real_roots["real"], train_df, val_df,
                    num_classes, args, DEVICE, seeds, sd_pool=bank_for_ratio,
                )

    print(f"\n✅ {dataset_name} done.")


def main():
    print("=" * 70)
    print("TRACK A — Multi-Dataset Diffusion-Augmentation-Failure Study")
    print("=" * 70)
    print(f"  Datasets:  {args.datasets}")
    print(f"  Backbones: {args.backbones}")
    print(f"  N_GRID:    {args.n_grid}")
    print(f"  DCGAN comparison:    {args.run_dcgan_comparison}")
    print(f"  LoRA rank ablation:  {args.run_lora_rank_ablation}")
    print(f"  Pretrain-only (A7):  {args.run_pretrain_only}")
    print(f"  Ratio ablation (A5): {args.run_ratio_ablation}")
    print(f"  Multiseed:           {args.run_multiseed}")

    for dataset_name in args.datasets:
        run_dataset(dataset_name)

    print("\n✅ Track A pipeline complete.")


if __name__ == "__main__":
    main()

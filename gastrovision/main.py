"""
main.py
=======
Pipeline orchestration — thin glue between modules.

Execution order:
  1. Data splits
  2. Baseline training (S1)
  3. Heavy-aug training (S2)
  4. SD domain adaptation
  5. Synthetic generation
  6. Augmented training (S3)
  7. Evaluation: val set (S1 / S2 / S3)
  8. Few-shot evaluation (CLIP + ProtoNet) for ultra-rare classes
  9. Ablation studies
 10. Final test set evaluation
 11. Generate all paper figures
"""

import json
import shutil
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Disable Flash Attention 2 globally — FA2 requires SM80+ (Ampere/Hopper).
# On Turing nodes (RTX 2080 Ti, SM75) PyTorch's SDPA selects FA2 during the
# UNet backward pass and raises: RuntimeError: Expected is_sm80 || is_sm90.
# Fall back to memory-efficient attention which works on SM70+.
try:
    import torch as _torch
    _torch.backends.cuda.enable_flash_sdp(False)
    _torch.backends.cuda.enable_mem_efficient_sdp(True)
    del _torch
except Exception:
    pass

from config import (
    args, OUTPUT_DIR, SPLITS_DIR, CKPT_DIR, RESULTS_DIR, SYNTH_DIR,
    NUM_CLASSES, RARE_CLASSES, ULTRA_RARE, LABEL_MAP, REV_LABEL_MAP,
    CLASS_NAMES,
)
import config

from dataset  import create_splits, GastroVisionDataset
from train    import train_classifier, train_classifier_heavy_aug, tune_classifier
from generate import domain_adapt_sd, generate_synthetic
from evaluate import evaluate_all, evaluate_heavy_aug, evaluate_test, compute_fid
from ensemble import ConfidenceEnsemble
from ablation import (
    ablation_ensemble_subsets, ablation_sampling,
    ablation_loss_function, ablation_synth_count, print_ablation_summary,
)
from fewshot  import run_clip_zeroshot, train_prototypical, eval_prototypical_on_rare
from visualize import generate_all_figures
from models   import load_checkpoint


# ==============================================================================
# Helpers
# ==============================================================================

def _normalize_path(p, base_dir):
    from pathlib import Path as P
    p = P(p)
    if not p.is_absolute():
        from config import IMAGE_ROOT_DIR
        if (IMAGE_ROOT_DIR / p).exists():
            return (IMAGE_ROOT_DIR / p).resolve()
        elif (OUTPUT_DIR / p).exists():
            return (OUTPUT_DIR / p).resolve()
        return p
    return p.resolve()


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 65)
    print("GastroVision Pipeline — Tiered Ensemble + Few-Shot")
    print("=" * 65)
    print(f"  Models:               {args.models}")
    print(f"  Rare threshold:       n < {args.rare_threshold}")
    print(f"  Ultra-rare threshold: n < {args.ultraRare_threshold}")
    print(f"  Samples per class:    {args.samples_per_class}")
    print(f"  Gen steps / guidance: {args.gen_steps} / {args.guidance_scale}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Data splits
    # ------------------------------------------------------------------
    train_csv = SPLITS_DIR / args.train_csv
    val_csv   = SPLITS_DIR / args.val_csv
    test_csv  = SPLITS_DIR / args.test_csv

    if not train_csv.exists():
        print("Creating splits...")
        train_df, val_df, test_df, _ = create_splits()
    else:
        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        test_df  = pd.read_csv(test_csv)

        all_labels = sorted(
            set(train_df.get("original_label", train_df["label"]).unique())
            | set(val_df.get("original_label",   val_df["label"]).unique())
            | set(test_df.get("original_label", test_df["label"]).unique())
        )
        config.LABEL_MAP     = {orig: i for i, orig in enumerate(all_labels)}
        config.REV_LABEL_MAP = {i: orig for orig, i in config.LABEL_MAP.items()}
        config.NUM_CLASSES   = len(all_labels)

        if "original_label" in train_df.columns:
            for df_ in (train_df, val_df, test_df):
                df_["label"] = df_["original_label"].map(config.LABEL_MAP)

        rare_orig      = [c for c in all_labels
                          if len(train_df[train_df.get("original_label", train_df["label"]) == c])
                          < args.rare_threshold]
        ultra_rare_orig = [c for c in all_labels
                           if len(train_df[train_df.get("original_label", train_df["label"]) == c])
                           < args.ultraRare_threshold]

        config.RARE_CLASSES = sorted([config.LABEL_MAP[c] for c in rare_orig])
        config.ULTRA_RARE   = sorted([config.LABEL_MAP[c] for c in ultra_rare_orig])

    print(f"NUM_CLASSES={config.NUM_CLASSES}  "
          f"RARE={config.RARE_CLASSES}  ULTRA_RARE={config.ULTRA_RARE}")

    if args.evaluate_only:
        evaluate_all(augmented=False)
        evaluate_heavy_aug()
        evaluate_all(augmented=True)
        evaluate_test()
        if args.run_fewshot:
            run_clip_zeroshot("test")
        generate_all_figures()
        return

    # ------------------------------------------------------------------
    # Step 2: Train S1 baselines
    # ------------------------------------------------------------------
    if not args.skip_training:
        print("\n" + "="*65 + "\nStep 2: Training S1 baselines\n" + "="*65)

        hparams_path = OUTPUT_DIR / "best_hparams.json"
        if hparams_path.exists():
            with open(hparams_path) as f:
                saved = json.load(f)
            from config import HPARAMS
            for k, v in saved.items():
                if k in HPARAMS:
                    HPARAMS[k].update(v)

        for name in args.models:
            ckpt = CKPT_DIR / f"sota_{name}.pt"
            if ckpt.exists():
                print(f"  ✅ {name} baseline exists — skipping")
                continue
            if args.tune:
                tune_classifier(name, train_csv, val_csv,
                                n_trials=args.tune_trials, tune_epochs=args.tune_epochs)
            train_classifier(name, train_csv, val_csv, augmented=False)

    # ------------------------------------------------------------------
    # Step 3: Train S2 heavy augmentation
    # ------------------------------------------------------------------
    if not args.skip_training:
        print("\n" + "="*65 + "\nStep 3: Training S2 heavy augmentation\n" + "="*65)
        for name in args.models:
            ckpt = CKPT_DIR / f"sota_{name}_heavy.pt"
            if ckpt.exists():
                print(f"  ✅ {name} heavy aug exists — skipping")
                continue
            train_classifier_heavy_aug(name, train_csv, val_csv)

    # ------------------------------------------------------------------
    # Step 4: SD domain adaptation
    # ------------------------------------------------------------------
    ema_adapter = CKPT_DIR / "sd_gastrovision_lora_ema_adapter"
    if not args.skip_domain_adapt:
        if ema_adapter.exists():
            print("\nStep 4: EMA adapter already exists — skipping domain adaptation")
        else:
            print("\n" + "="*65 + "\nStep 4: SD LoRA domain adaptation\n" + "="*65)
            domain_adapt_sd()

    # ------------------------------------------------------------------
    # Step 5: Synthetic generation
    # ------------------------------------------------------------------
    synth_csv_path = SYNTH_DIR / "synthetic_train.csv"
    if not args.skip_generation:
        already_done = all(
            len(list((SYNTH_DIR / str(cls)).glob("synth_*.png"))) >= args.samples_per_class
            for cls in config.RARE_CLASSES
        ) if config.RARE_CLASSES else False

        if already_done and synth_csv_path.exists():
            print("\nStep 5: Generation already complete — skipping")
            synth_df = pd.read_csv(synth_csv_path)
        else:
            print("\n" + "="*65 + "\nStep 5: Generating synthetic images\n" + "="*65)
            synth_df = generate_synthetic()
    else:
        synth_df = pd.read_csv(synth_csv_path) if synth_csv_path.exists() else None

    # ------------------------------------------------------------------
    # Step 6: Train S3 (augmented with synthetic)
    # ------------------------------------------------------------------
    if not args.skip_training and synth_df is not None:
        print("\n" + "="*65 + "\nStep 6: Building augmented dataset + retraining (S3)\n" + "="*65)

        # Leakage check
        val_abs   = {_normalize_path(p, None) for p in pd.read_csv(val_csv)["image_path"]}
        test_abs  = {_normalize_path(p, None) for p in pd.read_csv(test_csv)["image_path"]}
        synth_abs = {_normalize_path(p, None) for p in synth_df["image_path"]}
        overlap   = (synth_abs & val_abs) | (synth_abs & test_abs)
        if overlap:
            print(f"  ⚠ Leakage detected: {len(overlap)} synthetic images overlap with val/test — removing")
            synth_df = synth_df[~synth_df["image_path"].apply(
                lambda p: _normalize_path(p, None) in overlap
            )]
        else:
            print("  ✅ No leakage detected")

        aug_csv = SPLITS_DIR / args.aug_train_csv
        if not aug_csv.exists():
            aug_df = pd.concat(
                [train_df[["image_path", "label", "class_name"]], synth_df],
                ignore_index=True,
            )
            aug_df.to_csv(aug_csv, index=False)
            print(f"  Augmented dataset: {len(train_df)} real + {len(synth_df)} synth = {len(aug_df)}")
        else:
            print(f"  Augmented CSV exists — reusing {aug_csv.name}")

        for name in args.models:
            ckpt = CKPT_DIR / f"sota_{name}_aug.pt"
            if ckpt.exists():
                print(f"  ✅ {name} augmented exists — skipping")
                continue
            train_classifier(name, aug_csv, val_csv, augmented=True)

    # ------------------------------------------------------------------
    # Step 7: Validation evaluation (S1 / S2 / S3)
    # ------------------------------------------------------------------
    print("\n" + "="*65 + "\nStep 7: Validation evaluation\n" + "="*65)

    s1_path = RESULTS_DIR / "eval_results.json"
    if not s1_path.exists():
        results_s1 = evaluate_all(augmented=False)
    else:
        print("  ✅ S1 val evaluation exists — skipping")

    s2_path = RESULTS_DIR / "eval_results_heavy.json"
    if not s2_path.exists() and any((CKPT_DIR / f"sota_{n}_heavy.pt").exists() for n in args.models):
        results_s2 = evaluate_heavy_aug()
    elif s2_path.exists():
        print("  ✅ S2 val evaluation exists — skipping")

    s3_path = RESULTS_DIR / "eval_results_aug.json"
    if not s3_path.exists() and synth_df is not None:
        results_s3 = evaluate_all(augmented=True)
    elif s3_path.exists():
        print("  ✅ S3 val evaluation exists — skipping")

    # ------------------------------------------------------------------
    # Step 8: Few-shot evaluation (CLIP + ProtoNet) on ultra-rare classes
    # ------------------------------------------------------------------
    if args.run_fewshot and config.ULTRA_RARE:
        print("\n" + "="*65 + "\nStep 8: Few-shot evaluation on ultra-rare classes\n" + "="*65)

        clip_val_path = RESULTS_DIR / "fewshot_clip_val.json"
        if not clip_val_path.exists():
            print("\n[8a] CLIP zero-shot...")
            run_clip_zeroshot(split="val")
        else:
            print("  ✅ CLIP val results exist — skipping")

        proto_path = RESULTS_DIR / "proto_net.pt"
        proto_eval_path = RESULTS_DIR / "proto_eval_val.json"
        if not proto_eval_path.exists():
            print("\n[8b] Prototypical Network...")
            proto = train_prototypical(
                str(train_csv), str(val_csv),
                n_episodes=2000, n_way=min(10, len(config.ULTRA_RARE) + 5),
                k_shot=5, q_query=10,
            )
            eval_prototypical_on_rare(proto, str(train_csv), str(val_csv))
        else:
            print("  ✅ ProtoNet val results exist — skipping")

    # ------------------------------------------------------------------
    # Step 9: Ablation studies
    # ------------------------------------------------------------------
    if args.run_ablations:
        print("\n" + "="*65 + "\nStep 9: Ablation studies\n" + "="*65)
        best_model = args.models[0]

        if not (RESULTS_DIR / "ablation_ensemble.json").exists():
            ablation_ensemble_subsets(val_csv, suffix="")
        else:
            print("  ✅ Ensemble ablation exists — skipping")

        if (CKPT_DIR / f"sota_{best_model}_aug.pt").exists() and \
                not (RESULTS_DIR / "ablation_ensemble_aug.json").exists():
            ablation_ensemble_subsets(val_csv, suffix="_aug")

        if not (RESULTS_DIR / f"ablation_sampling_{best_model}.json").exists():
            ablation_sampling(best_model, train_csv, val_csv, epochs=10)
        else:
            print("  ✅ Sampling ablation exists — skipping")

        if not (RESULTS_DIR / f"ablation_loss_{best_model}.json").exists():
            ablation_loss_function(best_model, train_csv, val_csv, epochs=10)
        else:
            print("  ✅ Loss ablation exists — skipping")

        if synth_csv_path.exists() and \
                not (RESULTS_DIR / f"ablation_synth_count_{best_model}.json").exists():
            ablation_synth_count(best_model, train_csv, synth_csv_path, val_csv)

        print_ablation_summary()

    # ------------------------------------------------------------------
    # Step 10: Final test set evaluation
    # ------------------------------------------------------------------
    print("\n" + "="*65 + "\nStep 10: Final held-out test evaluation\n" + "="*65)

    test_path = RESULTS_DIR / "test_results.json"
    if not test_path.exists():
        evaluate_test()
    else:
        print("  ✅ Test evaluation exists — skipping")

    if args.run_fewshot:
        clip_test_path = RESULTS_DIR / "fewshot_clip_test.json"
        if not clip_test_path.exists():
            run_clip_zeroshot(split="test")
        else:
            print("  ✅ CLIP test results exist — skipping")

    # ------------------------------------------------------------------
    # Step 11: Generate all paper figures
    # ------------------------------------------------------------------
    print("\n" + "="*65 + "\nStep 11: Generating paper figures\n" + "="*65)
    try:
        best_name = args.models[0]
        model = load_checkpoint(best_name, suffix="")
        generate_all_figures(model=model, model_name=best_name)
    except Exception as e:
        print(f"  Figure generation error: {e}")
        generate_all_figures()

    # ------------------------------------------------------------------
    # Strategy comparison summary
    # ------------------------------------------------------------------
    print("\n" + "="*65 + "\nSTRATEGY COMPARISON (validation set)\n" + "="*65)
    try:
        s_files = {
            "S1: Real only":    RESULTS_DIR / "eval_results.json",
            "S2: Heavy aug":    RESULTS_DIR / "eval_results_heavy.json",
            "S3: SD synthetic": RESULTS_DIR / "eval_results_aug.json",
        }
        print(f"\n{'Strategy':<22} {'Model':<33} {'Acc':>8}  {'Mean F1':>8}  {'Rare F1':>8}  {'ECE':>7}")
        print("-" * 90)
        for strat, path in s_files.items():
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            for nm, res in data.items():
                if nm.startswith("_"):
                    continue
                mark = " ◄" if nm == "ensemble" else ""
                print(f"  {strat:<20} {nm:<33} {res['acc']:>8.4f}  "
                      f"{res['f1_mean']:>8.4f}  {res['f1_rare']:>8.4f}  "
                      f"{res.get('ece', float('nan')):>7.4f}{mark}")
            print()
    except Exception as e:
        print(f"  Could not print summary: {e}")

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()

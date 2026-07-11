"""
track_a/generative/lora.py
===========================
Dataset-parameterized version of gastrovision/generate.py's
domain_adapt_sd() / generate_synthetic(). Same diffusers+peft LoRA
fine-tuning approach, same SNR-weighted MSE loss, same EMA adapter — but
takes a dataset name, DataFrame, prompts, and rank as explicit arguments
instead of importing them from gastrovision/config.py. This lets the same
two functions run domain adaptation and generation for GastroVision,
HAM10000, and PathMNIST, and at both LoRA ranks used in the rank ablation
(see track_a/config.py's LORA_RANKS_TO_SWEEP / ABLATION_N_ANCHORS).

Two calls per dataset are expected:
  1. domain_adapt_sd(...)     — ONE call, on the full training set (all
                                  classes), producing one adapter per
                                  (dataset, rank).
  2. generate_synthetic_for_classes(...) — called once per (dataset, rank)
                                  with the dataset's SWEEP_CLASSES, producing
                                  a synthetic image pool per class.

Adapter/checkpoint naming: sd_{dataset}_lora_r{rank}[_ema]_adapter — the
rank is always in the path, so the main run (DEFAULT_LORA_RANK=32) and the
two rank-ablation runs (16, 128, plus 8) never collide with each other.
"""

import gc
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageEnhance

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from diffusers import (
    StableDiffusionPipeline, DDPMScheduler,
    UNet2DConditionModel, AutoencoderKL,
)
from diffusers.optimization import get_scheduler as get_diffusers_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from generative.ema import EMAModel, snr_weights
from generative.sd_dataset import DiffusionDataset


def _postprocess(img: Image.Image, sharpen: float = 1.4, contrast: float = 1.15) -> Image.Image:
    return ImageEnhance.Contrast(ImageEnhance.Sharpness(img).enhance(sharpen)).enhance(contrast)


# ==============================================================================
# Domain adaptation
# ==============================================================================

def domain_adapt_sd(dataset_name: str, train_df, data_dir, ckpt_dir, results_dir,
                     class_prompts: dict, domain_prefix: str, args, device,
                     rank: int = None, alpha: int = None) -> float:
    """
    Fine-tunes SD v1.5 with LoRA on the FULL training set of one dataset
    (all classes, not just the sweep classes) — this is the one-per-dataset
    domain-adaptation step, run once before any per-class/per-n generation.

    alpha defaults to 2 x rank when not explicitly passed — NOT
    args.lora_alpha's fixed default (64) — so effective LoRA scale
    (alpha/rank) stays constant at 2.0 across every rank in the rank
    ablation (RQ4/A6). Passing rank=4 with a fixed alpha=64 would silently
    change the effective scale to 16.0, a genuinely different training
    regime disguised as "just a smaller adapter" — this is exactly the
    mistake the rank ablation exists to avoid making by accident.
    Explicitly pass alpha=args.lora_alpha at the call site if you actually
    want the old fixed-alpha behavior (e.g. to deliberately study the
    scale confound rather than control for it).

    Returns the final smoothed training loss (also written to
    results_dir/sd_loss_{dataset_name}_r{rank}.png).
    """
    rank = rank if rank is not None else args.lora_rank
    alpha = alpha if alpha is not None else 2 * rank
    adapter_name = f"sd_{dataset_name}_lora_r{rank}"
    ckpt_dir = Path(ckpt_dir)

    ema_path = ckpt_dir / f"{adapter_name}_ema_adapter"
    if ema_path.exists():
        print(f"  [{dataset_name} r{rank}] EMA adapter already exists — skipping domain adaptation")
        return float("nan")

    print(f"Loading SD components for {dataset_name} (rank={rank})...")
    vram_gb     = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    offload_cpu = vram_gb < 20
    print(f"  GPU VRAM: {vram_gb:.0f}GB  |  CPU offload: {offload_cpu}")

    tokenizer    = CLIPTokenizer.from_pretrained(args.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model_id, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(args.sd_model_id, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(args.sd_model_id, subfolder="unet")
    noise_sched  = DDPMScheduler.from_pretrained(args.sd_model_id, subfolder="scheduler")

    if offload_cpu:
        unet = unet.to(device)
        text_encoder = text_encoder.cpu()
        vae = vae.cpu()
    else:
        text_encoder = text_encoder.to(device)
        vae = vae.to(device)
        unet = unet.to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=args.lora_dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    ema     = EMAModel(unet, decay=args.ema_decay, update_after_step=args.ema_warmup_steps)
    dataset = DiffusionDataset(train_df, data_dir, tokenizer, class_prompts, domain_prefix)
    loader  = DataLoader(dataset, batch_size=args.sd_batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)

    opt     = torch.optim.AdamW(unet.parameters(), lr=args.sd_lr, weight_decay=1e-4)
    lrsched = get_diffusers_scheduler(
        "cosine", optimizer=opt,
        num_warmup_steps=500, num_training_steps=args.domain_adapt_steps,
    )
    scaler  = GradScaler()

    resume_path = ckpt_dir / f"resume_{adapter_name}.pt"
    step = 0; losses = []

    if resume_path.exists():
        ck = torch.load(resume_path, map_location=device)
        unet.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        lrsched.load_state_dict(ck["scheduler"])
        step   = ck["global_step"]
        losses = ck.get("losses", [])
        if "ema" in ck:
            ema.load_state_dict(ck["ema"])
        print(f"Resumed at step {step}/{args.domain_adapt_steps}")

    print(f"\nDomain adaptation [{dataset_name} r{rank}]: {len(dataset)} images, "
          f"{step}→{args.domain_adapt_steps} steps")
    unet.train(); opt.zero_grad(); it = iter(loader); rl = 0.0

    while step < args.domain_adapt_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)

        pv = batch["pixel_values"].to(device)
        ii = batch["input_ids"].to(device)

        vae_on_gpu = next(vae.parameters()).device == device
        if not vae_on_gpu:
            vae.to(device)
        with torch.no_grad():
            lat = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor
        if offload_cpu:
            vae.cpu(); torch.cuda.empty_cache()

        noise = torch.randn_like(lat)
        t     = torch.randint(0, noise_sched.config.num_train_timesteps,
                              (lat.shape[0],), device=device).long()
        w     = snr_weights(noise_sched, t, device)
        nl    = noise_sched.add_noise(lat, noise, t)

        te_on_gpu = next(text_encoder.parameters()).device == device
        if not te_on_gpu:
            text_encoder.to(device)
        with torch.no_grad():
            hs = text_encoder(ii)[0]
        if offload_cpu:
            text_encoder.cpu(); torch.cuda.empty_cache()

        with autocast():
            pred = unet(nl, t, hs).sample
            lps  = F.mse_loss(pred, noise, reduction="none").mean(dim=[1, 2, 3])
            loss = (lps * w).mean() / args.sd_grad_accum

        scaler.scale(loss).backward()
        rl += loss.item() * args.sd_grad_accum

        if (step + 1) % args.sd_grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt); scaler.update(); lrsched.step(); opt.zero_grad()
            ema.step(unet)

        step += 1
        if step % 100 == 0:
            avg = rl / 100; losses.append(avg); rl = 0.0
            print(f"  [{dataset_name} r{rank}] Step {step:5d}/{args.domain_adapt_steps}  "
                  f"loss={avg:.4f}  lr={opt.param_groups[0]['lr']:.2e}")
        if step % 500 == 0:
            torch.save({
                "state_dict": unet.state_dict(), "optimizer": opt.state_dict(),
                "scheduler":  lrsched.state_dict(), "global_step": step,
                "losses":     losses, "ema": ema.state_dict(),
            }, resume_path)

    torch.save(unet.state_dict(), ckpt_dir / f"{adapter_name}.pt")
    unet.save_pretrained(ckpt_dir / f"{adapter_name}_adapter")
    ema.save_adapter(unet, ema_path)

    final = losses[-1] if losses else float("nan")
    print(f"\nDomain adaptation done [{dataset_name} r{rank}] — final loss: {final:.4f}")
    if final > 0.08:
        print(f"  ⚠ Loss > 0.08 — consider more steps (current: {args.domain_adapt_steps})")

    if losses:
        fig, ax = plt.subplots(figsize=(12, 4))
        xs = [i * 100 for i in range(1, len(losses) + 1)]
        ax.plot(xs, losses, color="#4878cf", linewidth=1.5, label="Loss")
        if len(losses) > 10:
            w = max(5, len(losses) // 20)
            sm = np.convolve(losses, np.ones(w) / w, mode="valid")
            ax.plot(xs[w-1:], sm, color="#d65f5f", linewidth=2.0, alpha=0.8, label="Smoothed")
        ax.axhline(0.05, color="#6acc65", linestyle="--", alpha=0.7, label="Target 0.05")
        ax.axhline(0.08, color="#f0a500", linestyle="--", alpha=0.7, label="Acceptable 0.08")
        ax.set_xlabel("Step"); ax.set_ylabel("SNR-weighted MSE Loss")
        ax.set_title(f"SD LoRA Domain Adaptation Loss — {dataset_name} (rank {rank})")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / f"sd_loss_{dataset_name}_r{rank}.png", dpi=150, bbox_inches="tight")
        plt.close()

    del unet, vae, text_encoder, tokenizer
    torch.cuda.empty_cache(); gc.collect()
    return final


# ==============================================================================
# Synthetic image generation
# ==============================================================================

def generate_synthetic_for_classes(dataset_name: str, ckpt_dir, out_dir,
                                    classes: list, class_prompts: dict, domain_prefix: str,
                                    negative_prompt: str, class_names: dict,
                                    samples_per_class: int, args, device,
                                    rank: int = None) -> pd.DataFrame:
    """
    Generates `samples_per_class` synthetic images for each class in
    `classes` (typically a dataset's SWEEP_CLASSES) using the EMA LoRA
    adapter fine-tuned by domain_adapt_sd() for this (dataset, rank).

    Note this generates ONE pool per class, independent of the n-grid — the
    n-grid mixing (how many synthetic images get combined with how many
    real images at each grid point) is decided later by the classifier
    training wrapper (Phase 5), not here.
    """
    rank = rank if rank is not None else args.lora_rank
    adapter_name = f"sd_{dataset_name}_lora_r{rank}"
    ckpt_dir = Path(ckpt_dir)
    out_dir  = Path(out_dir)

    free_gb = shutil.disk_usage(out_dir).free / (1024 ** 3)
    if free_gb < args.min_free_disk_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free_gb:.1f} GB free, "
            f"minimum {args.min_free_disk_gb} GB required."
        )
    print(f"Disk check: {free_gb:.1f} GB free")

    ema_adapter = ckpt_dir / f"{adapter_name}_ema_adapter"
    raw_adapter = ckpt_dir / f"{adapter_name}_adapter"
    adapter     = ema_adapter if ema_adapter.exists() else raw_adapter
    if not adapter.exists():
        raise FileNotFoundError(
            f"No LoRA adapter found for {dataset_name} r{rank} — run domain_adapt_sd() first."
        )

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, adapter)
    pipe.unet.eval()
    pipe.enable_attention_slicing()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    if vram_gb < 20:
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception as e:
            print(f"  CPU offload unavailable: {e}")
    else:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for cls in classes:
        cls_name = class_names.get(cls, f"class_{cls}")
        cls_dir  = out_dir / str(cls) / f"r{rank}"
        cls_dir.mkdir(parents=True, exist_ok=True)

        # One-time migration for pre-existing pools generated BEFORE this
        # fix: every rank used to write into a shared, non-rank-specific
        # folder (out_dir/{cls}/synth_*.png), so any rank OTHER than
        # whichever one ran first silently found that folder "already
        # complete" and skipped generating its own images entirely —
        # producing identical images (and identical KID) across every rank
        # in an ablation run. A plain (non-ablation) main.py invocation
        # only ever generates ONE rank (args.lora_rank), so any pre-existing
        # flat-layout images can only legitimately belong to that rank —
        # migrate them into its new namespaced folder rather than
        # wastefully regenerating already-good work. Every OTHER rank gets
        # an honestly empty folder here and generates for real.
        legacy_flat_dir = out_dir / str(cls)
        if rank == args.lora_rank and not any(cls_dir.glob("synth_*.png")):
            legacy_images = sorted(legacy_flat_dir.glob("synth_*.png"))
            if len(legacy_images) >= samples_per_class:
                print(f"  [{dataset_name} r{rank}] migrating {len(legacy_images)} pre-existing "
                      f"class {cls} images into rank-specific folder (one-time)")
                for p in legacy_images:
                    p.rename(cls_dir / p.name)

        prompt = domain_prefix + class_prompts.get(cls, f"medical image, {cls_name}")

        tokens = pipe.tokenizer(prompt, return_tensors="pt", truncation=False)
        n_tok  = tokens.input_ids.shape[1]
        if n_tok > 77:
            print(f"  ⚠ Prompt for class {cls} is {n_tok} tokens (>77) — CLIP will truncate")

        existing = sorted(cls_dir.glob("synth_*.png"))
        for p in existing:
            rows.append({"image_path": str(p.relative_to(out_dir)),
                         "label": cls, "class_name": cls_name, "source": "sd_lora_ema"})
        if len(existing) >= samples_per_class:
            print(f"  [{dataset_name} r{rank}] Class {cls}: already complete ({len(existing)} images)")
            continue

        start = len(existing)
        print(f"\n[{dataset_name} r{rank}] Class {cls} ({cls_name}): generating "
              f"{samples_per_class - start} images  [{prompt[:70]}...]")
        idx = start

        while idx < samples_per_class:
            n = min(args.gen_batch_size, samples_per_class - idx)
            torch.cuda.empty_cache(); gc.collect()
            gens = [
                torch.Generator(device=device).manual_seed(args.seed + cls * 100000 + idx + i)
                for i in range(n)
            ]
            with torch.no_grad():
                imgs = pipe(
                    prompt=[prompt] * n,
                    negative_prompt=[negative_prompt] * n,
                    num_inference_steps=args.gen_steps,
                    guidance_scale=args.guidance_scale,
                    height=512, width=512, generator=gens,
                ).images
            for img in imgs:
                img  = _postprocess(img.resize((args.img_size, args.img_size), Image.LANCZOS))
                path = cls_dir / f"synth_{idx:05d}.png"
                img.save(path)
                rows.append({"image_path": str(path.relative_to(out_dir)),
                             "label": cls, "class_name": cls_name, "source": "sd_lora_ema"})
                idx += 1
            if idx % 50 == 0 or idx >= samples_per_class:
                print(f"  {idx}/{samples_per_class}")
        print(f"  Class {cls} done")

    del pipe; torch.cuda.empty_cache(); gc.collect()

    synth_df = pd.DataFrame(rows)
    synth_df.to_csv(out_dir / f"synthetic_r{rank}.csv", index=False)
    print(f"\n[{dataset_name} r{rank}] {len(synth_df)} synthetic images → "
          f"{out_dir / f'synthetic_r{rank}.csv'}")
    return synth_df

"""
generate.py
===========
Stable Diffusion LoRA domain adaptation and synthetic image generation.
"""

import gc
import shutil
import json
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

import config as _config
from config import (
    args, DEVICE, OUTPUT_DIR, CKPT_DIR, SPLITS_DIR, RESULTS_DIR,
    CLASS_PROMPTS, DOMAIN_PREFIX, NEGATIVE_PROMPT, SYNTH_DIR,
)
# RARE_CLASSES and REV_LABEL_MAP are populated after splits — access via _config.X
from dataset import GastroVisionSDDataset
from train import EMAModel, _snr_weights


# ==============================================================================
# Post-processing
# ==============================================================================

def _postprocess(img: Image.Image, sharpen: float = 1.4, contrast: float = 1.15) -> Image.Image:
    return ImageEnhance.Contrast(ImageEnhance.Sharpness(img).enhance(sharpen)).enhance(contrast)


# ==============================================================================
# Domain adaptation
# ==============================================================================

def domain_adapt_sd():
    """Fine-tune SD v1.5 with LoRA on the full GastroVision training set."""
    train_csv = SPLITS_DIR / args.train_csv

    print("Loading SD components...")
    vram_gb    = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    offload_cpu = vram_gb < 20
    print(f"  GPU VRAM: {vram_gb:.0f}GB  |  CPU offload: {offload_cpu}")

    tokenizer    = CLIPTokenizer.from_pretrained(args.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model_id, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(args.sd_model_id, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(args.sd_model_id, subfolder="unet")
    noise_sched  = DDPMScheduler.from_pretrained(args.sd_model_id, subfolder="scheduler")

    if offload_cpu:
        unet = unet.to(DEVICE)
        text_encoder = text_encoder.cpu()
        vae = vae.cpu()
    else:
        text_encoder = text_encoder.to(DEVICE)
        vae = vae.to(DEVICE)
        unet = unet.to(DEVICE)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    ema     = EMAModel(unet, decay=args.ema_decay, update_after_step=args.ema_warmup_steps)
    dataset = GastroVisionSDDataset(train_csv, tokenizer)
    loader  = DataLoader(dataset, batch_size=args.sd_batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)

    opt     = torch.optim.AdamW(unet.parameters(), lr=args.sd_lr, weight_decay=1e-4)
    lrsched = get_diffusers_scheduler(
        "cosine", optimizer=opt,
        num_warmup_steps=500, num_training_steps=args.domain_adapt_steps,
    )
    scaler  = GradScaler()

    resume_path = CKPT_DIR / "resume_sd_lora.pt"
    step = 0; losses = []

    if resume_path.exists():
        ck = torch.load(resume_path, map_location=DEVICE)
        unet.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        lrsched.load_state_dict(ck["scheduler"])
        step   = ck["global_step"]
        losses = ck.get("losses", [])
        if "ema" in ck:
            ema.load_state_dict(ck["ema"])
        print(f"Resumed at step {step}/{args.domain_adapt_steps}")

    print(f"\nDomain adaptation: {len(dataset)} images, {step}→{args.domain_adapt_steps} steps")
    unet.train(); opt.zero_grad(); it = iter(loader); rl = 0.0

    while step < args.domain_adapt_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)

        pv = batch["pixel_values"].to(DEVICE)
        ii = batch["input_ids"].to(DEVICE)

        vae_on_gpu = next(vae.parameters()).device == DEVICE
        if not vae_on_gpu:
            vae.to(DEVICE)
        with torch.no_grad():
            lat = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor
        if offload_cpu:
            vae.cpu(); torch.cuda.empty_cache()

        noise = torch.randn_like(lat)
        t     = torch.randint(0, noise_sched.config.num_train_timesteps,
                              (lat.shape[0],), device=DEVICE).long()
        w     = _snr_weights(noise_sched, t, DEVICE)
        nl    = noise_sched.add_noise(lat, noise, t)

        te_on_gpu = next(text_encoder.parameters()).device == DEVICE
        if not te_on_gpu:
            text_encoder.to(DEVICE)
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
            print(f"  Step {step:5d}/{args.domain_adapt_steps}  loss={avg:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")
        if step % 500 == 0:
            torch.save({
                "state_dict": unet.state_dict(), "optimizer": opt.state_dict(),
                "scheduler":  lrsched.state_dict(), "global_step": step,
                "losses":     losses, "ema": ema.state_dict(),
            }, resume_path)

    torch.save(unet.state_dict(), CKPT_DIR / "sd_gastrovision_lora.pt")
    unet.save_pretrained(CKPT_DIR / "sd_gastrovision_lora_adapter")
    ema.save_adapter(unet, CKPT_DIR / "sd_gastrovision_lora_ema_adapter")

    final = losses[-1] if losses else float("nan")
    print(f"\nDomain adaptation done — final loss: {final:.4f}")
    if final > 0.08:
        print(f"  ⚠ Loss > 0.08 — consider more steps (current: {args.domain_adapt_steps})")

    # Loss curve (sd_loss.png)
    if losses:
        fig, ax = plt.subplots(figsize=(12, 4))
        xs = [i * 100 for i in range(1, len(losses) + 1)]
        ax.plot(xs, losses, color="#4878cf", linewidth=1.5, label="Loss")
        if len(losses) > 10:
            w = max(5, len(losses) // 20)
            sm = np.convolve(losses, np.ones(w)/w, mode="valid")
            ax.plot(xs[w-1:], sm, color="#d65f5f", linewidth=2.0, alpha=0.8, label="Smoothed")
        ax.axhline(0.05, color="#6acc65", linestyle="--", alpha=0.7, label="Target 0.05")
        ax.axhline(0.08, color="#f0a500", linestyle="--", alpha=0.7, label="Acceptable 0.08")
        ax.set_xlabel("Step"); ax.set_ylabel("SNR-weighted MSE Loss")
        ax.set_title("SD LoRA Domain Adaptation Loss"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "sd_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

    del unet, vae, text_encoder, tokenizer
    torch.cuda.empty_cache(); gc.collect()


# ==============================================================================
# Synthetic image generation
# ==============================================================================

def generate_synthetic():
    """Generate synthetic images for all rare classes using the fine-tuned LoRA adapter."""
    free_gb = shutil.disk_usage(OUTPUT_DIR).free / (1024**3)
    if free_gb < args.min_free_disk_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free_gb:.1f} GB free, "
            f"minimum {args.min_free_disk_gb} GB required."
        )
    print(f"Disk check: {free_gb:.1f} GB free")

    ema_path = CKPT_DIR / "sd_gastrovision_lora_ema_adapter"
    raw_path = CKPT_DIR / "sd_gastrovision_lora_adapter"
    adapter  = ema_path if ema_path.exists() else raw_path
    if not adapter.exists():
        raise FileNotFoundError("No LoRA adapter found — run domain_adapt_sd() first.")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id, torch_dtype=torch.float16, safety_checker=None
    ).to(DEVICE)
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

    real_df = pd.read_csv(SPLITS_DIR / args.train_csv)
    l2n = (dict(zip(real_df["label"].astype(int), real_df["class_name"]))
           if "class_name" in real_df.columns else {})

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for cls in _config.RARE_CLASSES:
        cls_name = l2n.get(cls, f"class_{cls}")
        cls_dir  = SYNTH_DIR / str(cls)
        cls_dir.mkdir(parents=True, exist_ok=True)

        original_label = _config.REV_LABEL_MAP.get(cls, cls)
        prompt = DOMAIN_PREFIX + CLASS_PROMPTS.get(
            original_label, f"endoscopy photo, {cls_name}, circular vignette"
        )

        # Token length warning
        tokens  = pipe.tokenizer(prompt, return_tensors="pt", truncation=False)
        n_tok   = tokens.input_ids.shape[1]
        if n_tok > 77:
            print(f"  ⚠ Prompt for class {cls} is {n_tok} tokens (>77) — CLIP will truncate")

        existing = sorted(cls_dir.glob("synth_*.png"))
        for p in existing:
            rows.append({"image_path": str(p.relative_to(OUTPUT_DIR)),
                         "label": cls, "class_name": cls_name, "source": "sd_ema"})
        if len(existing) >= args.samples_per_class:
            print(f"  Class {cls}: already complete ({len(existing)} images)")
            continue

        start = len(existing)
        print(f"\nClass {cls} ({cls_name}): generating "
              f"{args.samples_per_class - start} images  [{prompt[:70]}...]")
        idx = start

        while idx < args.samples_per_class:
            n = min(args.gen_batch_size, args.samples_per_class - idx)
            torch.cuda.empty_cache(); gc.collect()
            gens = [
                torch.Generator(device=DEVICE).manual_seed(args.seed + cls * 100000 + idx + i)
                for i in range(n)
            ]
            with torch.no_grad():
                imgs = pipe(
                    prompt=[prompt] * n,
                    negative_prompt=[NEGATIVE_PROMPT] * n,
                    num_inference_steps=args.gen_steps,
                    guidance_scale=args.guidance_scale,
                    height=512, width=512, generator=gens,
                ).images
            for img in imgs:
                img  = _postprocess(img.resize((args.img_size, args.img_size), Image.LANCZOS))
                path = cls_dir / f"synth_{idx:05d}.png"
                img.save(path)
                rows.append({"image_path": str(path.relative_to(OUTPUT_DIR)),
                             "label": cls, "class_name": cls_name, "source": "sd_ema"})
                idx += 1
            if idx % 50 == 0 or idx >= args.samples_per_class:
                print(f"  {idx}/{args.samples_per_class}")
        print(f"  Class {cls} done")

    del pipe; torch.cuda.empty_cache(); gc.collect()

    synth_df = pd.DataFrame(rows)
    synth_df.to_csv(SYNTH_DIR / "synthetic_train.csv", index=False)
    print(f"\n{len(synth_df)} synthetic images → {SYNTH_DIR / 'synthetic_train.csv'}")
    return synth_df

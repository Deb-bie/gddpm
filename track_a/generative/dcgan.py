"""
track_a/generative/dcgan.py
=============================
Per-class, per-n unconditional DCGAN — the RQ5 comparison arm ("does the
low-n augmentation failure generalize to the GAN-era approach the field
trusted before diffusion, or is it diffusion-specific?").

Design, per track_a_prior_work_review.docx Section 9 flag 5:
  - One small DCGAN trained PER CLASS (not one shared conditional model —
    mirrors Frid-Adar et al.'s actual per-lesion-type methodology, and
    structurally matches the per-class LoRA fine-tuning approach: one
    generative-model instance per class in both arms).
  - Trained per (class, n) — the same real-image subsample used for the
    SD+LoRA arm at that grid point is the ONLY data the DCGAN sees, so the
    comparison is at matched real-data budget, not matched compute budget.
  - Standard Radford et al. DCGAN architecture, native 64x64 resolution
    (transposed-conv architectures double resolution per layer; 224 isn't
    reachable via clean doubling). Generated images are resized to
    args.img_size after generation — same convention as the SD arm's
    512→img_size resize — so both arms feed the classifier at identical
    resolution.
  - At n=1 (the grid's extreme floor), GAN training is close to degenerate
    by construction — there is no distributional diversity to model, and
    the discriminator can trivially memorize the single real image. This
    mirrors what single-image LoRA fine-tuning ALSO does (near-identical
    reproductions), so it is not treated as a bug to special-case around;
    a near-collapsed n=1 GAN is itself part of the RQ5 finding, not noise
    to be filtered out before reporting.
"""

from pathlib import Path
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# Architecture (Radford et al. 2015, standard DCGAN)
# ==============================================================================

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64, channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8), nn.ReLU(True),
            # (fm*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4), nn.ReLU(True),
            # (fm*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2), nn.ReLU(True),
            # (fm*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps), nn.ReLU(True),
            # (fm) x 32 x 32
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # channels x 64 x 64
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class Discriminator(nn.Module):
    def __init__(self, feature_maps: int = 64, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # fm x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2), nn.LeakyReLU(0.2, inplace=True),
            # fm*2 x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4), nn.LeakyReLU(0.2, inplace=True),
            # fm*4 x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8), nn.LeakyReLU(0.2, inplace=True),
            # fm*8 x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def _weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ==============================================================================
# Per-class dataset
# ==============================================================================

class _ClassImageDataset(Dataset):
    """Loads one class's real-image subsample at native DCGAN resolution."""

    def __init__(self, image_paths, data_dir, native_res: int):
        self.paths = list(image_paths)
        self.data_dir = Path(data_dir)
        self.transform = T.Compose([
            T.Resize((native_res, native_res)),
            T.RandomHorizontalFlip(),  # mitigates trivial memorization at low n
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.data_dir / self.paths[idx]).convert("RGB")
        return self.transform(img)


# ==============================================================================
# Training
# ==============================================================================

def train_dcgan_for_class(dataset_name: str, class_id, n: int, image_paths, data_dir,
                           ckpt_dir, args, device) -> Path:
    """
    Trains one Generator/Discriminator pair on exactly the real-image
    subsample given (the same n-subsample used by the SD+LoRA arm at this
    grid point). Saves the generator's state_dict to
    ckpt_dir/dcgan_{dataset_name}_class{class_id}_n{n}.pt and returns that path.
    Skips training if the checkpoint already exists (resumability, same
    convention as the rest of the pipeline).
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / f"dcgan_{dataset_name}_class{class_id}_n{n}.pt"
    if ckpt_path.exists():
        print(f"  [DCGAN {dataset_name} class={class_id} n={n}] checkpoint exists — skipping")
        return ckpt_path

    ds = _ClassImageDataset(image_paths, data_dir, args.dcgan_native_res)
    # batch_size can't exceed n; drop_last only if we have more than one batch worth
    batch_size = min(args.dcgan_batch_size, len(ds))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=(len(ds) > batch_size))

    G = Generator(args.dcgan_latent_dim, channels=3).to(device)
    D = Discriminator(channels=3).to(device)
    G.apply(_weights_init); D.apply(_weights_init)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.dcgan_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.dcgan_lr, betas=(0.5, 0.999))

    print(f"\n[DCGAN {dataset_name} class={class_id} n={n}] "
          f"{len(ds)} real images, batch_size={batch_size}, {args.dcgan_epochs} epochs")

    fixed_noise = torch.randn(min(8, batch_size), args.dcgan_latent_dim, device=device)

    for epoch in range(args.dcgan_epochs):
        d_loss_sum = g_loss_sum = 0.0
        n_batches = 0
        for real in loader:
            real = real.to(device)
            bsz  = real.size(0)
            real_labels = torch.full((bsz,), 0.9, device=device)  # label smoothing
            fake_labels = torch.zeros(bsz, device=device)

            # --- Discriminator ---
            opt_d.zero_grad()
            d_real = D(real)
            loss_d_real = F.binary_cross_entropy_with_logits(d_real, real_labels)

            z = torch.randn(bsz, args.dcgan_latent_dim, device=device)
            fake = G(z)
            d_fake = D(fake.detach())
            loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # --- Generator ---
            opt_g.zero_grad()
            d_fake_for_g = D(fake)
            loss_g = F.binary_cross_entropy_with_logits(
                d_fake_for_g, torch.full((bsz,), 0.9, device=device)
            )
            loss_g.backward()
            opt_g.step()

            d_loss_sum += loss_d.item(); g_loss_sum += loss_g.item(); n_batches += 1

        if (epoch + 1) % max(1, args.dcgan_epochs // 20) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{args.dcgan_epochs}  "
                  f"D_loss={d_loss_sum/n_batches:.4f}  G_loss={g_loss_sum/n_batches:.4f}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), ckpt_path)
    print(f"  Saved generator → {ckpt_path}")
    return ckpt_path


# ==============================================================================
# Generation
# ==============================================================================

def generate_dcgan_images(dataset_name: str, class_id, n: int, ckpt_dir, out_dir,
                           num_images: int, class_name: str, args, device) -> pd.DataFrame:
    ckpt_dir = Path(ckpt_dir)
    out_dir  = Path(out_dir)
    ckpt_path = ckpt_dir / f"dcgan_{dataset_name}_class{class_id}_n{n}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No DCGAN checkpoint for {dataset_name} class={class_id} n={n} — "
            "run train_dcgan_for_class() first."
        )

    G = Generator(args.dcgan_latent_dim, channels=3).to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    G.eval()

    cls_dir = out_dir / str(class_id) / f"n{n}"
    cls_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(cls_dir.glob("dcgan_*.png"))
    rows = [{"image_path": str(p.relative_to(out_dir)), "label": class_id,
             "class_name": class_name, "source": "dcgan", "n": n} for p in existing]

    if len(existing) >= num_images:
        print(f"  [DCGAN {dataset_name} class={class_id} n={n}] already complete ({len(existing)})")
        return pd.DataFrame(rows)

    idx = len(existing)
    batch = 16
    with torch.no_grad():
        while idx < num_images:
            m = min(batch, num_images - idx)
            gen = torch.Generator(device=device).manual_seed(args.seed + int(class_id) * 100000 + n + idx)
            z = torch.randn(m, args.dcgan_latent_dim, device=device, generator=gen)
            fake = G(z)  # in [-1, 1], NCHW
            fake = ((fake.clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            for i in range(m):
                img = Image.fromarray(fake[i]).resize((args.img_size, args.img_size), Image.LANCZOS)
                path = cls_dir / f"dcgan_{idx:05d}.png"
                img.save(path)
                rows.append({"image_path": str(path.relative_to(out_dir)), "label": class_id,
                             "class_name": class_name, "source": "dcgan", "n": n})
                idx += 1

    print(f"  [DCGAN {dataset_name} class={class_id} n={n}] generated {num_images} images")
    return pd.DataFrame(rows)

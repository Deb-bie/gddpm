"""
train.py
========
Training engine: two-phase frozen/fine-tune, EMA, Optuna tuning,
heavy augmentation variant. WeightedRandomSampler is now wired in.
"""

import gc
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import args, DEVICE, CKPT_DIR, HPARAMS, NUM_CLASSES
from dataset import GastroVisionDataset, get_weighted_sampler
from losses import FocalLoss
from models import get_model

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ==============================================================================
# EMA
# ==============================================================================

class EMAModel:
    def __init__(self, model, decay=0.9999, update_after_step=100):
        self.decay = decay
        self.update_after_step = update_after_step
        self.step_count = 0
        self.shadow = {n: p.detach().cpu().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def step(self, model):
        self.step_count += 1
        decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        if self.step_count < self.update_after_step:
            for n, p in model.named_parameters():
                if n in self.shadow and p.requires_grad:
                    self.shadow[n] = p.detach().cpu().clone()
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow and p.requires_grad:
                    s = self.shadow[n].to(p.device)
                    s.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
                    self.shadow[n] = s.cpu()

    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow and p.requires_grad:
                p.data.copy_(self.shadow[n].to(p.device))

    def restore(self, model, orig):
        for n, p in model.named_parameters():
            if n in orig and p.requires_grad:
                p.data.copy_(orig[n].to(p.device))

    def state_dict(self):
        return {"shadow": self.shadow, "step_count": self.step_count, "decay": self.decay}

    def load_state_dict(self, s):
        self.shadow     = s["shadow"]
        self.step_count = s["step_count"]
        self.decay      = s.get("decay", self.decay)

    def save_adapter(self, model, path):
        from pathlib import Path
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        orig = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        try:
            self.copy_to(model)
            model.save_pretrained(path)
            print(f"  EMA adapter saved → {path}")
        finally:
            self.restore(model, orig)


def _snr_weights(scheduler, t, device, gamma=5.0):
    ac  = scheduler.alphas_cumprod.to(device)
    snr = (ac[t] ** 0.5 / ((1 - ac[t]) ** 0.5 + 1e-8)) ** 2
    return (torch.clamp(snr, max=gamma) / (snr + 1e-8)).detach()


# ==============================================================================
# Freeze / unfreeze helpers
# ==============================================================================

def _freeze(model, model_name):
    if "hybrid" in model_name or model_name == "dinov2":
        model.freeze_backbones()
    else:
        for p in model.parameters():
            p.requires_grad = False
        head = getattr(model, "head", None) or getattr(model, "classifier", None)
        if head is None:
            raise AttributeError(f"No head on {model_name}")
        for p in head.parameters():
            p.requires_grad = True


def _unfreeze(model, model_name):
    if "hybrid" in model_name or model_name == "dinov2":
        model.unfreeze_all()
    else:
        for p in model.parameters():
            p.requires_grad = True


# ==============================================================================
# Evaluation helper
# ==============================================================================

def _eval_acc(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            with autocast():
                preds = model(xb.to(DEVICE)).argmax(1)
            yp.append(preds.cpu().numpy())
            yt.append(yb.numpy())
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)
    return float((yt == yp).mean()), yt, yp


# ==============================================================================
# Main training function (with WeightedRandomSampler)
# ==============================================================================

def train_classifier(model_name: str, train_csv, val_csv, augmented: bool = False):
    cfg    = HPARAMS[model_name]
    crit   = FocalLoss(gamma=cfg["gamma"])
    scaler = GradScaler()
    model  = get_model(model_name)
    ckpt   = CKPT_DIR / f"sota_{model_name}{'_aug' if augmented else ''}.pt"

    # --- Datasets ---
    train_ds = GastroVisionDataset(train_csv, "train", heavy=False)
    val_ds   = GastroVisionDataset(val_csv,   "val")

    # WeightedRandomSampler replaces shuffle=True to handle class imbalance
    sampler = get_weighted_sampler(train_csv)
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                    num_workers=4, pin_memory=True)
    vl = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=4, pin_memory=True)

    history = {"train_loss": [], "val_acc": [], "phase": []}

    # --- Phase 1: frozen backbone ---
    _freeze(model, model_name)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"] * cfg.get("freeze_lr_mult", 10.0),
    )
    print(f"\n{'='*60}\n[{model_name}] Phase 1 — frozen ({cfg['freeze_epochs']} epochs)\n{'='*60}")

    for ep in range(cfg["freeze_epochs"]):
        model.train()
        rl = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast():
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            scaler.step(opt)
            scaler.update()
            rl += loss.item()
        acc, _, _ = _eval_acc(model, vl)
        history["train_loss"].append(rl / len(tl))
        history["val_acc"].append(acc)
        history["phase"].append("freeze")
        print(f"  Ep {ep+1:2d}/{cfg['freeze_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")

    # --- Phase 2: full fine-tune ---
    _unfreeze(model, model_name)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4)
    )
    sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["fine_tune_epochs"])
    best_acc = 0.0
    print(f"\n{'='*60}\n[{model_name}] Phase 2 — fine-tune ({cfg['fine_tune_epochs']} epochs)\n{'='*60}")

    for ep in range(cfg["fine_tune_epochs"]):
        model.train()
        rl = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast():
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            rl += loss.item()
        sch.step()
        acc, _, _ = _eval_acc(model, vl)
        history["train_loss"].append(rl / len(tl))
        history["val_acc"].append(acc)
        history["phase"].append("finetune")
        print(f"  Ep {ep+1:2d}/{cfg['fine_tune_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt)
            with open(ckpt.with_suffix(".meta.json"), "w") as f:
                json.dump({"num_classes": NUM_CLASSES}, f)
            print(f"  ✅ Saved (val_acc={best_acc:.4f})")

    print(f"\n  ★ {model_name} best val_acc: {best_acc:.4f}")
    return history


# ==============================================================================
# Heavy augmentation variant
# ==============================================================================

def train_classifier_heavy_aug(model_name: str, train_csv, val_csv):
    cfg    = HPARAMS[model_name]
    crit   = FocalLoss(gamma=cfg["gamma"])
    scaler = GradScaler()
    model  = get_model(model_name)
    ckpt   = CKPT_DIR / f"sota_{model_name}_heavy.pt"

    train_ds = GastroVisionDataset(train_csv, "train", heavy=True)
    val_ds   = GastroVisionDataset(val_csv,   "val")

    sampler = get_weighted_sampler(train_csv)
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                    num_workers=4, pin_memory=True)
    vl = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=4, pin_memory=True)

    _freeze(model, model_name)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"] * cfg.get("freeze_lr_mult", 10.0),
    )
    print(f"\n{'='*60}\n[{model_name}] Heavy Aug — Phase 1 ({cfg['freeze_epochs']} epochs)\n{'='*60}")
    for ep in range(cfg["freeze_epochs"]):
        model.train()
        rl = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast():
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            scaler.step(opt)
            scaler.update()
            rl += loss.item()
        acc = _eval_acc(model, vl)[0]
        print(f"  Ep {ep+1:2d}/{cfg['freeze_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")

    _unfreeze(model, model_name)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4)
    )
    sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["fine_tune_epochs"])
    best_acc = 0.0
    print(f"\n{'='*60}\n[{model_name}] Heavy Aug — Phase 2 ({cfg['fine_tune_epochs']} epochs)\n{'='*60}")
    for ep in range(cfg["fine_tune_epochs"]):
        model.train()
        rl = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast():
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            rl += loss.item()
        sch.step()
        acc = _eval_acc(model, vl)[0]
        print(f"  Ep {ep+1:2d}/{cfg['fine_tune_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt)
            with open(ckpt.with_suffix(".meta.json"), "w") as f:
                json.dump({"num_classes": NUM_CLASSES}, f)
            print(f"  ✅ Saved (val_acc={best_acc:.4f})")

    print(f"\n  ★ {model_name} heavy aug best val_acc: {best_acc:.4f}")
    return best_acc


# ==============================================================================
# Optuna tuning
# ==============================================================================

def tune_classifier(model_name: str, train_csv, val_csv,
                    n_trials: int = 15, tune_epochs: int = 8):
    if not OPTUNA_AVAILABLE:
        print(f"  Optuna not available — skipping tuning for {model_name}")
        return

    print(f"\nTuning {model_name} ({n_trials} trials × {tune_epochs} epochs)...")

    def objective(trial):
        lr             = trial.suggest_float("lr",             1e-5, 5e-4, log=True)
        gamma          = trial.suggest_float("gamma",          0.5,  3.0)
        freeze_lr_mult = trial.suggest_float("freeze_lr_mult", 2.0,  15.0)
        weight_decay   = trial.suggest_float("weight_decay",   1e-5, 1e-2, log=True)
        batch_size     = trial.suggest_categorical("batch_size", [8, 16])

        train_ds = GastroVisionDataset(train_csv, "train")
        val_ds   = GastroVisionDataset(val_csv,   "val")
        sampler  = get_weighted_sampler(train_csv)
        tl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                        num_workers=2, pin_memory=True)
        vl = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

        model  = get_model(model_name)
        crit   = FocalLoss(gamma=gamma)
        scaler = GradScaler()

        _freeze(model, model_name)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr * freeze_lr_mult,
        )
        for _ in range(min(3, tune_epochs // 2)):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                with autocast():
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                scaler.step(opt)
                scaler.update()

        _unfreeze(model, model_name)
        opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tune_epochs)
        best_acc = 0.0

        for ep in range(tune_epochs):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                with autocast():
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            sch.step()
            acc      = _eval_acc(model, vl)[0]
            best_acc = max(best_acc, acc)
            trial.report(acc, ep)
            if trial.should_prune():
                del model
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        del model
        torch.cuda.empty_cache()
        return best_acc

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial.params
    print(f"  Best val_acc: {study.best_value:.4f}")
    HPARAMS[model_name].update({
        "lr":             best["lr"],
        "gamma":          best["gamma"],
        "freeze_lr_mult": best["freeze_lr_mult"],
        "weight_decay":   best["weight_decay"],
        "batch_size":     best["batch_size"],
    })
    return study

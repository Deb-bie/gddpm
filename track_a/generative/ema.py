"""
track_a/generative/ema.py
==========================
EMAModel and SNR-loss-weighting, copied verbatim (not imported) from
gastrovision/train.py.

Why copied rather than imported: gastrovision/train.py does
`from config import args, DEVICE, CKPT_DIR, HPARAMS`, and gastrovision's
config.py calls parse_args() at import time against its OWN argparse
parser. Importing anything from gastrovision.train would trigger that
import chain and crash on Track A's own CLI flags (--datasets,
--run_dcgan_comparison, etc. are not recognized by gastrovision's parser).
These two pieces have no dataset-specific logic in them at all — they're
pure PyTorch utilities — so duplication here is a clean break of that
import chain rather than a maintenance burden (nothing dataset-specific
could drift between the two copies).
"""

import torch


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


def snr_weights(scheduler, t, device, gamma=5.0):
    ac  = scheduler.alphas_cumprod.to(device)
    snr = (ac[t] ** 0.5 / ((1 - ac[t]) ** 0.5 + 1e-8)) ** 2
    return (torch.clamp(snr, max=gamma) / (snr + 1e-8)).detach()

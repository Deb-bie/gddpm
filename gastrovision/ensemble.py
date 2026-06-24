"""
ensemble.py
===========
Confidence-weighted ensemble: loads multiple checkpoints and combines
softmax probabilities weighted by per-sample peak confidence.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from config import DEVICE, CKPT_DIR
from models import get_model


class ConfidenceEnsemble:
    """
    Combines N models via confidence-weighted soft voting.
    Each sample's vote weight = softmax confidence of that model on that sample.
    """

    def __init__(self, model_names: list, suffix: str = ""):
        self.models = {}
        self.suffix = suffix
        for name in model_names:
            ckpt = CKPT_DIR / f"sota_{name}{suffix}.pt"
            if not ckpt.exists():
                print(f"  Ensemble: skipping {name} — {ckpt.name} not found")
                continue
            try:
                m = get_model(name)
                m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                m.eval()
                self.models[name] = m
                print(f"  Ensemble: loaded {name}{suffix}")
            except Exception as e:
                print(f"  Ensemble: failed to load {name}: {e}")

        if not self.models:
            raise RuntimeError(
                f"Ensemble: no models loaded for suffix='{suffix}'. Train models first."
            )
        print(f"  Ensemble ready: {len(self.models)} models [{', '.join(self.models.keys())}]")

    def predict(self, x: torch.Tensor):
        """Returns (predicted_class, ensemble_probs)."""
        x          = x.to(DEVICE)
        probs_list = []
        with torch.no_grad():
            for m in self.models.values():
                with autocast():
                    probs_list.append(F.softmax(m(x), dim=1))

        stacked     = torch.stack(probs_list, dim=0)              # (M, B, C)
        confidences = stacked.max(dim=2).values.permute(1, 0)     # (B, M)
        weights     = confidences / confidences.sum(dim=1, keepdim=True)
        ens_probs   = (stacked * weights.permute(1, 0).unsqueeze(-1)).sum(dim=0)  # (B, C)
        return ens_probs.argmax(dim=1), ens_probs

    def subset_predict(self, x: torch.Tensor, names: list):
        """Predict using a named subset of models (for ablation)."""
        x          = x.to(DEVICE)
        probs_list = []
        with torch.no_grad():
            for name in names:
                if name not in self.models:
                    continue
                with autocast():
                    probs_list.append(F.softmax(self.models[name](x), dim=1))
        if not probs_list:
            raise ValueError(f"No valid models in subset {names}")
        stacked     = torch.stack(probs_list, dim=0)
        confidences = stacked.max(dim=2).values.permute(1, 0)
        weights     = confidences / confidences.sum(dim=1, keepdim=True)
        ens_probs   = (stacked * weights.permute(1, 0).unsqueeze(-1)).sum(dim=0)
        return ens_probs.argmax(dim=1), ens_probs


def eval_ensemble(ensemble: ConfidenceEnsemble, loader, subset=None):
    """
    Run ensemble inference over a DataLoader.
    Returns (accuracy, y_true, y_pred, probabilities).
    """
    yt_list, yp_list, pr_list = [], [], []
    for xb, yb in loader:
        if subset:
            preds, probs = ensemble.subset_predict(xb, subset)
        else:
            preds, probs = ensemble.predict(xb)
        yt_list.append(yb.numpy())
        yp_list.append(preds.cpu().numpy())
        pr_list.append(probs.cpu().numpy())

    yt = np.concatenate(yt_list)
    yp = np.concatenate(yp_list)
    pr = np.concatenate(pr_list)
    return float((yt == yp).mean()), yt, yp, pr

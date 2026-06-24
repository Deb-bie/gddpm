"""
losses.py
=========
Loss functions used in training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    gamma=2.0 is the standard setting; higher values focus more on hard examples.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropy(nn.Module):
    """Weighted cross-entropy — used in ablation study vs FocalLoss."""

    def __init__(self, class_counts, device):
        super().__init__()
        counts  = torch.tensor(class_counts, dtype=torch.float32)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("weights", weights.to(device))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights)

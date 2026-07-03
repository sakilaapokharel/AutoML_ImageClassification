"""Loss search space: Cross Entropy, Label Smoothing, Focal Loss,
Class-Balanced Loss, Weighted Cross Entropy (the diagram's rightmost box)."""
import torch
import torch.nn as nn
import torch.nn.functional as F



class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss based on Effective Number of Samples
    (Cui et al., CVPR 2019)."""

    def __init__(self, class_counts, beta=0.999):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float)
        eff_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / eff_num.clamp(min=1e-8)
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("weights", weights)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weights.to(logits.device))


def get_loss(name, class_counts=None, device="cpu"):
    """name in {'cross_entropy', 'label_smoothing', 'focal', 'class_balanced',
    'weighted_cross_entropy'}. class_counts (per-class sample counts in the
    training split) is required for the two class-count-aware losses."""
    name = name.lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()

    if name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)

    if name == "class_balanced":
        if class_counts is None:
            raise ValueError("class_counts required for class_balanced loss")
        return ClassBalancedLoss(class_counts)

    raise ValueError(f"Unknown loss '{name}'")

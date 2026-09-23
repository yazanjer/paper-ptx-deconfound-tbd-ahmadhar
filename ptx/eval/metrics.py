"""Point metrics. Everything takes numpy arrays of labels y (0/1) and scores s (probabilities)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def threshold_at_specificity(y, s, spec=0.95):
    """Smallest threshold whose specificity on (y, s) is >= spec (chosen on validation only)."""
    neg = np.sort(np.asarray(s)[np.asarray(y) == 0])
    if len(neg) == 0:
        return 0.5
    k = int(np.ceil(spec * len(neg))) - 1
    k = min(max(k, 0), len(neg) - 1)
    return float(np.nextafter(neg[k], np.inf))  # strictly above the k-th negative


def sens_spec_at(y, s, thr):
    y, s = np.asarray(y), np.asarray(s)
    pred = s >= thr
    sens = pred[y == 1].mean() if (y == 1).any() else np.nan
    spec = (~pred[y == 0]).mean() if (y == 0).any() else np.nan
    return float(sens), float(spec)


def sens_at_spec(y, s, spec=0.95):
    """Sensitivity at a given specificity computed on the same data (used only for validation / model selection)."""
    t = threshold_at_specificity(y, s, spec)
    return sens_spec_at(y, s, t)[0]


def ece(y, p, n_bins=15):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fit_temperature(y, logits):
    """Temperature scaling on validation logits (A7)."""
    import torch
    t = torch.nn.Parameter(torch.ones(1))
    x = torch.tensor(np.asarray(logits), dtype=torch.float32)
    yy = torch.tensor(np.asarray(y), dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x / t.clamp(min=0.05), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min=0.05))


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-image Dice; both empty -> 1.0 (Kaggle convention). Report positives-only Dice separately."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    return float(2 * (p & g).sum() / (p.sum() + g.sum()))

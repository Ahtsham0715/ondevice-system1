# Adapted from laya/common.py (Convai Innovations, Apache-2.0) — see NOTICE.
"""Confidence, calibration error, and per-bucket temperature scaling."""
import math

import numpy as np
import torch

from .decision_model import QTYPE_NAMES

TEMP_MIN = 0.5
TEMP_MAX = 5.0


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across confidence bins."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if not a number.

    A fitted temperature below 1 *sharpens* logits instead of softening them —
    Laya shipped a 0.1006 bucket once that turned a 0.24 top probability into a
    published 0.99 (see docs/LAYA_ANALYSIS.md). No honest calibration needs to
    sharpen this hard, so refuse to apply one that does.
    """
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(hi, max(lo, t))


def fit_one_temperature(logits_targets, max_iter: int = 100) -> float:
    """Fit a single temperature by LBFGS-minimizing NLL. `logits_targets`: list of (z, target) pairs."""
    if len(logits_targets) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in logits_targets)
    Z = torch.full((len(logits_targets), kmax), -1e4)
    T = torch.zeros((len(logits_targets), kmax))
    for i, (z, t) in enumerate(logits_targets):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())

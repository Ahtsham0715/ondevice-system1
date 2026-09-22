# Adapted from laya/common.py (Convai Innovations, Apache-2.0) — see NOTICE.
# proper_reward is architecture-agnostic (operates on logits over option
# markers), so it is reused unchanged for RLCD-style training of any model
# size. See docs/LAYA_ANALYSIS.md "Training: RLCD" for the full mechanism.
"""RLCD loss components: strictly proper scoring rule reward + logit-space policy gradient."""
import torch

from .decision_model import QTYPES


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Strictly proper scoring rule reward: log score + spherical score + ranked probability score.

    q: [..., N, K] reported distributions. target: [N, K] (one-hot or soft targets).
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def rlcd_loss(logits, target, qtype, mask, group_size: int = 4, sigma: float = 0.2, ce_weight: float = 1.0):
    """One RLCD training step's loss: zero-order policy gradient + soft cross-entropy.

    logits: [B, K] raw scorer output (pre-softmax). target: [B, K] soft distribution.
    Mirrors laya's train_ddp.py exactly (see docs/LAYA_ANALYSIS.md); exposed as a
    reusable function here instead of being inlined in a training script.
    """
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)

    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return loss_rl + ce_weight * loss_ce, {"loss_rl": loss_rl.item(), "loss_ce": loss_ce.item(), "reward": r.mean().item()}

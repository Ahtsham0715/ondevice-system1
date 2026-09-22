# Adapted from laya/common.py (Convai Innovations, Apache-2.0) — see NOTICE.
# Changes from upstream: encoder is any HF AutoModel (not just ModernBERT/mmBERT);
# head_layers/dropout are exposed per model tier via config instead of being
# hardcoded; the per-option head token budget in src/model/sequence.py replaces
# upstream's single shared head_max_len (see docs/LAYA_ANALYSIS.md, "Design
# implications", point 3 — the one deliberate architectural deviation).
"""Typed decision model: bidirectional encoder + shallow scoring head.

One forward pass answers N typed questions (choice / score / noul) over a state.
See docs/LAYA_ANALYSIS.md for the full mechanism this is adapted from.
"""
from typing import Dict

import torch
import torch.nn as nn

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


class DecisionModel(nn.Module):
    """Encoder backbone + typed decision head.

    `encoder` is any HF `AutoModel` bidirectional encoder (ModernBERT, mmBERT,
    BERT, ...). Swapping model size is swapping `encoder` — the head is
    encoder-width-agnostic by construction (`d = encoder.config.hidden_size`).
    """

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits

    def num_parameters(self) -> Dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc, "head": total - enc, "total": total}


def build_model(cfg: Dict, pretrained: bool = True) -> DecisionModel:
    """Build a DecisionModel from a config dict (see configs/*/model.json).

    `pretrained=False` builds the encoder from config only (random init) —
    used for from-scratch or randomly-initialized-encoder ablations.
    """
    from transformers import AutoConfig, AutoModel

    if pretrained:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    else:
        ecfg = AutoConfig.from_pretrained(cfg["encoder"])
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    return DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1,
                          cfg.get("dropout", 0.1))

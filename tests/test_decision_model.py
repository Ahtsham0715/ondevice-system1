"""Smoke test: build a tiny random-init model, run a forward pass, check shapes.

Uses a real tiny HF config (no download) so this runs offline and fast — not the
full pretrained tiers, just an architecture sanity check.
"""
import torch
from transformers import BertConfig, BertModel

from src.model.decision_model import DecisionModel
from src.model.collate import collate_items


def _tiny_encoder():
    cfg = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64)
    return BertModel(cfg)


def test_forward_shapes():
    model = DecisionModel(_tiny_encoder(), head_layers=1, n_act=2)
    items = [
        {"ids": [1, 2, 3, 4, 5], "markers": [1, 3], "qtype": 0},
        {"ids": [1, 2, 3], "markers": [1], "qtype": 2},
    ]
    batch = collate_items(items, pad_id=0)
    logits, act = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
    assert logits.shape == (2, 2)  # kmax=2 options
    assert act.shape == (2, 2)  # n_act=2
    # second item has only 1 valid marker; its second logit slot must be masked out
    assert torch.isinf(logits[1, 1]) or logits[1, 1] < -1e3


def test_num_parameters():
    model = DecisionModel(_tiny_encoder(), head_layers=1, n_act=2)
    counts = model.num_parameters()
    assert counts["total"] == counts["encoder"] + counts["head"]
    assert counts["encoder"] > 0 and counts["head"] > 0


if __name__ == "__main__":
    test_forward_shapes()
    test_num_parameters()
    print("ok")

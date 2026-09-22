"""Train (or fine-tune) a DecisionModel with the RLCD loss, single-process.

This is the "does a smaller/different encoder even work with this head and loss" sanity check
(ROADMAP.md, Next experiments #4) before any distillation is attempted — a from-scratch decision
head trained directly on typed-decisions/train, no teacher involved. Mirrors upstream's
train_ddp.py loss and hyperparameter shape (docs/LAYA_ANALYSIS.md) but single-GPU/CPU/MPS, since
this project's dev hardware has no multi-GPU cluster.

Usage:
    python -m src.training.train_rlcd --config configs/base/model.json --out runs/base_sanity
"""
import argparse
import json
import os
import random

import torch

import src.compat  # noqa: F401 — must import before transformers, see src/compat.py
from transformers import AutoTokenizer

from src.data.typed_decisions import load, to_training_items
from src.model.calibration import fit_one_temperature
from src.model.collate import collate_items
from src.model.decision_model import build_model
from src.model.losses import rlcd_loss


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr_encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--sigma_start", type=float, default=0.4)
    ap.add_argument("--sigma_end", type=float, default=0.1)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--workflow", default="all")
    ap.add_argument("--limit", type=int, default=None, help="cap training examples for a quick run")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = _device()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    model = build_model(cfg, pretrained=True).to(device)
    print("params:", model.num_parameters())

    examples = load(split="train", workflow=args.workflow)
    if args.limit:
        examples = examples[: args.limit]
    items = to_training_items(
        tok, examples, max_len=cfg["max_len"],
        min_instruction_tokens=cfg.get("min_instruction_tokens", 24),
        min_opt_tokens=cfg.get("min_opt_tokens", 8),
        max_opt_tokens=cfg.get("max_opt_tokens", 48),
    )
    print(f"{len(examples)} examples -> {len(items)} question items")

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr_encoder}, {"params": head_params, "lr": args.lr_head}],
        weight_decay=0.01,
    )
    total_steps = max(1, (len(items) // args.batch_size)) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    model.train()
    for epoch in range(args.epochs):
        random.seed(args.seed + epoch)
        random.shuffle(items)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * (epoch / max(1, args.epochs - 1))
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, len(items), args.batch_size):
            chunk = items[i : i + args.batch_size]
            batch = collate_items(chunk, tok.pad_token_id)
            optimizer.zero_grad(set_to_none=True)

            logits, act = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device),
            )
            loss, stats = rlcd_loss(
                logits, batch["target"].to(device), batch["qtype"].to(device), batch["marker_mask"].to(device),
                group_size=args.group_size, sigma=sigma,
            )
            loss = loss + 0.0 * act.sum()  # keep act_head in the graph even though it's not supervised here yet
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            if n_batches % 20 == 0:
                print(f"epoch {epoch+1}/{args.epochs} step {n_batches} loss {loss.item():.4f} reward {stats['reward']:.3f}")

        print(f"=== epoch {epoch+1}/{args.epochs} avg loss {epoch_loss/max(1,n_batches):.4f} ===")

    os.makedirs(args.out, exist_ok=True)
    from safetensors.torch import save_file
    model.eval()
    save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(args.out, "model.safetensors"))
    tok.save_pretrained(os.path.join(args.out, "tokenizer"))
    with open(os.path.join(args.out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("saved to", args.out)


if __name__ == "__main__":
    main()

"""Evaluate the Laya teacher checkpoint zero-shot on LocalLLaMA/typed-decisions test split.

This is the teacher baseline row of the comparison table (BENCHMARKING.md) — every student
result gets compared against this, not against upstream's cited numbers, because upstream's
2,000-decision run used their own fine-tuned checkpoint (laya-typed-decisions), not the zero-shot
base model. Run before touching a student architecture (ROADMAP.md, Next experiments #3).

Usage:
    python scripts/eval_teacher_baseline.py --model convaiinnovations/laya --workflow all
"""
import argparse
import json

import src.compat  # noqa: F401 — must import before transformers/laya, see src/compat.py
import laya

from src.data.typed_decisions import load
from src.evaluation.metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--device", default="cpu", help="torch 2.2.2 in this environment can't autocast on mps; see src/compat.py")
    ap.add_argument("--workflow", default="all")
    ap.add_argument("--limit", type=int, default=None, help="cap examples for a quick smoke run")
    ap.add_argument("--out", default=None, help="write raw predictions + metrics JSON here")
    args = ap.parse_args()

    agent = laya.load(args.model, subfolder=args.subfolder, device=args.device)
    examples = load(split="test", workflow=args.workflow)
    if args.limit:
        examples = examples[: args.limit]

    predictions, gold = [], []
    for ex in examples:
        questions = {qid: {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}
                     for qid, q in ex["questions"].items()}
        result = agent.predict(ex["state"], questions)
        for qid, q in ex["questions"].items():
            ans = result["answers"][qid]
            predictions.append(ans)
            gold.append({"type": q["type"], "answer": q["answer"]})

    metrics = evaluate(predictions, gold)
    print(json.dumps(metrics, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "subfolder": args.subfolder, "workflow": args.workflow,
                       "n_examples": len(examples), "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    main()

"""Loader for LocalLLaMA/typed-decisions (Apache-2.0, verified — see docs/LAYA_ANALYSIS.md).

This is Laya's own published fine-tuning dataset: 4 workflow themes (agent_trace_observability,
customer_service, invoice_processing, security_incidents) plus an "all" split combining them,
each with train/test. Real schema (fetched from the HF datasets-server API, not assumed):

    id, workflow, split, state (JSON str), questions (JSON str), gold (JSON str),
    factors (JSON str), label_agreement (JSON str), n_questions (int)

`gold[qid]` carries a full soft probability distribution (`probabilities`), not just a top
label — annotator disagreement is preserved, which is exactly the `target` tensor RLCD training
consumes (see src/model/losses.py, docs/LAYA_ANALYSIS.md "Training: RLCD").
"""
import json
from typing import Dict, List, Optional

from src.model.decision_model import QTYPES
from src.model.sequence import build_sequence, render_options


def load(split: str = "train", workflow: str = "all") -> List[Dict]:
    """Load examples into this project's dataset schema (see DATASET_CONTRIBUTION.md).

    Requires the `datasets` package (see requirements.txt).
    """
    from datasets import load_dataset

    ds = load_dataset("LocalLLaMA/typed-decisions", workflow, split=split)
    examples = []
    for row in ds:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        q_out = {}
        for qid, q in questions.items():
            g = gold[qid]
            q_out[qid] = {
                "type": q["type"],
                "instructions": q["instructions"],
                "criteria": q.get("criteria"),  # optional for noul, matching laya/agent.py._to_internal
                "answer": g["label"],
                "target": g["probabilities"],
            }
        examples.append({"id": row["id"], "workflow": row["workflow"], "state": state, "questions": q_out})
    return examples


def _target_vector(q_internal: Dict, target_probs: Dict[str, float]) -> List[float]:
    """Order target probabilities to match render_options' option order."""
    if q_internal["t"] == "choice":
        keys = list(q_internal["crit"].keys())
    elif q_internal["t"] == "score":
        keys = [str(i) for i in range(len(q_internal["crit"]))]
    else:
        keys = ["false", "true"]
    return [float(target_probs.get(k, 0.0)) for k in keys]


def to_training_items(tok, examples: List[Dict], max_len: int = 512, **seq_kwargs) -> List[Dict]:
    """Tokenize examples into flat per-question items ready for collate_items.

    Mirrors the preprocessing train_ddp.py does before writing train_items.pt.
    """
    items = []
    for ex in examples:
        for qid, q in ex["questions"].items():
            crit = q["criteria"]
            if q["type"] == "choice" and isinstance(crit, list):
                crit = {c: None for c in crit}
            q_internal = {"t": q["type"], "ins": q["instructions"], "crit": crit}
            seq, markers = build_sequence(tok, ex["state"], q_internal, max_len=max_len, **seq_kwargs)
            if len(markers) != len(render_options(q_internal)):
                continue  # option(s) truncated out by the token budget — skip rather than train on a corrupt target
            target = _target_vector(q_internal, q["target"])
            items.append({
                "ids": seq,
                "markers": markers,
                "target": target,
                "qtype": QTYPES[q["type"]],
                "example_id": ex["id"],
                "qid": qid,
            })
    return items

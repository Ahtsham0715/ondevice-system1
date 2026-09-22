"""Typed-decision evaluation metrics: Choice accuracy, Score MAE, Noul F1/Brier, ECE.

Consumes model outputs in the same shape `Agent.predict`/`DecisionModel` produce
(see src/model/), so it scores any checkpoint — teacher or student — the same way.
"""
from typing import Dict, List

import numpy as np

from src.model.calibration import ece_score


def evaluate(predictions: List[Dict], gold: List[Dict]) -> Dict[str, float]:
    """Score a list of predictions against gold answers.

    predictions[i] / gold[i]: {"type": "choice"|"score"|"noul", "answer": ..., "probabilities": {...}, "confidence": float}
    For "choice": answer is the label string. For "score": answer is the integer level.
    For "noul": answer is bool; predictions[i]["noul"] is P(true).
    """
    by_type = {"choice": [], "score": [], "noul": []}
    for p, g in zip(predictions, gold):
        by_type[g["type"]].append((p, g))

    out: Dict[str, float] = {}

    if by_type["choice"]:
        correct = np.array([1.0 if p["choice"] == g["answer"] else 0.0 for p, g in by_type["choice"]])
        conf = np.array([p["confidence"] for p, g in by_type["choice"]])
        out["choice_accuracy"] = float(correct.mean())
        out["choice_ece"] = ece_score(conf, correct)

    if by_type["score"]:
        pred_s = np.array([p["score"] for p, g in by_type["score"]])
        gold_s = np.array([float(g["answer"]) for p, g in by_type["score"]])
        out["score_mae"] = float(np.abs(pred_s - gold_s).mean())

    if by_type["noul"]:
        pred_p = np.array([p["noul"] for p, g in by_type["noul"]])
        gold_b = np.array([1.0 if g["answer"] else 0.0 for p, g in by_type["noul"]])
        pred_b = (pred_p >= 0.5).astype(float)
        tp = float(((pred_b == 1) & (gold_b == 1)).sum())
        fp = float(((pred_b == 1) & (gold_b == 0)).sum())
        fn = float(((pred_b == 0) & (gold_b == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out["noul_f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out["noul_brier"] = float(((pred_p - gold_b) ** 2).mean())
        conf = np.array([max(p, 1 - p) for p in pred_p])
        correct = (pred_b == gold_b).astype(float)
        out["noul_ece"] = ece_score(conf, correct)

    accs = [v for k, v in [("choice", out.get("choice_accuracy")), ("noul", out.get("noul_f1"))] if v is not None]
    if accs:
        out["overall"] = float(np.mean(accs))
    return out


def _demo():
    predictions = [
        {"type": "choice", "choice": "billing", "confidence": 0.9},
        {"type": "noul", "noul": 0.8, "confidence": 0.8},
        {"type": "score", "score": 1.9},
    ]
    gold = [
        {"type": "choice", "answer": "billing"},
        {"type": "noul", "answer": True},
        {"type": "score", "answer": 2},
    ]
    result = evaluate(predictions, gold)
    assert result["choice_accuracy"] == 1.0
    assert result["noul_f1"] == 1.0
    assert abs(result["score_mae"] - 0.1) < 1e-6
    print("ok:", result)


if __name__ == "__main__":
    _demo()

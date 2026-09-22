from src.evaluation.metrics import evaluate


def test_choice_accuracy():
    predictions = [{"type": "choice", "choice": "billing", "confidence": 0.9}]
    gold = [{"type": "choice", "answer": "billing"}]
    assert evaluate(predictions, gold)["choice_accuracy"] == 1.0


def test_noul_f1_and_brier():
    predictions = [
        {"type": "noul", "noul": 0.9, "confidence": 0.9},
        {"type": "noul", "noul": 0.1, "confidence": 0.9},
    ]
    gold = [
        {"type": "noul", "answer": True},
        {"type": "noul", "answer": False},
    ]
    result = evaluate(predictions, gold)
    assert result["noul_f1"] == 1.0
    assert result["noul_brier"] < 0.02


def test_score_mae():
    predictions = [{"type": "score", "score": 1.5}]
    gold = [{"type": "score", "answer": 2}]
    result = evaluate(predictions, gold)
    assert abs(result["score_mae"] - 0.5) < 1e-9

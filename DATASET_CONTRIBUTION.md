# Contributing a dataset

## Required before opening a PR

1. **License check.** State the exact license (e.g. `apache-2.0`, `cc-by-4.0`) and where you verified it (HF dataset API `cardData.license`, the source repo's `LICENSE` file, or the paper's data statement). Datasets with no clear license, or a non-commercial/research-only clause, are not accepted — this project's checkpoints are meant to be redistributable.
2. **Schema conformance.** Every example must map to:

```json
{
  "state": "text, or a JSON-serializable dict/list",
  "questions": {
    "<question_id>": {
      "type": "choice | score | noul",
      "instructions": "the question being asked",
      "criteria": "dict (choice: label -> description) | list (score: ordered levels) | dict{false,true} (noul, optional)",
      "answer": "the gold label / level index / boolean",
      "target": "optional: a full probability distribution over options, if your source has soft labels or annotator disagreement"
    }
  }
}
```

This mirrors the typed-question format Laya's `Agent.predict` consumes (see `docs/LAYA_ANALYSIS.md`) — reusing it means any dataset here works with the evaluation harness and both `choice`/`score`/`noul` training paths without a special case.

3. **A loader in `src/data/`** that yields this schema from the raw source format, plus a short `datasets/<name>/README.md` stating: source, license, size, and which System-1 category it covers (intent, routing, boolean decision, scoring, safety, ...).
4. **A train/test split** that doesn't leak (if the source doesn't provide one, document how you made it and with what seed).

## What we're short on

Broad category coverage beyond what `LocalLLaMA/typed-decisions` already provides (see `docs/LAYA_ANALYSIS.md`): tool/agent routing, action selection, "does this need deeper reasoning" detection, and device/app-action datasets are all open gaps.

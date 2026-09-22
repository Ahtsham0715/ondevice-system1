# Contributing a checkpoint

## Required fields for a submitted checkpoint

A checkpoint PR (or Hugging Face upload linked from a PR here) must include a model card (`model_cards/<name>.md`) with:

- **Architecture**: encoder name + HF repo, hidden size, layers, head_layers, total params (measured via `DecisionModel.num_parameters()`, not estimated).
- **Training**: from scratch, fine-tuned, or distilled — and from which teacher/base if so. Config file used (link into `configs/`).
- **Training data**: every dataset used, with license (cross-reference `datasets/`).
- **Evaluation**: results on the shared evaluation suite (`src/evaluation/`) — Choice accuracy, Score MAE, Noul F1/Brier, ECE, at minimum. No cherry-picked subset without also reporting the full suite.
- **Device benchmarks**: at least one real measurement (CPU or mobile) from `src/evaluation/` — model size (fp32/int8/int4), P50 latency, peak RAM. `TBD` is acceptable for hardware you don't have access to; a fabricated number is not.
- **Limitations**: known weak spots, following Laya's own precedent of stating these plainly rather than omitting them (see `docs/LAYA_ANALYSIS.md`, "Calibration honesty").
- **License** of the resulting weights.

## Compatibility

A checkpoint's `rl_agent_config.json`-equivalent (`configs/<tier>/model.json`) must be enough to reconstruct the exact architecture via `build_model()` — this is what `src/model/decision_model.py`'s intended-to-be-added weight-compatibility check (mirroring Laya's `_verify_compatibility`) will validate against.

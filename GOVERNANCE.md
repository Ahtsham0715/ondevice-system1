# Governance

Early-stage, single-maintainer-led project. This document will grow formal structure (maintainer list, RFC process, voting) once there's more than one active maintainer — no point speculatively defining a process nobody has stress-tested yet.

## Decisions that need discussion first (open an issue)

- Changes to `src/model/decision_model.py` or the sequence format (`src/model/sequence.py`) — these are the compatibility contract every checkpoint and export target depends on.
- Adding a new model tier or removing one.
- Adopting a new teacher checkpoint for distillation.

## Decisions anyone can make via normal PR review

- New dataset loaders (with a verified license).
- New evaluation metrics or benchmark scripts.
- Documentation fixes, new export targets, quantization methods.

## Benchmark integrity

Anyone can submit a benchmark result (`BENCHMARKING.md`), but numbers that contradict an existing published result require the raw output (logs, JSON) attached to the PR — not just the summary table.

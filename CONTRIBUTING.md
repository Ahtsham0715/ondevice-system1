# Contributing

## Ground rules

- **No fabricated numbers.** Every accuracy, latency, or size figure in a PR must come from a command someone else can re-run. If you haven't measured it, write `TBD`, not a plausible-looking placeholder. See `BENCHMARKING.md`.
- **No dataset without a checked license.** Before adding a loader for a new dataset, verify its license (check the HF dataset API `cardData.license` or the source's own terms) and state it in the PR description and in `datasets/`.
- **Every architectural claim traces to `docs/LAYA_ANALYSIS.md` or your own experiment in `RESEARCH_LOG.md`.** Don't assert something about upstream Laya from memory — re-check the source.

## Code changes

1. Open an issue or discussion first for anything that changes `src/model/decision_model.py` or the sequence format — these affect every checkpoint's compatibility.
2. Config changes (new tier, new encoder) go in `configs/<tier>/`, not hardcoded into training scripts.
3. Run the relevant self-check before opening a PR:
   - `python -m src.model.decision_model` style module smoke tests where present (see individual files for their `if __name__ == "__main__":` checks).
   - Any evaluation change: re-run `src/evaluation/` against the smallest available config and paste the output in the PR.
4. Keep diffs to what the task needs — see the project's own engineering-principle section in this README-adjacent guidance: no speculative abstractions, no unused config knobs.

## Training / distillation experiments

Log every experiment in `RESEARCH_LOG.md` — hypothesis, config, result, and conclusion, including negative results. A distillation run that didn't beat the baseline is exactly as valuable to record as one that did.

## Dataset and model contributions

See `DATASET_CONTRIBUTION.md` and `MODEL_CONTRIBUTION.md` — both have a required-fields checklist that PRs are checked against.

## Attribution

Code adapted from Laya carries a header comment saying so (see `NOTICE`). If you port something from another Apache-2.0 or MIT project, do the same.

# Benchmarking

## The one rule

**Never fabricate a result. Leave the cell blank until it's measured.** This project exists partly to give honest, reproducible answers where marketing copy usually gives approximate ones — see how bluntly upstream Laya's own `BENCHMARKS.md` states its weaknesses (`docs/LAYA_ANALYSIS.md`, "Calibration honesty"). Keep that norm.

## What to measure

**Model quality** (`src/evaluation/`):
- Choice accuracy, Score MAE (mean absolute error against the gold ordinal level), Noul F1 / Brier score
- Expected Calibration Error (ECE), by (question type, option-count) bucket — mirrors how Laya buckets its own temperature fitting
- OOD / held-out behavior: report separately from in-training-mix results, labeled as such

**Device performance** (`src/evaluation/device_bench.py`):
- Model size on disk (fp32 / int8 / int4)
- Peak RAM during inference
- Cold-start / initialization time
- P50 / P95 latency, single question and batched
- Hardware and OS the measurement was taken on — always state it, never leave it implicit

## Comparison table format

```
| Model         | Params | Choice Acc | Noul F1 | ECE  | INT8 Size | P50 (device) |
|---------------|-------:|-----------:|--------:|-----:|----------:|-------------:|
| Laya Teacher  |   421M |            |         |      |           |               |
| OnDevice-Tiny |        |            |         |      |           |               |
```

Every blank cell means "not yet measured," not zero. Fill in the source dataset and hardware for each row in the surrounding text, not just the table.

## Jev comparisons

Do not write unsupported comparative claims ("faster than Jev"). If citing a public Jev benchmark, cite it exactly as published (see `docs/LAYA_ANALYSIS.md`'s Benchmarks section for the pattern: Laya's own docs label Jev numbers "published, never measured here"). We have no TypeSafe API access either — treat Jev figures the same way Laya's own docs do.

## Submitting a result

Attach the raw output (JSON/log) the summary table was generated from, plus the exact command run and hardware. PRs that only add a summary-table row without raw output will be asked for it before merge (see `GOVERNANCE.md`).

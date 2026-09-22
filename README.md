# OnDevice-System1

A community-driven, open-source model family for fast, calibrated, **on-device typed decisions** — derived from and building on [Laya](https://github.com/NandhaKishorM/laya)'s System-1 approach, optimized for phones, laptops, and edge hardware instead of datacenter GPUs.

**This is a model project, not an app.** The deliverable is weights, training code, evaluation, quantized variants, model cards, and inference examples — not a mobile wrapper.

```
state (text / JSON / conversation)
        │
        ▼
On-Device System-1 Model (30–200M params, one forward pass, no generation)
        │
        ▼
typed decisions: choice · score · noul  (label, ordinal level, calibrated probability)
```

## Why

Large autoregressive LLMs are inefficient for tiny, frequent decisions — intent, routing, tool selection, confidence, boolean gates. Generating and parsing tokens to answer "is this urgent, yes or no" is doing far more work than the question needs. A non-autoregressive typed-decision model answers in a single forward pass: nothing to parse, nothing to hallucinate. Laya demonstrated this approach at 322–421M params on GPU-class hardware; this project asks how far it shrinks while staying useful on a phone, a laptop CPU, or a Raspberry Pi.

## Status

Early. Phase 1 (understand Laya) is done — see [`docs/LAYA_ANALYSIS.md`](docs/LAYA_ANALYSIS.md) for the full architecture, training-loss, and licensing writeup, produced by reading the upstream source directly, not from memory. Phase 2 (design + train the first student) is in progress — see [`ROADMAP.md`](ROADMAP.md) for current milestone and [`RESEARCH_LOG.md`](RESEARCH_LOG.md) for every experiment, including failures.

No trained checkpoint exists yet. Every number in this repository that looks like a benchmark result is either cited from upstream Laya (marked as such) or left blank pending an actual run — see `BENCHMARKING.md` for why that rule is non-negotiable here.

## Architecture in one paragraph

A bidirectional transformer encoder (any HF `AutoModel` — ModernBERT, BERT, mmBERT, ...) plus a shallow transformer head. Each answer option gets one `[MASK]` token in the input; the head scores each option's final hidden state with a small MLP, and softmax over those scores gives the answer — a label (`choice`), an ordinal expected value (`score`), or a calibrated probability (`noul`). One forward pass answers every question about a state at once. Full detail and the exact sequence format: [`docs/LAYA_ANALYSIS.md`](docs/LAYA_ANALYSIS.md).

## Repository layout

```
src/
  model/          decision model architecture, sequence builder, RLCD loss, calibration
  training/        training loops (from-scratch, fine-tune)
  distillation/    teacher/student distillation experiments
  data/            dataset loaders and schemas
  evaluation/      accuracy, calibration, and on-device performance benchmarks
  quantization/    INT8 / INT4 export and PTQ/QAT
  export/          ONNX / CoreML / TFLite export
configs/           tiny / small / base model + training configs
benchmarks/        measured results (populated as experiments complete)
datasets/          community-contributed dataset schemas
model_cards/       one per released checkpoint
docs/              architecture analysis, design docs
```

## Model family (candidate — validate before trusting)

| Model | Encoder | Total params (measured) | Status |
|---|---|---:|---|
| OnDevice-Tiny | `google/bert_uncased_L-4_H-512_A-8` | 32.3M (28.8M encoder + 3.6M head) | builds, untrained |
| OnDevice-Small | `google/bert_uncased_L-8_H-512_A-8` | 48.1M (41.4M encoder + 6.7M head) | builds, untrained |
| OnDevice-Base | `answerdotai/ModernBERT-base` | 164.0M (149.0M encoder + 15.0M head) | builds, untrained |

These are starting points chosen because they're existing, license-clean, pretrained encoders — not arbitrary parameter targets. `docs/LAYA_ANALYSIS.md` explains the reasoning. Params measured 2026-09-22 via `DecisionModel.num_parameters()` (see `RESEARCH_LOG.md`); each config was confirmed to actually build, not just estimated. They will change if evaluation says they should.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for code, [`DATASET_CONTRIBUTION.md`](DATASET_CONTRIBUTION.md) for datasets, [`MODEL_CONTRIBUTION.md`](MODEL_CONTRIBUTION.md) for checkpoints, and [`BENCHMARKING.md`](BENCHMARKING.md) for how to submit a measured result.

## License

Apache 2.0 (see `LICENSE`). Portions of `src/model/` are adapted from Laya (Convai Innovations, Apache 2.0) — see `NOTICE`.

# typed-decisions

- **Source**: [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) on Hugging Face.
- **License**: Apache-2.0 (verified via the HF dataset API `cardData.license`, 2026-09-22).
- **Size**: 1,200 train examples / 400 test examples, 5 `choice`/`score`/`noul` questions per example (~2,000 test decisions), across 4 workflow themes.
- **Categories** (System-1 decision types, see project README): agent-trace observability (action/risk/urgency decisions), customer service, invoice processing, security incidents.
- **Splits**: `train` / `test` per workflow, plus an `all` config combining all four.
- **Schema**: each row has `state` (JSON), `questions` (JSON: per-question type/instructions/criteria), `gold` (JSON: per-question label + full soft probability distribution — annotator disagreement is preserved, not just a single top label), `factors`, `label_agreement`, `n_questions`. Loader: `src/data/typed_decisions.py`.
- **This is also Laya's own published fine-tuning dataset** — see `docs/LAYA_ANALYSIS.md`, "Training: RLCD". Using it here means our teacher-baseline and student numbers are directly comparable to upstream's own `laya-typed-decisions` fine-tune, on the identical data.

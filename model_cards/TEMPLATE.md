# Model card: <name>

Copy this file to `model_cards/<name>.md` when submitting a checkpoint (see `MODEL_CONTRIBUTION.md` for the required-fields rationale).

## Architecture
- Encoder: `<HF repo>` (`<params>` params)
- Head: `head_layers=<n>`, hidden size `<d>`
- Total params (measured via `DecisionModel.num_parameters()`): `<total>`
- Config: `configs/<tier>/model.json`

## Training
- Method: from scratch / fine-tuned from `<checkpoint>` / distilled from `<teacher>`
- Data: `<datasets, with license>`
- Recipe: `<link to script + hyperparameters actually used>`

## Evaluation

| | Choice Acc | Score MAE | Noul F1 | Noul Brier | ECE |
|---|---:|---:|---:|---:|---:|
| this checkpoint | | | | | |

Dataset/split evaluated on: `<...>`. Raw output: `<path or link>`.

## Device benchmarks

| Hardware | Format | Size | P50 latency | Peak RAM |
|---|---|---:|---:|---:|
| | fp32 / int8 / int4 | | | |

## Limitations

State known weak spots plainly — see `docs/LAYA_ANALYSIS.md` "Calibration honesty" for why this matters.

## License
`<license>`

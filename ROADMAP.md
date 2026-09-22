# Roadmap

## Current milestone: Phase 1 → Phase 2 (understand Laya → reproduce → evaluate → first student)

```
[x] Understand Laya          docs/LAYA_ANALYSIS.md
[x] Reproduce upstream inference   real CPU latency measured, see RESEARCH_LOG.md 2026-09-22
[x] Build evaluation suite    src/evaluation/metrics.py, verified against real teacher output
[x] Design smaller student    3 tier configs, all verified to build with real measured params
[~] Train/distill student     RLCD training loop verified end-to-end (smoke run); real training job not yet run; distillation not started
[ ] Compare against teacher    blocked on full teacher baseline (stopped mid-run) + a real student training job
[ ] Quantize
[ ] Export
[ ] Run on ARM/mobile hardware
[ ] Publish benchmark
```

## Completed

- 2026-09-22: Cloned and read upstream Laya (`laya/*.py`, `train_ddp.py`, `BENCHMARKS.md`, `LICENSE`). Wrote `docs/LAYA_ANALYSIS.md`. Verified: `answerdotai/ModernBERT-base`, `jhu-clsp/mmBERT-small`, `google/bert_uncased_L-{4,8}-H-512-A-8` exist on HF and are viable student encoder candidates. Verified `LocalLLaMA/typed-decisions` (Laya's own fine-tuning dataset) is Apache-2.0. Adapted `DecisionModel`, sequence builder, RLCD loss, and calibration code into `src/model/` with attribution (`NOTICE`).
- 2026-09-22: Repo scaffolded (`src/`, `configs/{tiny,small,base}/model.json`, community docs). Git initialized.

## Next experiments (Phase 2)

1. [x] Install `laya` + run its README example locally — done 2026-09-22, real CPU latency 813.3ms (see RESEARCH_LOG.md). Two real environment compat bugs found and fixed (`src/compat.py`).
2. [x] Load `LocalLLaMA/typed-decisions`, write `src/data/typed_decisions.py` — done, real schema verified, one real bug found and fixed (optional `criteria` on `noul` questions).
3. [~] Teacher baseline on `typed-decisions/test`: harness built and verified (`scripts/eval_teacher_baseline.py`, real 20-example smoke run done), but the full 400-example run was started then stopped mid-run at the user's request — not yet a reportable number. Resume with: `PYTHONPATH=. python scripts/eval_teacher_baseline.py --workflow all --out benchmarks/teacher_baseline_laya_english_cpu.json`.
4. [x] All three tier configs (`configs/{tiny,small,base}/model.json`) verified to actually build — real measured params: tiny 32.3M, small 48.1M, base 164.0M (RESEARCH_LOG.md).
5. [x] `src/training/train_rlcd.py` verified end-to-end on a real 16-example smoke run: trains, saves, reloads, and produces a working forward pass (RESEARCH_LOG.md). Not yet run as a real training job (more data, more epochs, real convergence check).
6. **Next**: run a real training job (full `typed-decisions/train`, several epochs) on `configs/base` (ModernBERT-base — same family as the teacher, best distillation compatibility per `docs/LAYA_ANALYSIS.md`), then evaluate against the teacher baseline once that's also complete.
7. Only after (6) produces a working from-scratch baseline: add logit-distillation from the Laya teacher and compare (`src/distillation/` — not started).

## Blockers

- No GPU in the current development environment — it's an Intel x86_64 Mac (i9-9980HK), CPU only, no CUDA and no MPS (MPS needs Apple Silicon). This is representative of one real on-device target (laptop CPU) but slow for large sweeps; larger training runs will need a cloud GPU the user provisions separately (Colab/Kaggle free tier, as Laya's own fine-tuning notebook uses, is a candidate — same pattern, no new infra needed).
- mmBERT license not yet independently re-verified — blocks committing to it for the multilingual tier until checked.

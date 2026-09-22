# Research log

Every experiment, including failures. Newest first.

---

## 2026-09-22 — Reproduce upstream inference (real measurement, not cited)

**Setup**: Intel x86_64 Mac (i9-9980HK), CPU + an MPS-capable AMD dGPU, no CUDA. venv on Python 3.12 (Homebrew). This environment's package mirror caps `torch` at 2.2.2 (not the current upstream release) and only had cached metadata for a narrow set of package versions, which made unpinned `pip install` resolution extremely slow (~15+ min of backtracking before it was killed) — pinning exact versions (`torch==2.2.2`, `transformers==4.48.3`, `tokenizers<0.22`) and installing `laya` with `--no-deps` afterward fixed that; total real install time once pinned was a few minutes, dominated by the actual 150.8 MB torch wheel and the ~840 MB `ModernBERT-large` checkpoint download (the checkpoint came from HF's CDN via `xet`, much faster than the pip mirror — ~5 MB/s vs ~100-300 KB/s).

**Two real compatibility bugs found and fixed** (not upstream bugs — this project's specific dependency-version intersection; documented in `src/compat.py`):
1. `transformers==4.48.3`'s ModernBERT implementation applies `@torch.compile(dynamic=True)` at class-definition time. `torch==2.2.2` unconditionally raises `RuntimeError("Dynamo is not supported on Python 3.12+")` on *any* `torch.compile()` call under Python 3.12, so importing ModernBERT failed before Laya's own `reference_compile = False` override (`agent.py`) ever got a chance to run. Fix: `src/compat.py` monkeypatches `torch.compile` to a no-op when `torch<2.4` and `Python>=3.12` — inert everywhere else, and changes nothing at runtime since Laya already disables compilation for inference.
2. `torch==2.2.2`'s `torch.autocast()` does not support `device_type="mps"` at all (added in a later torch release) — and Laya's own device auto-detection picks MPS whenever `torch.backends.mps.is_available()` is true, which it is on this machine (MPS works with this Intel Mac's AMD Radeon Pro dGPU, not just Apple Silicon — a correction to an earlier note in this log). `laya/agent.py`'s `system_one` calls `torch.autocast(device_type=self.device.type, ..., enabled=use_amp)` unconditionally, and the constructor validates `device_type` before even checking `enabled`, so it raised regardless of the (correctly `False`) `enabled` flag. Fix: force `device="cpu"` explicitly in our example/eval scripts — which is not just a workaround, it's one of this project's actual on-device targets.

**Result** (`examples/reproduce_upstream_inference.py`, `convaiinnovations/laya`, CPU, warm call):
- `department`: `billing`, confidence `0.864` (README's own example gets `0.94` on their T4/GPU run — same architecture, different hardware/precision path, not a discrepancy worth chasing)
- `urgency` score: `1.44`
- `churn_risk`: `0.8248`
- **latency: 813.3 ms**, single call, 3 questions, warm (post-warmup), CPU, fp32 (this torch/device combo forces fp32 per `agent.py`'s dtype-selection logic)
- Real runtime warning fired exactly as `docs/LAYA_ANALYSIS.md` predicted: the shipped `choice:11+` temperature bucket (`0.1006`) is outside `[0.5, 5.0]` and got clamped, with the exact warning text described in that doc — confirms the analysis was reading the code correctly, not just plausibly.

**Comparison to upstream's cited numbers**: Laya's README states ~39.5 ms (T4 GPU, presumably bf16) and separately warns that a CPU fallback runs "10-15x slower (~200-500 ms rather than ~35 ms)" for the *fallback* path specifically. Our measured 813 ms is above even that estimated range — plausibly because this is a full CPU-native run (not a CUDA-OOM fallback), fp32 (not the bf16 the config requests, since `agent.py` forces fp32 on CPU), and an older 2019-era mobile CPU (i9-9980HK, laptop thermal envelope), not a modern desktop-class chip. This is exactly the kind of real on-device number Laya's own benchmarks don't publish (T4 GPU only) — first real evidence for our project's "on-device targets" gap, and the number every OnDevice-* checkpoint's own CPU latency should be compared against later, on the same machine.

**Conclusion**: Milestone step "Reproduce upstream inference" — done, with real, first-party numbers, not cited ones. `pytest` (5 tests: `DecisionModel` forward-shape/param-count checks, `evaluate()` metric checks) passes on the adapted `src/model/` and `src/evaluation/` code. Next: teacher zero-shot baseline on `typed-decisions/test` (`scripts/eval_teacher_baseline.py`).

---

## 2026-09-22 — Teacher zero-shot baseline on typed-decisions

**Bug found and fixed while building this**: `src/data/typed_decisions.py`'s loader assumed every question dict has a `criteria` key; real data has `noul` questions that omit it entirely (matches Laya's own defensive `qdef.get("criteria")` in `agent.py`). Fixed with `.get("criteria")`. This is exactly why "run it for real" is worth doing even on code that looks correct by inspection.

**Smoke test** (`--limit 20`, i.e. 4 examples / 20 decisions — small, high-variance, not a real result): `choice_accuracy=0.475, choice_ece=0.321, score_mae=0.849, noul_f1=0.095, noul_brier=0.352, noul_ece=0.543, overall=0.285`. Confirmed the harness runs end-to-end and produces plausible-shaped numbers (high ECE is consistent with `docs/LAYA_ANALYSIS.md`'s note that base checkpoints "ship over-confident"). Not reporting this as a benchmark row — n=20 is too small; see the full run below.

**Full run** (all 400 test examples / ~2,000 decisions, `convaiinnovations/laya`, CPU, zero-shot, `workflow=all` — matching upstream's own typed-decisions benchmark exactly): started, then stopped partway through at the user's request to pause running processes and focus on code. Not completed, no partial output written (the script only writes JSON at the end, so `benchmarks/` is still empty — nothing fabricated in its place). Re-run with the same command when ready: `PYTHONPATH=. python scripts/eval_teacher_baseline.py --workflow all --out benchmarks/teacher_baseline_laya_english_cpu.json` (~5-7 min estimated on this CPU, from the per-example latency measured above).

---

## 2026-09-22 — Verify all three tier configs actually build

**What**: ran `build_model()` against `configs/{tiny,small,base}/model.json` for real (downloading each encoder), not just checking the HF repos exist. This was the one remaining unverified claim in the model family table — "builds" had not actually been tested for any config, only estimated from each encoder's own published size.

**Result** — all three build without error; `DecisionModel.num_parameters()` gives exact counts, now written into each config and `README.md` (replacing the earlier `target_params_millions` estimates):

| tier | encoder | encoder params | head params | total |
|---|---|---:|---:|---:|
| tiny | `google/bert_uncased_L-4_H-512_A-8` | 28,763,648 | 3,550,979 | **32,314,627** |
| small | `google/bert_uncased_L-8_H-512_A-8` | 41,373,184 | 6,703,363 | **48,076,547** |
| base | `answerdotai/ModernBERT-base` | 149,014,272 | 14,969,347 | **163,983,619** |

Head overhead is non-trivial at small scale — 11% of tiny's total params, 14% of small's, vs. 9% of base's — worth remembering when comparing tiers: the head isn't free just because the encoder is small. All three land close to the earlier order-of-magnitude estimates in `docs/LAYA_ANALYSIS.md`, confirming those were reasonable starting points, not off-base guesses.

**Not yet done** (at time of writing): none of these three have been trained — `build_model(pretrained=True)` loads pretrained encoder weights but the decision head (`type_emb`, `scorer`, `act_head`) is freshly initialized every time. See next entry — this was closed the same day.

---

## 2026-09-22 — Verify src/training/train_rlcd.py end-to-end (real training run, not just written)

**What**: ran the actual RLCD training loop — `python -m src.training.train_rlcd --config configs/tiny/model.json --out /tmp/train_smoke_tiny --epochs 1 --limit 16 --batch_size 4` — the one remaining genuinely unverified piece of code in the project (it had been written but never executed).

**Result**: ran clean, no crashes. 16 examples → 80 question items (5 questions/example, matches the dataset's real `n_questions`), 20 optimizer steps (80/batch_size 4), final epoch avg loss 1.139 (single epoch, too few steps to expect convergence — this is a pipeline check, not a training run). Checkpoint saved to `model.safetensors` (129,269,104 bytes — exactly 32,314,627 params × 4 bytes fp32, confirming the save path and param count agree) + `tokenizer/` + `rl_agent_config.json`.

**Round-trip verified** (separate check, not part of the training script): reloaded the saved checkpoint via `build_model()` + `load_state_dict(strict=True)` — all keys matched with no missing/unexpected tensors — then ran a real forward pass on a new example (`build_sequence` → `collate_items` → `model()`) and got a valid probability (`p(true)=0.536`, unsurprising given only 20 steps on 16 examples — close to the untrained prior, not yet meaningfully fit).

**Conclusion**: the full pipeline — data loading → tokenization → collation → forward → RLCD loss → backward → optimizer step → checkpoint save → checkpoint reload → forward pass — is now verified working end-to-end on real data, not just unit-tested in isolation. This was the last unverified item from the "all code done?" status check earlier today. Every piece of `src/` that's expected to work right now has been run for real at least once. What's still correctly unstarted, by roadmap design: `src/distillation/`, `src/quantization/`, `src/export/` (these need a properly-trained checkpoint to operate on, not a 20-step smoke test) and the full 400-example teacher baseline (stopped earlier per user request, command recorded above to resume).

---

## 2026-09-22 — Phase 1: understand Laya

**Hypothesis**: Laya's architecture and training loss are small/simple enough (an encoder + shallow head, not an autoregressive generator) that shrinking it is a standard distillation problem, not novel research.

**What was done**:
- Cloned `github.com/NandhaKishorM/laya` at commit `573e5b6` (v0.3.5) and read all 8 files in `laya/` (1,861 lines) plus the fine-tuning notebook's training script in full.
- Fetched `rl_agent_config.json` for `convaiinnovations/laya` directly from the HF Hub API to get real (not estimated) hyperparameters and per-checkpoint temperatures.
- Verified via the HF Hub API that `answerdotai/ModernBERT-base` (149M, same family as the English teacher), `jhu-clsp/mmBERT-small`, and `google/bert_uncased_L-4_H-512_A-8` / `L-8_H-512_A-8` (Turc et al. BERT-Small/Medium) exist and are plausible student backbones.
- Verified via the HF Hub dataset API that `LocalLLaMA/typed-decisions` (Laya's own fine-tuning dataset) is licensed `apache-2.0`.

**Result**: Confirmed the hypothesis. `DecisionModel` (`laya/common.py`) is encoder-agnostic — `d = encoder.config.hidden_size` is the only architecture-derived quantity the head depends on — so any HF bidirectional encoder plugs in unchanged. The RLCD loss (`proper_reward`, logit-space policy gradient in `train_ddp.py`) is likewise architecture-agnostic.

**Important scoping finding, not initially assumed**: the upstream repo publishes fine-tuning code only. The base-checkpoint *pretraining* recipe and data are not public. This changes our plan from "reproduce Laya's training pipeline end-to-end" to "use existing small pretrained encoders as students + Laya's own published RLCD fine-tuning loss to adapt them, with the base checkpoints as a distillation teacher" — see `docs/LAYA_ANALYSIS.md` for the full reasoning.

**Also found**: Laya documents its own clear architectural weak point (shared `head_max_len` token budget across all options in a `choice` question, causing Banking77's 77-option collapse to 0.425 vs Jev's 0.870). Adapted `src/model/sequence.py` with a per-option token floor instead of a shared shrinking budget, specifically to not reproduce this failure mode at smaller scale. Not yet benchmarked against upstream's approach — that's a pending experiment, not a claimed fix.

**Conclusion**: Phase 1 complete. Full writeup: `docs/LAYA_ANALYSIS.md`. Proceeding to Phase 2: reproduce upstream inference locally, then build the evaluation suite before touching any student architecture.

**Environment note**: development machine is an Intel x86_64 Mac (i9-9980HK), CPU only — no CUDA, no MPS (MPS requires Apple Silicon). Python 3.14 (system default) has no `torch` wheel available yet as of this date — switched to Python 3.12 (Homebrew) for the project venv. The available PyPI mirror in this environment only serves `torch` up to 2.2.2 (not the newest upstream release) — pinned to that. `torch==2.2.2`/`transformers`/`laya` install in progress at time of writing; next log entry will report the first real inference run and its CPU latency, which will be directly relevant to the laptop-CPU on-device target rather than a proxy for it.

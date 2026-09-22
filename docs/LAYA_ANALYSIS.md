# Laya analysis

Source: [github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), commit `573e5b6` (2026-09-21, v0.3.5), studied by cloning the repo and reading `laya/*.py` line by line, downloading the published checkpoint configs from Hugging Face, and executing the fine-tuning notebook's training script statically. Nothing in this document is guessed; where something could not be verified from the published repo, it is marked as such below.

## What Laya actually ships

`pip install laya` is an **inference SDK**, not a training framework. The repository contains:

- `laya/` (1,861 lines, 8 files) — model definition, sequence builder, inference `Agent`, a language-routing wrapper (`Router`), preset question schemas, and a large-label-space shortlist helper. No training loop lives here.
- `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` — the **only** published training code. It fine-tunes an already-pretrained checkpoint; it does not pretrain one.
- `research/` — benchmark scripts and their raw JSON results (used to produce `BENCHMARKS.md`).
- Three checkpoints on Hugging Face (`convaiinnovations/laya`, `.../laya-multilingual`, `.../laya-typed-decisions`), each Apache-2.0.

**The base-checkpoint pretraining recipe and data are not published.** The README calls the approach "RLCD" (reinforcement learning against strictly proper scoring rules) and the fine-tuning script implements that loss end-to-end, so the *loss function* is fully known and reproducible. What is not known: the pretraining corpus, question/schema mix, curriculum, or compute budget that produced the zero-shot `laya` / `laya-multilingual` weights before any fine-tuning. Treat those two checkpoints as a black-box teacher signal, not a reproducible recipe.

## Architecture

Defined in `laya/common.py`. One class, `DecisionModel(nn.Module)`:

```
input_ids, attention_mask
        │
        ▼
 encoder (HF AutoModel, e.g. ModernBERT / mmBERT)   ← the only large component
        │  last_hidden_state [B, L, d]
        ▼
 + type_emb[qtype]                                   (Embedding(3, d), broadcast-added to every token)
        │
        ▼
 head: nn.TransformerEncoder, head_layers=2, batch_first, norm_first  ← small, encoder-width, trainable on top
        │
        ▼
 gather hidden states at each [MASK]-marker position (one per answer option)
        │
        ▼
 scorer: LayerNorm → Linear(d,d) → GELU → Linear(d,1)   → one logit per option
        │
        ▼
 softmax over the option logits (masked to the question's actual option count) → the answer
```

A second small head (`act_head`, a 2-layer MLP over `[CLS] + 4 hand-built features`: top-1 prob, top1−top2 margin, normalized entropy, option count) produces an auxiliary "act vs escalate" probability, returned as `answers[...]["action"]["act_probability"]`. It is trained (`cfg["act_costs"]`, `cost_wrong_act"`) but the public `Agent.predict` does not gate on it — that decision is left to the caller.

**Single forward pass, no autoregression.** Every option gets one `[MASK]` token in the input; the model reads all options and the state simultaneously and scores them in parallel. This is the whole basis of the 33 ms / single-pass claim, and it directly motivated the "typed decision, not a chatbot" framing in our project brief: there is no generation loop to distill away, only an encoder + a shallow scoring head.

### Sequence format (`build_sequence`)

```
[CLS] <qtype> question: <instructions> [SEP] [MASK] <opt0 tokens> [MASK] <opt1 tokens> ... [SEP] <state tokens> [SEP]
```

- `head_max_len` (192 English / 256 multilingual by default) is a **shared budget** between the instructions and all options. Options are truncated to fit; if the budget is too tight, each option is capped to `max(4, (head_max_len-16)//n_options)` tokens.
- `max_len − len(head)` tokens are left for the serialized state (dict/list state is JSON-dumped).
- This shared, fixed-size head budget is the documented root cause of Laya's one clear weakness: Banking77 (77 options) gets ~3–4 tokens/option and scores 0.425 vs Jev's 0.870. Anything we design must either make this budget configurable per question or adopt a different option-encoding scheme (see "Design implications" below).

### The three typed primitives

| Type | `QTYPES` id | Options | Head output |
|---|---|---|---|
| `choice` | 0 | N labeled options (dict `label: description`) | argmax label + full softmax + entropy-based confidence |
| `score` | 1 | K ordinal levels (list, index = level) | `score = Σ i·p(i)` (expected value under the softmax) + full distribution |
| `noul` | 2 | exactly 2 (`false`, `true`) | `p(true)` directly used as a calibrated probability |

"Noul" is Laya's boolean/probability primitive — not a standard term elsewhere; functionally it's `choice` restricted to 2 fixed options where the positive-class probability is surfaced as the answer instead of an argmax label. Confidence for `choice`/`score` is `1 − H(p)/log(k)` (normalized Shannon entropy, `confidence_from_probs`); confidence for `noul` is `max(p, 1−p)`.

### Encoder backbones actually used

| Checkpoint | encoder | encoder params (HF config) | total params | context | temperature (choice/score/noul) |
|---|---|---|---|---|---|
| `laya` | `answerdotai/ModernBERT-large` | ~395M | **421M** | 512 (`head_max_len=192`) | fetched from `rl_agent_config.json`: `[1.637, 1.251, 1.983]` (English root) |
| `laya-multilingual` | mmBERT-base | ~307M | **322M** | 1024 (`head_max_len=256`) | ships **unfitted** (README states no temperatures shipped) |
| `laya-typed-decisions` | `answerdotai/ModernBERT-large` | ~395M | **421M** | 1024 | fine-tuned from `laya` root, config confirms `"fine_tuned_from_checkpoint": true`, `"updates": 7313`, `"epochs_completed": 1`, `"hours": 1.96` |

(Head overhead beyond the encoder — 2-layer `TransformerEncoder` at encoder width + scorer + act_head + type_emb — accounts for the remaining ~20–26M params in each case; not broken out separately in the config, computed here as `total − encoder`.)

`rl_agent_config.json` for `convaiinnovations/laya` (fetched directly, not reconstructed):

```json
{
  "encoder": "answerdotai/ModernBERT-large",
  "head_layers": 2,
  "max_len": 512,
  "head_max_len": 192,
  "max_prefixes": 6,
  "act_costs": {"escalate": 0.5},
  "cost_wrong_act": 3.0,
  "amp_dtype": "bf16"
}
```

## Training: RLCD (verified from `train_ddp.py`, the notebook's fine-tuning script)

This is the **only** training code Laya publishes, and it fine-tunes, not pretrains. Mechanism, read directly from the script:

1. **Not autoregressive sampling.** Because the model outputs a fixed-size logit vector (one logit per option) rather than tokens, "RL" here means a zero-order / evolution-strategies-style policy gradient directly in **logit space**:
   - Draw `G=4` (`GROUP_SIZE`) noise samples `ε ~ N(0, σ²)` per example, projected to zero-sum within the valid-option mask (`ε -= ε.mean_over_valid_options`).
   - Perturbed logits `z = logits.detach() + ε` → softmax → `q`.
   - `σ` anneals `0.4 → 0.1` linearly over training (`SIGMA_START/SIGMA_END`), i.e. exploration shrinks as training progresses — standard ES annealing.
2. **Reward = a strictly proper scoring rule** (`proper_reward` in `common.py`): `log_score + w_sph·spherical_score`, minus a ranked-probability-score penalty (`w_rps·rps`) specifically for ordinal `score`-type questions (this is what makes ordinal errors cost more the further off they are — a `score` head predicting "level 0" when the truth is "level 3" is penalized harder than predicting "level 2"). Reward is baseline-corrected (`advantage = r − mean(r over the 4 samples)`) and standardized.
3. **Policy-gradient loss**: `logp ∝ -||z - logits||² / (2σ²)` (Gaussian score function), then `loss_rl = -(advantage · logp).mean()`.
4. **Plus a supervised soft-label term**, full weight 1.0: `loss_ce = cross_entropy(logits, target)` where `target` is a full probability distribution, not a one-hot label — so the base training data already carries soft targets (a teacher distribution, human labels with disagreement, or both — **not stated in the published fine-tuning notebook**, since it only consumes a pre-built `target` tensor from `LocalLLaMA/typed-decisions`).
5. **Two learning rates**: encoder `2.5e-5`, head `1e-4` — encoder is fine-tuned, not frozen, but at ~4x lower LR than the newly-initialized-ish decision head.
6. **Effective batch 64** (micro-batch 8 × grad-accum 4 × 2 GPUs), 4 epochs, cosine LR schedule, gradient clipping at norm 1.0, fp16 AMP, gradient checkpointing on both encoder and head.
7. **Post-training calibration**: one temperature per `(qtype, option-count bucket)` fitted by LBFGS-minimizing NLL on a 400-example calibration slice, clamped to `[0.1, 10.0]` at fit time and further clamped to `[0.5, 5.0]` at load time in `agent.py` (a shipped temperature outside that band is rejected with a warning rather than trusted — the repo has clearly been burned by a bad calibration fit before, see the comment in `common.py` about a `choice:11+` bucket of 0.1 sharpening a 0.24 probability into a published 0.99).

**Data for fine-tuning**: `LocalLLaMA/typed-decisions` on Hugging Face — 1,200 train / 400 test cases, 2,000 test decisions across 4 workflow themes (invoice processing, security incidents, customer service, agent-trace observability). This is the only dataset whose provenance is concretely named in the published repo. Everything else in `BENCHMARKS.md` (MASSIVE, XNLI, AG News, DAIR Emotion, Banking77, deepset/prompt-injections) is used for **evaluation only**, against the zero-shot base checkpoints — those are not claimed as training data, though the benchmarks doc does flag which held-out-vs-in-training-mix status it knows for each ("AG News: in training mix" — meaning the *base* pretraining, not the published fine-tune, saw it; consistent with the pretraining corpus being undisclosed).

## Calibration honesty (worth preserving as a design value, not just a number)

Laya's own documentation is unusually candid about where it's weak, and that candor is worth explicitly carrying into our project's norms:
- Base checkpoints score **below the majority-class baseline** zero-shot on typed-decisions (0.36–0.35 vs 0.461 majority-class, 0.318 random) — all capability comes from fine-tuning.
- `laya-multilingual` ships with **no fitted temperatures at all**.
- English checkpoint **collapses outside English while staying confident** (Khmer: 0.000 accuracy at 95.2% mean confidence) — confidence gating cannot catch this failure mode, which is exactly why the `Router` does script/language detection *before* the forward pass rather than trusting post-hoc confidence.
- Ordinal `score` is explicitly named the weakest primitive (SST-5: 0.372).
- Banking77 (77-option `choice`) is a stated, architectural, token-budget failure, not swept under the rug.

## License

- **Code**: Apache License 2.0 (full text in upstream `LICENSE`), Convai Innovations.
- **Weights**: README states Apache 2.0 for all three checkpoints (`convaiinnovations/laya`, `-multilingual`, `-typed-decisions`) — consistent with the PyPI package license classifier.
- **Base encoders**: `answerdotai/ModernBERT-{base,large}` are Apache 2.0 (AnswerDotAI). mmBERT family (`jhu-clsp/mmBERT-*`) — license not independently re-verified in this pass; **verify before redistributing any mmBERT-derived checkpoint**, flagged here rather than assumed.
- **Fine-tuning dataset** `LocalLLaMA/typed-decisions` — **verified Apache-2.0** (`cardData.license` on the HF dataset API, checked 2026-09-22). 4 workflow themes (`agent_trace_observability`, `customer_service`, `invoice_processing`, `security_incidents`) plus an `all` split combining them, each with `train`/`test` parquet files. Usable for our own fine-tuning/eval.
- Apache 2.0 permits derivative works, modification, and redistribution provided attribution is retained and changed files are marked (§4(b), §4(c)). Our `common.py`-derived code carries an explicit attribution header and this project's `NOTICE` file credits Convai Innovations / Laya.

## Benchmarks (as published, reproduced here only as citations — not re-measured by us yet)

From `BENCHMARKS.md`, `research/results/*.json`:

| | Laya (routed) | Jev 1.13.0 (published, third-party, not independently measured by Laya's authors) |
|---|---|---|
| typed-decisions, 2,000 decisions | 0.766 | 0.727 |
| AG News (4 labels) | 0.950 | 0.910 |
| DAIR Emotion (6 labels) | 0.595 | 0.480 |
| Banking77 (77 labels) | 0.425 | 0.870 (Jev leads — architectural, see above) |
| ECE, post temperature-fit | 0.081 | 0.246 |
| p50 latency, 1 question | 32.8 ms (T4 GPU) | 236–276 ms |

Per this project's brief: we make no unsupported comparative claims of our own about Jev. These numbers are Laya's own published citations, reproduced here for context on the ecosystem we're building next to, not verified independently by us.

## What we can legally and technically reuse

| Component | Reuse? | Notes |
|---|---|---|
| `DecisionModel` architecture (type-embedding + shallow transformer head + per-option `[MASK]`-gather scorer) | **Yes** | Apache 2.0, architecture is encoder-agnostic — this is the part we adapt for smaller backbones. |
| `build_sequence` / prompt format | **Yes, as a starting point** | Reuse the format; the shared `head_max_len` budget is the one part we should redesign (see below) since it's Laya's own documented failure mode. |
| `proper_reward` / RLCD loss | **Yes** | Fully specified in `train_ddp.py`; portable to any encoder size. |
| Calibration approach (per-bucket temperature scaling, clamped) | **Yes** | Cheap, well-tested, and the clamping-with-warning pattern is worth keeping. |
| ModernBERT-large / mmBERT-base checkpoints as **teachers** | **Yes**, for distillation signal (not redistributing their weights unmodified — we'd be producing derivative student checkpoints) | Apache 2.0 checkpoints, usable as frozen teachers. |
| ModernBERT-base (149M) as an actual **student backbone** | **Yes** | Existing, independently pretrained, Apache 2.0, same tokenizer family as the `laya`/`laya-typed-decisions` teacher — best distillation compatibility of any candidate we found. |
| mmBERT-small (as multilingual student backbone) | **Provisionally** | Exists on HF, same family as `laya-multilingual`'s mmBERT-base; license not yet independently re-verified (see above) — verify before use, don't assume. |
| Base pretraining data/recipe | **No — not published.** | We cannot reproduce Laya's own pretraining. Our path is: (a) use existing small pretrained encoders as students, (b) distill decision-head + encoder-adaptation behavior from the Laya teacher via the RLCD loss above, not full from-scratch LM pretraining. |
| `LocalLLaMA/typed-decisions` dataset | **Yes — verified Apache-2.0** | Our first real fine-tuning/eval dataset; see `src/data/typed_decisions.py`. |

## Design implications for the on-device family

1. **The architecture is already small relative to a generative LLM** — the "large" part is a single BERT-family encoder forward pass, not an autoregressive loop. This validates the project's core premise without further argument: shrinking Laya is a normal encoder-distillation problem, not a novel research question.
2. **Distillation target is well-defined**: teacher forward pass → `(logits over options, act_logits)` per question, both real-valued and directly comparable to a student's outputs at matching marker positions. Standard logit-distillation (KL/soft-CE against teacher softmax) is a direct, low-risk first experiment — no exotic technique needed to get a baseline.
3. **The shared `head_max_len` budget is worth fixing, not just shrinking.** Any student we build should make the option-token budget scale with option count (or adopt per-option budgets) rather than reproduce Banking77's 0.425 ceiling at a smaller size too.
4. **ModernBERT-base is a legitimate zero-cost starting point for the Base tier** — before running any custom distillation, we should benchmark it directly with a freshly-trained (not distilled) decision head on the same fine-tuning recipe, as the honest baseline every distilled student must beat.
5. **RLCD's zero-order policy gradient is architecture-agnostic** — it operates on logits over markers, so it ports to any encoder size unchanged; no student-specific loss redesign is needed for the head training itself.

## Explicitly unresolved (flagged, not guessed)

- Base-checkpoint pretraining data, schema mix, and compute budget: undisclosed upstream.
- mmBERT license terms: not yet independently verified. (`LocalLLaMA/typed-decisions` now verified Apache-2.0, see above.)
- Actual parameter breakdown of head vs. encoder per checkpoint: derived by subtraction from published totals, not from a component-level manifest (none is published).
- Real-hardware (phone/RPi) latency for any Laya checkpoint: not published upstream at all — T4 GPU only. Our on-device benchmark suite fills a genuine gap here, not just a smaller-model gap.

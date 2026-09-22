# Adapted from laya/common.py build_sequence (Convai Innovations, Apache-2.0) — see NOTICE.
#
# Deliberate deviation from upstream: upstream splits a *fixed* head_max_len
# between instructions and all options combined, so a 77-option question gets
# ~4 tokens/option and collapses (docs/LAYA_ANALYSIS.md documents this as
# Laya's own stated Banking77 failure, 0.425 vs Jev's 0.870 — architectural,
# not a capability gap). Here each option gets a fixed per-option token
# allowance (min_opt_tokens) and the head budget grows with option count
# instead of shrinking each option's share. This costs sequence length on
# high-cardinality questions, which is an acceptable trade for smaller,
# faster on-device models where extra context tokens are cheap.
"""Typed-question sequence builder: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]."""
import json
from typing import Dict, List, Optional, Union


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    min_instruction_tokens: int = 24,
    min_opt_tokens: int = 8,
    max_opt_tokens: int = 48,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
):
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    head_ids = head_ids[:max(min_instruction_tokens, 64)]

    opt_ids = []
    for i in order:
        opt_ids.append(
            [tok.mask_token_id]
            + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:max_opt_tokens]
        )
    # Fixed per-option floor instead of a shared shrinking budget: reallocate
    # only oversized options, never starve every option equally as count grows.
    for o in opt_ids:
        if len(o) > min_opt_tokens + 1:
            o[min_opt_tokens + 1:] = o[min_opt_tokens + 1:][: max(0, max_opt_tokens - min_opt_tokens)]

    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]

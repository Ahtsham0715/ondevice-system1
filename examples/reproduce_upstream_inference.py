"""Reproduce upstream Laya's README quickstart locally, on whatever hardware this runs on.

This is the "Reproduce upstream inference" step of the first milestone (ROADMAP.md) — it
imports the real `laya` package (pip install laya) and runs its own documented example, so the
timing and output printed here are real measurements on this machine, not upstream's cited T4
numbers. Record the printed latency in RESEARCH_LOG.md when you run this.

Run from the repo root: PYTHONPATH=. python examples/reproduce_upstream_inference.py
"""
import time

import src.compat  # noqa: F401 — must import before transformers/laya, see src/compat.py
import laya


def main():
    # torch 2.2.2 (the newest this environment's mirror serves) doesn't support
    # torch.autocast(device_type="mps") at all, which laya/agent.py hits unconditionally on any
    # MPS-capable device (this machine has an MPS-capable AMD GPU despite being Intel — MPS
    # isn't Apple-Silicon-only). Forcing CPU sidesteps that incompatibility and is also a real
    # on-device target in its own right (ROADMAP.md), not just a workaround. See RESEARCH_LOG.md
    # 2026-09-22.
    agent = laya.load("convaiinnovations/laya", device="cpu")

    state = {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts",
                "other": "everything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
        },
        "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    }

    # warm-up (excludes model/lazy-init cost from the timed run)
    agent.predict(state, questions)

    t0 = time.perf_counter()
    result = agent.predict(state, questions)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print("device:", agent.device)
    print("department:", result["answers"]["department"]["choice"], result["answers"]["department"]["confidence"])
    print("urgency score:", result["answers"]["urgency"]["score"])
    print("churn_risk:", result["answers"]["churn_risk"]["noul"])
    print("latency (this machine, warm): %.1f ms" % elapsed_ms)


if __name__ == "__main__":
    main()

"""Environment compatibility shims. Import this before `transformers` or `laya`.

Not upstream bugs — workarounds for this project's specific dev environment (see
RESEARCH_LOG.md 2026-09-22 for how each was diagnosed).
"""
import sys

import torch


def patch_torch_compile_on_py312():
    """transformers' ModernBERT applies `@torch.compile(dynamic=True)` at class-definition
    time. torch < 2.4 unconditionally raises on any torch.compile() call under Python 3.12+
    ("Dynamo is not supported on Python 3.12+"), so importing ModernBERT fails before Laya's
    own `reference_compile = False` override (laya/agent.py) ever runs — that override already
    disables compilation for inference, so a no-op decorator changes nothing at runtime.
    Inert on torch>=2.4 or Python<=3.11.
    """
    version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if sys.version_info >= (3, 12) and version < (2, 4):
        torch.compile = lambda model=None, *a, **kw: (model if model is not None else (lambda f: f))


patch_torch_compile_on_py312()

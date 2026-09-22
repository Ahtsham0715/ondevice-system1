# Security

This project distributes model weights and training code, not a network service — the main risks are: (1) a malicious checkpoint (pickle/unsafe deserialization), and (2) a training/eval script that executes untrusted data.

- Checkpoints are loaded via `safetensors`, not `torch.load`/pickle, precisely because safetensors can't execute code on load — keep this invariant in any new loading code.
- Dataset loaders should treat dataset content as data, not code — no `eval()`/`exec()` on dataset fields.

To report a vulnerability, open a private security advisory on this repository rather than a public issue.

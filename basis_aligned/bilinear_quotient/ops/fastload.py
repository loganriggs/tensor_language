"""fastload -- an opt-in, verified-identical, faster loader for the bilin18 checkpoint (ops lane, additive).

Measured 2026-09-04 06:23Z: `mlp_in_situ_usage_rank_map_probe.load_model()` costs ~3.0 s on CPU plus 0.5 s to
move to CUDA, against small-rung receipts whose whole runtime is 4.7-5.4 s -- so on those rungs roughly 88% of
the measured cost is loading the model. Broken down: **construction 2.05 s, torch.load 0.77 s, float copy
0.01 s, load_state_dict 0.14 s**. The 2.05 s is RANDOM INITIALISATION of 546M parameters that are immediately
overwritten (no-oping `torch.nn.init` drops construction to 0.08 s), and `torch.load(mmap=True)` makes the read
effectively free.

So `load_model_fast()` does exactly two things, and touches no weight:
  1. constructs the model with `torch.nn.init.*` temporarily no-oped -- allocation still happens on the REAL
     device, only the randomisation is skipped;
  2. `torch.load(..., mmap=True)` then `load_state_dict(..., assign=True)`.

WHY NOT `meta`: the first version of this module constructed under `torch.device("meta")`, which is faster
still and WRONG here. `jacclust.tt_model.Rotary` sets `self.inv_freq = 1.0 / (base ** ...)` as a PLAIN
ATTRIBUTE -- not a registered buffer -- so it is invisible to `state_dict()` AND to `named_buffers()`, stays a
meta tensor, and the first forward dies with "Cannot copy out of meta tensor". The original `verify_identical()`
compared state_dicts and therefore could not see the one tensor that was broken; the bug was caught only when a
rung tried to run. `verify_identical()` now compares parameters, buffers, plain-attribute tensors found by
walking every module's `__dict__`, AND the logits of an actual forward pass -- the last of which would have
caught it immediately.
"""
from __future__ import annotations

import contextlib
import json
import sys
import time

import torch


def _paths():
    sys.path.insert(0, "/workspace/tensor_language/basis_aligned/bilinear_quotient/ops")
    import mlp_in_situ_usage_rank_map_probe as R
    cfg = json.load(open(R.SNAP / "config.json"))
    cfg.pop("step", None)
    return cfg, R.BLOB, R


_INIT_FNS = ("normal_", "uniform_", "kaiming_uniform_", "kaiming_normal_",
             "xavier_uniform_", "xavier_normal_", "trunc_normal_")


@contextlib.contextmanager
def _no_random_init():
    """Skip the randomisation of weights that are about to be overwritten; keep the allocation."""
    import torch.nn.init as I
    saved = {n: getattr(I, n) for n in _INIT_FNS if hasattr(I, n)}
    for n in saved:
        setattr(I, n, lambda t, *a, **k: t)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(I, n, f)


def load_model_fast():
    """Same weights and same forward as `mlp_in_situ_usage_rank_map_probe.load_model()`, fewer copies."""
    cfg, blob, _R = _paths()
    sys.path.insert(0, "/workspace/tensor_language")
    import jacclust.tt_model as TT
    with _no_random_init():
        m = TT.GPT(TT.GPTConfig(**cfg)).float().eval()
    try:
        sd = torch.load(blob, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):          # older torch, or a non-zipfile checkpoint
        sd = torch.load(blob, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True, assign=True)
    m = m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _all_tensors(model):
    """Every tensor reachable: parameters, buffers, AND plain attributes set in __init__.

    The plain-attribute sweep exists because `Rotary.inv_freq` is one, and the first version of this
    module shipped a bug that only such a tensor could have.
    """
    out = {}
    for k, v in model.state_dict().items():
        out[f"state_dict:{k}"] = v
    for n, b in model.named_buffers():
        out[f"buffer:{n}"] = b
    for name, mod in model.named_modules():
        for attr, val in vars(mod).items():
            if isinstance(val, torch.Tensor):
                out[f"attr:{name}.{attr}"] = val
    return out


def verify_identical(verbose=True):
    """Bit-identical tensors AND bit-identical logits on a real forward. Returns (ok, n_tensors, max_logit_dev)."""
    _cfg, _blob, R = _paths()
    a = R.load_model()
    b = load_model_fast()
    ta, tb = _all_tensors(a), _all_tensors(b)
    if set(ta) != set(tb):
        if verbose:
            print("KEY MISMATCH", set(ta) ^ set(tb))
        return False, 0, float("nan")
    n = 0
    for k in ta:
        x, y = ta[k], tb[k]
        if x.dtype != y.dtype or x.shape != y.shape or x.device.type != y.device.type or not torch.equal(x, y):
            if verbose:
                print(f"MISMATCH {k}: {x.dtype}/{x.shape}/{x.device} vs {y.dtype}/{y.shape}/{y.device}")
            return False, n, float("nan")
        n += 1
    # the check the first version lacked: run both models and compare outputs
    tok = torch.arange(1, 65, dtype=torch.long).unsqueeze(0) % 50000
    with torch.no_grad():
        la = a(tok, tok)
        lb = b(tok, tok)
    dev = float((la - lb).abs().max()) if torch.is_tensor(la) else abs(float(la) - float(lb))
    if verbose:
        print(f"forward deviation: {dev:.3g} over {n} tensors")
    return dev == 0.0, n, dev


if __name__ == "__main__":
    _cfg, _blob, R = _paths()
    t = time.time(); R.load_model(); slow = time.time() - t
    t = time.time(); load_model_fast(); fast = time.time() - t
    ok, n, dev = verify_identical()
    print(f"existing load_model(): {slow:.2f}s")
    print(f"load_model_fast():     {fast:.2f}s   ({slow / max(fast, 1e-9):.1f}x)")
    print(f"identical: {ok}  tensors compared: {n}  max forward deviation: {dev:.3g}")

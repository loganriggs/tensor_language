"""fastload -- an opt-in, bit-identical, faster loader for the bilin18 checkpoint (ops lane, additive).

Measured 2026-09-04 06:23Z: `mlp_in_situ_usage_rank_map_probe.load_model()` costs 3.05 s on CPU plus 0.51 s to
move to CUDA, against receipts whose whole runtime is 4.7-5.4 s -- so on a small rung roughly 88% of the
measured cost is loading the model, and five consecutive rungs in one hour each paid it in full.

Three costs are avoidable without changing a single weight:
  1. `TT.GPT(cfg)` random-initialises 546M parameters that are about to be overwritten. Constructing on the
     `meta` device skips the initialisation and the allocation.
  2. `torch.load` reads the whole 2.07 GB file into RAM. `mmap=True` maps it instead (zipfile checkpoints only).
  3. `{k: v.float() for ...}` materialises a second full copy before `load_state_dict` copies again.
     `assign=True` binds the tensors directly.

`load_model_fast()` returns a model whose every parameter is BIT-IDENTICAL to `load_model()`'s -- verified by
`verify_identical()`, which this module's self-test runs. NOTHING adopts this automatically: it is a drop-in
alternative a rung may import, and the existing loader is untouched, so no registered script changes behaviour
unless its author edits it.
"""
from __future__ import annotations

import json
import sys
import time

import torch


def _cfg_and_paths():
    sys.path.insert(0, "/workspace/tensor_language/basis_aligned/bilinear_quotient/ops")
    import mlp_in_situ_usage_rank_map_probe as R
    cfg = json.load(open(R.SNAP / "config.json"))
    cfg.pop("step", None)
    return cfg, R.BLOB, R


def load_model_fast():
    """Same weights as `mlp_in_situ_usage_rank_map_probe.load_model()`, fewer copies."""
    cfg, blob, _R = _cfg_and_paths()
    sys.path.insert(0, "/workspace/tensor_language")
    import jacclust.tt_model as TT
    with torch.device("meta"):
        m = TT.GPT(TT.GPTConfig(**cfg))
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


def verify_identical(verbose=True):
    """Every parameter and buffer bit-identical to the existing loader. Returns (ok, n_tensors)."""
    _cfg, _blob, R = _cfg_and_paths()
    a = R.load_model()
    b = load_model_fast()
    da, db = dict(a.state_dict()), dict(b.state_dict())
    if set(da) != set(db):
        return False, 0
    n = 0
    for k in da:
        x, y = da[k], db[k]
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
            if verbose:
                print(f"MISMATCH {k}")
            return False, n
        n += 1
    return True, n


if __name__ == "__main__":
    _cfg, _blob, R = _cfg_and_paths()
    t = time.time(); R.load_model(); slow = time.time() - t
    t = time.time(); load_model_fast(); fast = time.time() - t
    ok, n = verify_identical()
    print(f"existing load_model(): {slow:.2f}s")
    print(f"load_model_fast():     {fast:.2f}s   ({slow / max(fast, 1e-9):.1f}x)")
    print(f"bit-identical: {ok} over {n} tensors")

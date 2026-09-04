"""frontier_fitcache -- persist the SS312 fitted stack so identical baseline fits are not recomputed. (ops lane, additive, opt-in.)

Written 2026-09-04 11:06Z from this hour's measurement.

WHAT THE LAST CHANGE BOUGHT. `ops/frontier_evalarms.py` (fit-once/eval-many, shipped 10:06Z) had its first full hour: nine multi-arm
rungs evaluated **117 arms in 1,344 GPU-seconds**, against ~11,010 under the per-arm pattern they replaced -- **9,666 GPU-seconds saved,
88%**. That is 14x the 700 GPU-s I projected, because the rungs written since have averaged 13 arms rather than 4.

WHAT IS LEFT. With arms nearly free, the cost is the FIT. A 15-arm run took 142.1 s, of which ~90 s is fitting and ~3.5 s per
evaluation. Every rung refits the SAME unmodified stack for its baseline: **15 rungs x ~90 s = ~1,350 GPU-seconds, 44% of the hour,
recomputing an identical object.**

WHAT THIS DOES. `save_stack` / `load_stack` persist the fitted `S`, `cfgF` and `order2` keyed by the model blob and the fit-relevant
configuration, so a rung whose fit phase is unmodified can load in seconds instead of refitting. SS2876 measured the pipeline as
deterministic to four decimals, so a cached stack must reproduce exactly -- and every rung in this family already carries pred_a, the
SS2125 reproduction gate, which is a live verifier of precisely that.

WHY THE VERIFIER IS RECURSIVE, AND WHY THAT MATTERS. `S` maps a key to a TUPLE whose members include dicts of tensors (`LW`), so a
comparison that only walks the top level is structurally incapable of seeing a changed link map. That is exactly how `ops/fastload.py`
shipped broken on 2026-09-04 06:24: its verifier compared state_dicts, reported "bit-identical over 218 tensors", and could not see a
plain attribute left on the meta device. `verify_stack` walks tuples, lists and dicts to arbitrary depth, counts every tensor it
compares, and `test_frontier_fitcache.py` asserts it CATCHES a difference planted three levels down.

NOTHING ADOPTS THIS AUTOMATICALLY. No landed rung is modified -- their bytes are cited by ledger sections and their receipts must stay
reproducible -- and `ops/frontier_fisher8.py` (SS2125 rung 30) is untouched.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

CACHE_DIR = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/.fitcache")


def stack_key(*parts) -> str:
    """A cache key identical across processes (blake2b, not the salted builtin hash)."""
    return hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=16).hexdigest()


def cache_path(key: str) -> Path:
    return CACHE_DIR / f"stack_{key}.pt"


def save_stack(key: str, S, cfgF, order2) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(key)
    torch.save({"S": S, "cfgF": list(cfgF), "order2": list(order2)}, p)
    return p


def load_stack(key: str, device=None):
    """Return (S, cfgF, order2) or None when the key is absent.

    `device=None` keeps everything on CPU with mmap, which is what the CPU-side certificate analysis wants
    and is why the cache was written this way. Pass `device="cuda"` (or a torch.device) when the stack is
    to be installed back into a live model: SS2911 died with "indices should be either on cpu or on the
    same device as the indexed tensor" because a CPU token table was indexed by a CUDA index. The mmap
    path is skipped when a device is requested, since mmap and a device copy do not compose usefully.
    """
    p = cache_path(key)
    if not p.is_file():
        return None
    if device is None:
        d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        return d["S"], d["cfgF"], d["order2"]
    d = torch.load(p, map_location=device, weights_only=False)
    return _to_device(d["S"], device), d["cfgF"], d["order2"]


def _to_device(obj, device):
    """Move every tensor reachable through tuples, lists and dicts -- the same recursion verify_stack uses."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _walk(obj, path="", out=None):
    """Yield (path, tensor) for every tensor reachable through tuples, lists and dicts."""
    if out is None:
        out = []
    if torch.is_tensor(obj):
        out.append((path, obj))
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _walk(obj[k], f"{path}[{k!r}]", out)
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            _walk(v, f"{path}[{i}]", out)
    return out


def verify_stack(a, b):
    """Compare two stacks tensor-by-tensor at arbitrary depth.

    Returns (ok, n_compared, max_abs_deviation, first_mismatch_path). A structural difference -- a key or
    length present in one and not the other -- is a mismatch, not a skipped comparison.
    """
    ta, tb = _walk(a), _walk(b)
    if len(ta) != len(tb):
        return False, min(len(ta), len(tb)), float("inf"), f"tensor count {len(ta)} vs {len(tb)}"
    worst, where = 0.0, None
    for (pa, xa), (pb, xb) in zip(ta, tb):
        if pa != pb:
            return False, len(ta), float("inf"), f"path {pa} vs {pb}"
        if xa.shape != xb.shape or xa.dtype != xb.dtype:
            return False, len(ta), float("inf"), pa
        # SS2913: the whole point of load_stack(device=) is that the two sides can live on DIFFERENT devices --
        # a CPU reference against a stack reloaded onto CUDA. Compare on CPU rather than crashing on the subtraction.
        d = float((xa.float().cpu() - xb.float().cpu()).abs().max()) if xa.numel() else 0.0
        if d > worst:
            worst, where = d, pa
    return worst == 0.0, len(ta), worst, where


def expected_saving(n_rungs, fit_seconds=90.0, load_seconds=5.0):
    """GPU-seconds saved per hour when `n_rungs` share one cached baseline fit."""
    return round(n_rungs * (fit_seconds - load_seconds), 1)

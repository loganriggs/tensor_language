"""Memoized context-covariance loader (ops helper, 2026-09-01).

Many rungs recompute identical MLP-input covariances from the same frozen
fit-cache rows (72 import sites of the covariance/logits helpers).  This
helper caches them to .covcache/ keyed by (cache file, row range, layers),
saving a model forward pass (~10-30s GPU) per reuse.  Purely additive: it
calls the SAME M._covariances/_manual_logits code on a miss, so numbers are
bit-identical to the uncached path.  Adopt in new scripts via:

    from covcache import cached_covariances
    cov = cached_covariances(model, "fineweb_n192_skip11000.pt", (24, 48), layers)
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
CACHE_DIR = ROOT / ".covcache"
CACHE_DIR.mkdir(exist_ok=True)


def cached_covariances(model, cache_name: str, rows_half_open: tuple, layers):
    import sys
    sys.path.insert(0, str(ROOT / "ops"))
    import mlp_all_layer_context_metric_shared_input_screen as M
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    key = f"{cache_name}|{rows_half_open[0]}:{rows_half_open[1]}|{','.join(map(str, sorted(layers)))}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    path = CACHE_DIR / f"cov_{digest}.pt"
    if path.exists():
        blob = torch.load(path, map_location="cpu")
        if blob.get("key") == key:
            return {layer: blob["cov"][str(layer)] for layer in layers}
    cached = torch.load(ROOT / f".rowcache/{cache_name}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    rows = cached[rows_half_open[0]:rows_half_open[1], :257].long().contiguous()
    M.LAYERS = tuple(sorted(layers))
    cov = M._covariances(model, rows, _manual_logits)
    torch.save({"key": key, "cov": {str(l): cov[l].cpu() for l in layers}}, path)
    return cov

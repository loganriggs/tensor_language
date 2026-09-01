"""Canonical scoring path (ops helper, 2026-09-01).

Five of this hour's eleven landings were RERUNS caused by scoring-path
mismatches (405 ran 3x: loss-aggregation order, then a missing float32
logit cast; 409 ran 2x: closure constant).  This module freezes THE
convention that rungs 403/404 established and 405's third run matched:

  1. softcapped logits are cast to float32 BEFORE cross-entropy
     (the parent facade's explicit cast);
  2. per-document mean CE is computed in float32 over the scored
     positions of that document;
  3. document means are pooled in float64.

New scripts should call these helpers instead of re-deriving the path.
Purely additive: existing registered scripts are untouched.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def document_mean_ce(logits: torch.Tensor, targets: torch.Tensor,
                     scoring: slice | None = None) -> torch.Tensor:
    """Float32 mean CE for ONE document's scored positions.
    logits: [T, V] (any float dtype; cast to float32 here, matching the
    parent facade), targets: [T] long."""
    if scoring is not None:
        logits = logits[scoring]
        targets = targets[scoring]
    return F.cross_entropy(logits.float(), targets, reduction="mean")


def pool_document_ces(doc_means) -> float:
    """Float64 pool of per-document float32 means (the 403/404 order)."""
    t = torch.stack([m.detach().double().cpu() if torch.is_tensor(m)
                     else torch.tensor(float(m), dtype=torch.float64) for m in doc_means])
    return float(t.mean())

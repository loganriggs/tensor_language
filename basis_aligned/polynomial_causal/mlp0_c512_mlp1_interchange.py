"""Physical replay primitives for the frozen C512/MLP1 interchange assay.

This module contains no row selection, inference thresholds, or authority mutation.
The evaluator installs either native MLP0 Down or the frozen C512 proxy before
calling :func:`capture_through_mlp1`.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F


BACKGROUNDS = ("live", "mlp2_omit")
PHYSICAL_ARMS = ("OO", "CC", "CO", "OC")


@torch.no_grad()
def capture_through_mlp1(model: Any, blocks: Sequence[Any], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run embeddings and blocks 0--1, returning the physical MLP1 interface.

    `s` is the residual after block-1 attention and before block-1 MLP; `m` is
    the physical MLP1 write. `v1` is the carried block-0 value state needed by
    all later attention layers. `pre_mlp0` supports the already-frozen cell map.
    """
    if len(blocks) < 3 or idx.ndim != 2:
        raise ValueError("the interchange requires token rows and at least three blocks")
    d_model = int(model.transformer.wte.weight.shape[1])
    x = F.rms_norm(model.transformer.wte(idx), (d_model,))
    x0 = x
    v1 = None

    block0 = blocks[0]
    x = block0.lambdas[0] * x + block0.lambdas[1] * x0
    attn0, v1 = block0.attn(F.rms_norm(x, (d_model,)), v1)
    pre_mlp0 = x + attn0
    m0 = block0.mlp(F.rms_norm(pre_mlp0, (d_model,)))
    x = pre_mlp0 + m0

    block1 = blocks[1]
    x = block1.lambdas[0] * x + block1.lambdas[1] * x0
    attn1, v1 = block1.attn(F.rms_norm(x, (d_model,)), v1)
    s = x + attn1
    m = block1.mlp(F.rms_norm(s, (d_model,)))
    return {
        "x0": x0,
        "v1": v1,
        "pre_mlp0": pre_mlp0,
        "m0": m0,
        "attn1": attn1,
        "s": s,
        "m": m,
        "post": s + m,
    }


@torch.no_grad()
def suffix_forward(
    model: Any,
    blocks: Sequence[Any],
    post_mlp1: torch.Tensor,
    v1: torch.Tensor,
    x0: torch.Tensor,
    *,
    background: str = "live",
) -> torch.Tensor:
    """Replay blocks 2..end and the readout from a physical post-MLP1 state."""
    if background not in BACKGROUNDS:
        raise ValueError(f"unknown suffix background: {background}")
    if post_mlp1.shape != x0.shape:
        raise ValueError("post-MLP1 state and embedding residual have different shapes")
    d_model = int(post_mlp1.shape[-1])
    x = post_mlp1
    carried = v1
    for layer, block in enumerate(blocks[2:], start=2):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attn, carried = block.attn(F.rms_norm(x, (d_model,)), carried)
        x = x + attn
        write = block.mlp(F.rms_norm(x, (d_model,)))
        if background == "mlp2_omit" and layer == 2:
            write = torch.zeros_like(write)
        x = x + write
    return (30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (d_model,))) / 30.0)).float()


def physical_post_states(
    exact: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Construct the registered O/C state-by-write factorial."""
    for key in ("s", "m"):
        if exact[key].shape != candidate[key].shape:
            raise ValueError(f"exact and candidate {key} shapes differ")
    return {
        "OO": exact["s"] + exact["m"],
        "CC": candidate["s"] + candidate["m"],
        "CO": candidate["s"] + exact["m"],
        "OC": exact["s"] + candidate["m"],
    }


def norm_matched_native_write(delta_m: torch.Tensor, native_m: torch.Tensor) -> torch.Tensor:
    """Scale the native MLP1 write at each position to `||delta_m||`."""
    if delta_m.shape != native_m.shape:
        raise ValueError("write tensors have different shapes")
    target = delta_m.float().norm(dim=-1, keepdim=True)
    source = native_m.float().norm(dim=-1, keepdim=True)
    scale = torch.where(source > 0, target / source, torch.zeros_like(source))
    return (native_m.float() * scale).to(native_m.dtype)


def document_derangement(document: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Return a within-cell vector permutation with no same-document donors.

    Within each cell, positions are grouped by source document and circularly
    shifted by the largest document occupancy. If no document owns more than half
    the cell, this is an exact multiset permutation with no same-document edge.
    """
    document = document.detach().long().cpu().flatten()
    cell = cell.detach().long().cpu().flatten()
    if document.shape != cell.shape or document.numel() == 0:
        raise ValueError("document and cell labels must be nonempty and aligned")
    permutation = torch.empty(document.numel(), dtype=torch.long)
    for cell_id in torch.unique(cell, sorted=True).tolist():
        positions = torch.where(cell == cell_id)[0]
        labels = document[positions]
        order = torch.argsort(labels, stable=True)
        grouped_positions = positions[order]
        grouped_labels = labels[order]
        _, counts = torch.unique_consecutive(grouped_labels, return_counts=True)
        largest = int(counts.max())
        if largest * 2 > len(positions):
            raise RuntimeError(f"cell {cell_id} cannot be deranged across documents")
        donors = torch.roll(grouped_positions, shifts=-largest)
        permutation[grouped_positions] = donors
    if not torch.equal(cell, cell[permutation]):
        raise RuntimeError("derangement crossed a registered cell")
    if bool((document == document[permutation]).any()):
        raise RuntimeError("derangement retained a source document")
    if torch.unique(permutation).numel() != permutation.numel():
        raise RuntimeError("derangement is not a vector-multiset permutation")
    return permutation


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


def additive_interaction_prediction(logits: dict[str, torch.Tensor]) -> torch.Tensor:
    """Gauge-fix the no-interaction CC logit prediction CO + OC - OO."""
    if not all(key in logits for key in PHYSICAL_ARMS):
        raise ValueError("physical arm logits are incomplete")
    return centered_logits(logits["CO"] + logits["OC"] - logits["OO"])

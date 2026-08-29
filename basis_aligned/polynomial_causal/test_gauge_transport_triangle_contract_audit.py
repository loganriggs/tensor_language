"""Focused pre-execution contract checks for the transport triangle.

These tests use only synthetic CPU tensors.  They do not load the checkpoint,
FineWeb rows, or any outcome artifact.  Their purpose is to distinguish the
finite held-out commuting triangle from the earlier infinitesimal response-rank
screen while preserving the current fail-closed row-provenance requirement.
"""

from __future__ import annotations

import pytest
import torch

import gauge_transport_triangle as triangle


def _synthetic_receipt(*, duplicate_within_basis: bool = False) -> dict:
    sets = {}
    for n, skip in (triangle.BASIS_SPEC, triangle.FIT_SPEC, triangle.EVAL_SPEC):
        rows = [
            {
                "document_id": f"document-{skip}-{index}",
                "chunk_id": 0,
            }
            for index in range(n)
        ]
        sets[f"n{n}_skip{skip}"] = rows
    if duplicate_within_basis:
        key = f"n{triangle.BASIS_SPEC[0]}_skip{triangle.BASIS_SPEC[1]}"
        sets[key][1]["document_id"] = sets[key][0]["document_id"]
        sets[key][1]["chunk_id"] = 1
    return {"document_provenance": {"schema_version": 1, "sets": sets}}


def test_headline_rejects_two_chunks_from_one_document_within_a_split():
    triangle.require_document_disjoint_receipt(_synthetic_receipt())
    with pytest.raises(RuntimeError, match="one sequence per document"):
        triangle.require_document_disjoint_receipt(
            _synthetic_receipt(duplicate_within_basis=True)
        )


class _CPUModel:
    def __init__(self) -> None:
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def parameters(self):
        yield self.anchor


def _finite_fake_forward(
    _model,
    idx: torch.Tensor,
    *,
    patch_layer: int | None = None,
    patch_delta: torch.Tensor | None = None,
    capture_sites: tuple[int, ...] = (),
    return_logits: bool = True,
):
    """A finite nonlinear-output harness with deliberately poisoned true L11.

    A source edit has the same physical response at L14 regardless of whether it
    is installed at L8 or L14.  The true L11 response is seven times larger, so a
    chain that secretly consumes it cannot accidentally pass with identity maps.
    """

    base = torch.stack((idx.float(), 0.5 * idx.float()), dim=-1)
    response = torch.zeros_like(base) if patch_delta is None else patch_delta.float()
    captures = {}
    for site in capture_sites:
        multiplier = 7.0 if site == 11 and patch_layer == 8 else 1.0
        live = patch_layer is not None and patch_layer <= site
        captures[site] = base + (multiplier * response if live else 0.0)
    if not return_logits:
        return None, None, captures

    # Later computation makes a single finite edit affect its complete causal
    # suffix.  This is deliberately not a Jacobian or infinitesimal response.
    propagated = response.cumsum(dim=1)
    raw = torch.stack((idx.float(), -idx.float()), dim=-1) + propagated
    return raw, raw, captures


def _run_synthetic_triangle(monkeypatch, *, break_chain: bool = False):
    production_sparse_delta = triangle.sparse_physical_delta
    monkeypatch.setattr(triangle, "D", 2)
    monkeypatch.setattr(triangle, "K", 2)
    monkeypatch.setattr(triangle, "SUPPORT_RANK", 2)
    monkeypatch.setattr(triangle, "SEQ", 5)
    monkeypatch.setattr(triangle, "MIN_POSITION", 1)
    monkeypatch.setattr(triangle, "BATCH", 2)
    monkeypatch.setattr(triangle, "native_forward", _finite_fake_forward)
    monkeypatch.setattr(
        triangle,
        "sparse_physical_delta",
        lambda coordinates, basis, positions, length=5: production_sparse_delta(
            coordinates, basis, positions, length=5
        ),
    )

    # The first half is the donor family and the second half is the untouched
    # target family, matching the production evaluator's sealed half split.
    rows = torch.tensor(
        [
            [4, 4, 4, 4, 4, 0],
            [5, 5, 5, 5, 5, 0],
            [1, 1, 1, 1, 1, 0],
            [2, 2, 2, 2, 2, 0],
        ],
        dtype=torch.long,
    )
    identity = torch.eye(2)
    maps = {
        "8_11": torch.zeros(2, 2) if break_chain else identity.clone(),
        "8_14": identity.clone(),
        "11_14": identity.clone(),
    }
    return triangle.evaluate_triangle(
        _CPUModel(),
        rows,
        bases={8: identity, 11: identity, 14: identity},
        supports={8: identity, 11: identity, 14: identity},
        maps=maps,
        amplitude=0.4,
    )


def test_finite_heldout_chain_uses_maps_not_true_intermediate(monkeypatch):
    result = _run_synthetic_triangle(monkeypatch)
    assert result["direct"]["coordinate_response_r2"] == pytest.approx(1.0)
    assert result["chain"]["coordinate_response_r2"] == pytest.approx(1.0)
    assert result["direct"]["e_out"] == pytest.approx(0.0, abs=1e-12)
    assert result["chain"]["e_out"] == pytest.approx(0.0, abs=1e-12)


def test_broken_first_map_fails_chain_without_harming_direct(monkeypatch):
    result = _run_synthetic_triangle(monkeypatch, break_chain=True)
    assert result["direct"]["coordinate_response_r2"] == pytest.approx(1.0)
    assert result["direct"]["e_out"] == pytest.approx(0.0, abs=1e-12)
    assert result["chain"]["coordinate_response_r2"] == pytest.approx(0.0)
    assert result["chain"]["e_out"] == pytest.approx(1.0)

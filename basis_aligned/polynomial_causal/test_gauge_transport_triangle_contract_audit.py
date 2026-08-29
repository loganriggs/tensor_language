"""Focused pre-execution contract checks for the transport triangle.

These tests use only synthetic CPU tensors.  They do not load the checkpoint,
FineWeb rows, or any outcome artifact.  Their purpose is to distinguish the
finite held-out commuting triangle from the earlier infinitesimal response-rank
screen while preserving the current fail-closed row-provenance requirement.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

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


def test_completed_v2_row_metadata_is_exact_and_non_authorizing_without_tensor_load(
    monkeypatch,
):
    monkeypatch.setattr(
        torch, "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata audit must not deserialize row tensors")
        ),
    )
    authority = triangle.load_pinned_json(
        triangle.ROW_AUTHORITY, triangle.ROW_AUTHORITY_FILE_SHA256, "row authority",
    )
    manifest = triangle.load_pinned_json(
        triangle.ROW_MANIFEST, triangle.ROW_MANIFEST_FILE_SHA256, "row manifest",
    )
    receipt = triangle.load_pinned_json(
        triangle.ROW_RECEIPT, triangle.ROW_RECEIPT_FILE_SHA256, "row receipt",
    )
    triangle.validate_v2_row_metadata(authority, manifest, receipt)
    assert manifest["role_sizes"] == {"basis": 96, "fit": 96, "evaluation": 192}
    assert receipt["unique_document_count"] == 384
    assert receipt["triangle_runner_authorized_by_this_receipt"] is False


def test_main_is_fail_closed_without_execution_authority(monkeypatch, tmp_path):
    monkeypatch.setattr(triangle, "RUN_AUTHORITY", tmp_path / "absent.json")
    monkeypatch.setattr(
        triangle, "load_unique_v2_rows",
        lambda: (_ for _ in ()).throw(AssertionError("row loader must remain unopened")),
    )
    with pytest.raises(RuntimeError, match="execution authority is absent"):
        triangle.main()


def test_execution_authority_binds_sources_inputs_and_terminals(monkeypatch):
    source = "synthetic/source.py"
    monkeypatch.setattr(triangle, "RUN_SOURCE_FILES", (source,))
    monkeypatch.setattr(triangle, "file_sha256", lambda _path: "a" * 64)
    body = {
        "schema": "gauge_transport_triangle_v2_recovery_authority",
        "status": "source_closed_go",
        "source_commit": "1" * 40,
        "source_files": [source],
        "source_sha256s": {source: "a" * 64},
        "row_artifact_file_sha256": triangle.ROW_ARTIFACT_FILE_SHA256,
        "row_receipt_file_sha256": triangle.ROW_RECEIPT_FILE_SHA256,
        "model_weights_sha256": "b" * 64,
        "terminal_outputs": {
            "result": triangle.OUT.name,
            "state": triangle.STATE_OUT.name,
            "receipt": triangle.RUN_RECEIPT.name,
            "failure": triangle.RUN_FAILURE.name,
        },
    }
    authority = {**body, "authority_sha256": triangle.canonical_sha256(body)}
    triangle.validate_execution_authority(authority)
    authority["source_sha256s"][source] = "c" * 64
    authority["authority_sha256"] = triangle.canonical_sha256({
        key: value for key, value in authority.items() if key != "authority_sha256"
    })
    with pytest.raises(RuntimeError, match="source changed"):
        triangle.validate_execution_authority(authority)


def test_create_only_json_refuses_overwrite(tmp_path):
    path = tmp_path / "terminal.json"
    triangle.create_only_json(path, {"status": "first"})
    with pytest.raises(FileExistsError):
        triangle.create_only_json(path, {"status": "second"})


def _synthetic_v2_payload_contract(monkeypatch):
    monkeypatch.setattr(triangle, "ROLE_SIZES", {"basis": 1, "fit": 1, "evaluation": 1})
    monkeypatch.setattr(triangle, "ROW_TOKEN_LENGTH", 3)
    records = {
        role: [{"document_id": f"{role}-document", "role_index": 0}]
        for role in triangle.ROLE_SIZES
    }
    roles = {
        role: torch.tensor([[index, index + 1, index + 2]], dtype=torch.long)
        for index, role in enumerate(triangle.ROLE_SIZES)
    }
    authority = {"selection_plan": {"roles": copy.deepcopy(records)}}
    manifest = {
        "role_record_sha256s": {
            role: triangle.canonical_sha256(value) for role, value in records.items()
        },
    }
    receipt = {
        "role_tensor_composite_sha256s": {
            role: triangle.composite_tensor_sha256(value) for role, value in roles.items()
        },
    }
    payload = {
        "schema": "gauge_transport_triangle_unique_rows_v2_rows",
        "authority_sha256": triangle.ROW_AUTHORITY_SHA256,
        "selection_plan_sha256": triangle.ROW_SELECTION_PLAN_SHA256,
        "roles": roles,
        "records": records,
    }
    return authority, manifest, receipt, payload


def test_v2_payload_requires_exact_roles_records_hashes_and_global_document_disjointness(
    monkeypatch,
):
    authority, manifest, receipt, payload = _synthetic_v2_payload_contract(monkeypatch)
    rows = triangle.validate_v2_row_payload(payload, authority, manifest, receipt)
    assert set(rows) == {"basis", "fit", "evaluation"}

    changed = copy.deepcopy(payload)
    changed["records"]["evaluation"][0]["document_id"] = "basis-document"
    authority["selection_plan"]["roles"]["evaluation"] = copy.deepcopy(
        changed["records"]["evaluation"]
    )
    manifest["role_record_sha256s"]["evaluation"] = triangle.canonical_sha256(
        changed["records"]["evaluation"]
    )
    with pytest.raises(RuntimeError, match="globally unique"):
        triangle.validate_v2_row_payload(changed, authority, manifest, receipt)


def test_v2_pinned_json_detects_mutation_during_read(monkeypatch, tmp_path: Path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"status": "complete"}))
    expected = triangle.file_sha256(path)
    hashes = iter((expected, "0" * 64))
    monkeypatch.setattr(triangle, "file_sha256", lambda _path: next(hashes))
    with pytest.raises(RuntimeError, match="changed during pinned read"):
        triangle.load_pinned_json(path, expected, "synthetic receipt")

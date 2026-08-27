from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import prepare_affine_compiler_rows_v1 as prep


def _record(document: str, index: int) -> dict[str, object]:
    return {
        "document_id": document,
        "dataset_document_index": index,
        "chunk_id": 0,
        "token_start": 0,
    }


def _tiny_inputs():
    tensors = {}
    provenance = {}
    for offset, spec in enumerate(prep.SPECS):
        n, _ = spec
        rows = torch.arange(n * prep.T_LEN, dtype=torch.long).view(n, prep.T_LEN)
        rows = rows + offset * 10_000_000
        tensors[spec] = rows
        provenance[spec] = [_record(f"new-{offset}-{row}", row) for row in range(n)]
    return tensors, provenance


def test_registered_roles_are_fresh_and_ordered() -> None:
    assert prep.ROLE_SPECS == {
        "compiler_fit": (480, 15000),
        "compiler_validation": (192, 19000),
        "compiler_final": (192, 23000),
    }
    assert max(spec[1] for spec in prep.SPECS) == 23000
    prereg = json.loads(prep.PREREG.read_text())
    assert prereg["fresh_rows"]["compiler_final"] == {"n": 192, "skip": 23000}
    assert prereg["training_authority"]["base_model_updates"] is False
    assert prereg["training_authority"]["training_license_sites"] == [0, 1]


def test_disjointness_accepts_new_document_complete_roles() -> None:
    tensors, provenance = _tiny_inputs()
    summary = prep.validate_disjointness(tensors, provenance, {"old-0"})
    assert set(summary) == set(prep.ROLE_SPECS)
    assert all(row["prior_oracle_document_overlap"] == 0 for row in summary.values())


def test_disjointness_rejects_prior_document_overlap() -> None:
    tensors, provenance = _tiny_inputs()
    provenance[prep.SPECS[0]][0]["document_id"] = "old-0"
    with pytest.raises(RuntimeError, match="prior oracle documents"):
        prep.validate_disjointness(tensors, provenance, {"old-0"})


def test_disjointness_rejects_cross_role_document_leakage() -> None:
    tensors, provenance = _tiny_inputs()
    provenance[prep.SPECS[1]][0]["document_id"] = provenance[prep.SPECS[0]][0][
        "document_id"
    ]
    with pytest.raises(RuntimeError, match="document leakage"):
        prep.validate_disjointness(tensors, provenance, set())


def test_load_validator_requires_training_scope(monkeypatch, tmp_path: Path) -> None:
    receipt = {
        "status": "frozen_before_predictor_fit",
        "authority": "isolated_compiler_experiment",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": False,
        "training_license_sites": [],
        "preregistration_sha256": prep.PREREG_SHA256,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(prep, "RECEIPT", path)
    with pytest.raises(RuntimeError, match="authorized_for_training"):
        prep.load_and_validate()

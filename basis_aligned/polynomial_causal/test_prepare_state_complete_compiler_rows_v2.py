from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import prepare_state_complete_compiler_rows_v2 as prep


def _record(document: str, index: int) -> dict[str, object]:
    return {"document_id": document, "dataset_document_index": index,
            "chunk_id": 0, "token_start": 0}


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


def test_registered_roles_match_preregistration() -> None:
    assert prep.ROLE_SPECS == {
        "compiler_fit": (480, 27000),
        "compiler_validation": (192, 31000),
        "compiler_final": (192, 35000),
    }
    prereg = json.loads(prep.PREREG.read_text())
    assert prereg["fresh_rows"]["compiler_final"] == {"n": 192, "skip": 35000}
    assert prep.PREREG_SHA256 == prep.file_sha256(prep.PREREG)


def test_disjointness_accepts_fully_fresh_roles() -> None:
    tensors, provenance = _tiny_inputs()
    summary = prep.validate_disjointness(tensors, provenance,
                                         ({"old-doc"}, {"old-row"}, {(1, 2, 3)}))
    assert set(summary) == set(prep.ROLE_SPECS)
    assert all(row["prior_prefix32_overlap"] == 0 for row in summary.values())


def test_disjointness_rejects_each_prior_currency() -> None:
    tensors, provenance = _tiny_inputs()
    first = prep.SPECS[0]
    provenance[first][0]["document_id"] = "old-doc"
    with pytest.raises(RuntimeError, match="prior documents"):
        prep.validate_disjointness(tensors, provenance,
                                   ({"old-doc"}, set(), set()))

    tensors, provenance = _tiny_inputs()
    row_hash = prep.tensor_sha256(tensors[first][0])
    with pytest.raises(RuntimeError, match="prior full rows"):
        prep.validate_disjointness(tensors, provenance,
                                   (set(), {row_hash}, set()))

    prefix = tuple(tensors[first][0, :32].tolist())
    with pytest.raises(RuntimeError, match="prior prefix-32"):
        prep.validate_disjointness(tensors, provenance,
                                   (set(), set(), {prefix}))


def test_disjointness_rejects_cross_role_leakage() -> None:
    tensors, provenance = _tiny_inputs()
    provenance[prep.SPECS[1]][0]["document_id"] = provenance[prep.SPECS[0]][0][
        "document_id"
    ]
    with pytest.raises(RuntimeError, match="document leakage"):
        prep.validate_disjointness(tensors, provenance, (set(), set(), set()))


def test_load_validator_requires_prelabel_status(monkeypatch, tmp_path: Path) -> None:
    receipt = {
        "status": "wrong",
        "authority": "isolated_state_complete_compiler_v2",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": True,
        "training_license_sites": [0, 1],
        "preregistration_sha256": prep.PREREG_SHA256,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(prep, "RECEIPT", path)
    with pytest.raises(RuntimeError, match="status"):
        prep.load_and_validate()


def test_role_loader_byte_validates_but_never_deserializes_final(
    monkeypatch, tmp_path: Path,
) -> None:
    specs = {"compiler_fit": (2, 1), "compiler_validation": (2, 2),
             "compiler_final": (2, 3)}
    monkeypatch.setattr(prep, "ROLE_SPECS", specs)
    monkeypatch.setattr(prep, "require_pinned_sources", lambda: ({}, []))
    entries = {}
    provenance = {}
    file_hashes = {}
    paths = {}
    for offset, (role, spec) in enumerate(specs.items()):
        tensor = torch.arange(2 * prep.T_LEN, dtype=torch.long).view(2, prep.T_LEN)
        tensor += offset * 100_000
        path = tmp_path / f"{role}.pt"
        torch.save(tensor, path)
        records = [_record(f"{role}-{row}", row) for row in range(2)]
        paths[role] = path
        file_hashes[role] = prep.file_sha256(path)
        entries[role] = {
            "request": {"n": spec[0], "skip": spec[1]},
            "cache_path": str(path),
            "tensor_full_raw_sha256": prep.tensor_sha256(tensor),
            "tensor_prefix257_raw_sha256": prep.tensor_sha256(tensor[:, :257]),
            "provenance_records_sha256": prep.logical_json_sha256(records),
        }
        provenance[role] = records
    receipt = {
        "status": "frozen_before_any_label_or_gradient_capture",
        "authority": "isolated_state_complete_compiler_v2",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": True,
        "training_license_sites": [0, 1],
        "preregistration_sha256": prep.PREREG_SHA256,
        "entries": entries,
        "document_provenance": {"sets": provenance},
        "disjointness_gates": {"all": True},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(prep, "RECEIPT", receipt_path)
    monkeypatch.setattr(prep, "CACHE_FILE_SHA256", file_hashes)
    original_load = torch.load
    loaded = []

    def observed_load(path, *args, **kwargs):
        loaded.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", observed_load)
    _, rows = prep.load_roles_and_validate(("compiler_fit", "compiler_validation"))
    assert set(rows) == {"compiler_fit", "compiler_validation"}
    assert paths["compiler_final"] not in loaded

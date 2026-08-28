from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

import tensor_bilin18_tangent_pilot as pilot
import tensor_bilin18_tangent_authority as authority
from tensor_bilin18_tangent_collector import (
    WriteCovarianceGeometry, WriteGeometryBank, _json_sha256, _tensor_sha256,
)
from test_tensor_bilin18_program import tiny_program


def test_pilot_binds_admitted_program_plan_parents_and_create_only_result() -> None:
    source = Path(pilot.__file__).read_text()
    assert pilot.RANK == 640
    assert pilot.CUTS == (1, 2, 3)
    assert pilot.EXPECTED_PLAN_FINGERPRINT in source
    for fragment in (
        "rank640_parent", "causal_parent", "516_707_766", "os.O_EXCL",
        "collect_write_geometry_bank", "TensorBilin18TangentTransaction",
        "target hash ledger", "compare_split_cuts", "consequence_stage_authorized",
    ):
        if fragment == "os.O_EXCL":
            assert fragment in Path(authority.__file__).read_text()
        else:
            assert fragment in source


def test_stage1_requires_every_cut_not_a_favorable_subset() -> None:
    passing = {str(cut): {"passes": True} for cut in pilot.CUTS}
    assert pilot.stage1_passes(passing)
    passing["2"]["passes"] = False
    assert not pilot.stage1_passes(passing)
    assert not pilot.stage1_passes({"1": {"passes": True}})


def test_source_manifest_covers_every_new_scientific_boundary() -> None:
    names = {path.name for path in pilot.SOURCES}
    assert {
        "FINITE_HORIZON_TANGENT_REALIZATION_PREREGISTRATION.md",
        "finite_horizon_tangent_plan.json",
        "tensor_bilin18_tangent_collector.py",
        "tensor_bilin18_tangent_authority.py",
        "finite_horizon_tangent_response_bank.py",
        "finite_horizon_tangent_realization.py",
        "tensor_bilin18_program.py",
        "test_tensor_bilin18_tangent_pilot.py",
        "test_tensor_bilin18_tangent_collector.py",
    } <= names


def test_program_manifest_is_content_sensitive_and_canonical() -> None:
    program = tiny_program()
    first = authority.program_buffer_manifest(program, chunk_bytes=7)
    second = authority.program_buffer_manifest(program, chunk_bytes=11)
    assert first == second
    program.unembedding[0, 0] += 1
    changed = authority.program_buffer_manifest(program, chunk_bytes=7)
    assert changed["tree_sha256"] != first["tree_sha256"]


def test_program_receipt_must_equal_admitted_parent() -> None:
    parent = json.loads(authority.RANK640_PARENT.read_text())
    receipt = {
        "checkpoint": parent["checkpoint"], "attention_fit": parent["fit"],
        "cost": parent["cost"],
    }
    authority.validate_program_receipt(receipt)
    receipt["cost"] = {**parent["cost"], "total_stored_values": 1}
    with pytest.raises(RuntimeError, match="cost"):
        authority.validate_program_receipt(receipt)


def test_atomic_create_only_publication_and_lock(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    authority.publish_json_create_only(output, {"status": "complete"})
    assert json.loads(output.read_text()) == {"status": "complete"}
    with pytest.raises(FileExistsError):
        authority.publish_json_create_only(output, {"status": "replaced"})
    lock = tmp_path / "run.lock"
    with authority.exclusive_run_lock(lock) as claim:
        assert lock.exists()
        claim.assert_owned()
        with pytest.raises(FileExistsError):
            with authority.exclusive_run_lock(lock):
                pass
    assert not lock.exists()


def test_replaced_lock_is_not_owned_or_unlinked(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    with authority.exclusive_run_lock(lock) as claim:
        lock.unlink()
        lock.write_text("replacement\n")
        with pytest.raises(RuntimeError, match="removed or replaced"):
            claim.assert_owned()
    assert lock.read_text() == "replacement\n"


def production_geometry_bank() -> WriteGeometryBank:
    covariance = torch.eye(1152, dtype=torch.float64)
    directions = torch.zeros(32, 1152, dtype=torch.float64)
    directions[torch.arange(32), torch.arange(32)] = 1152**0.5
    geometries = {
        site: WriteCovarianceGeometry(
            site=site, count=18_432, mean=torch.zeros(1152, dtype=torch.float64),
            covariance=covariance.clone(), support_rank=1152,
            eigenvalues=torch.ones(1152, dtype=torch.float64),
            directions=directions.clone(), covariance_sha256=_tensor_sha256(covariance),
            directions_sha256=_tensor_sha256(directions),
        )
        for site in (0, 1, 2)
    }
    sites = {
        str(site): {
            "count": geometry.count, "support_rank": geometry.support_rank,
            "covariance_sha256": geometry.covariance_sha256,
            "directions_sha256": geometry.directions_sha256,
        }
        for site, geometry in geometries.items()
    }
    return WriteGeometryBank(geometries=geometries, receipt={
        "status": "complete", "plan_fingerprint": "p" * 64, "rows": 96,
        "score_support": [64, 256], "write_samples_per_site": 18_432,
        "direction_rule": "registered", "psd_rtol": 1e-10,
        "support_rtol": 1e-12, "sites": sites,
        "geometry_manifest_sha256": _json_sha256(sites),
        "raw_write_codes_returned": False,
    })


def geometry_authority_receipt(artifact: Path, bank: WriteGeometryBank) -> dict:
    return {
        "status": "tangent_geometry_frozen_no_score_outcomes",
        "protected_snapshot_fingerprint": "s" * 64,
        "program_tree_sha256": "t" * 64, "plan_fingerprint": "p" * 64,
        "program_authority_sha256": "a" * 64,
        "artifact_sha256": authority.sha256_file(artifact),
        "geometry_receipt_sha256": _json_sha256(bank.receipt),
        "geometry_receipt": bank.receipt, "score_targets_sampled": False,
        "score_gradients_computed": False, "runtime_environment": {"device": "test"},
    }


def test_frozen_geometry_reload_recomputes_inner_and_outer_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "geometry.pt"
    receipt_path = tmp_path / "geometry.json"
    bank = production_geometry_bank()
    authority.publish_torch_create_only(artifact, pilot.geometry_payload(bank))
    authority.publish_json_create_only(
        receipt_path, geometry_authority_receipt(artifact, bank),
    )
    monkeypatch.setattr(pilot, "GEOMETRY_ARTIFACT", artifact)
    monkeypatch.setattr(pilot, "GEOMETRY_RECEIPT", receipt_path)
    loaded = pilot.load_frozen_geometry()
    assert loaded.receipt == bank.receipt
    assert set(loaded.geometries) == {0, 1, 2}

    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload["geometries"]["0"]["directions"][0, 0] += 1
    new_hash = _tensor_sha256(payload["geometries"]["0"]["directions"])
    payload["geometries"]["0"]["directions_sha256"] = new_hash
    payload["receipt"]["sites"]["0"]["directions_sha256"] = new_hash
    payload["receipt"]["geometry_manifest_sha256"] = _json_sha256(
        payload["receipt"]["sites"]
    )
    changed = tmp_path / "changed.pt"
    authority.publish_torch_create_only(changed, payload)
    monkeypatch.setattr(pilot, "GEOMETRY_ARTIFACT", changed)
    altered_bank = WriteGeometryBank(
        geometries={
            int(site): WriteCovarianceGeometry(**row)
            for site, row in payload["geometries"].items()
        }, receipt=payload["receipt"],
    )
    receipt_path.write_text(json.dumps(geometry_authority_receipt(changed, altered_bank)))
    altered = pilot.load_frozen_geometry()
    with pytest.raises(RuntimeError, match="exact replay|exact program/row replay"):
        pilot.require_geometry_replay_identity(altered, bank)


def test_scientific_stages_reject_direct_invocation_without_owned_lock() -> None:
    with pytest.raises(TypeError, match="owned run lock"):
        pilot.freeze_program_authority(torch.device("cpu"), None, {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="owned run lock"):
        pilot.freeze_geometry_authority(torch.device("cpu"), None, {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="owned run lock"):
        pilot.run(None, {})  # type: ignore[arg-type]

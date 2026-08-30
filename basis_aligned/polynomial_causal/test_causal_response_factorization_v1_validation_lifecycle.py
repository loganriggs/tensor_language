import hashlib
import inspect
import json

import pytest
import torch

import causal_response_factorization_v1_validation_lifecycle as lifecycle
import causal_response_factorization_v1_validation_scorer as scorer
from causal_response_factorization_v1 import ResponseProgram
from test_causal_response_factorization_v1_validation_input import _freeze_inputs, _input


def _redirect(monkeypatch, tmp_path):
    terminal_directory = tmp_path / "terminal"
    monkeypatch.setattr(lifecycle, "TERMINAL_DIR", terminal_directory)
    for name, filename in {
        "AUTHORITY": "authority.json", "TABLE": "table.json",
        "MANIFEST": "manifest.json", "LOCK": "lock",
    }.items():
        monkeypatch.setattr(lifecycle, name, tmp_path / filename)


def test_protocol_exposes_validation_only_and_no_arguments():
    value = lifecycle.protocol()
    assert value["role"] == "FIT_INTERNAL_VALIDATION"
    assert value["validation_documents"] == 114
    assert value["training_response_values_exposed"] == 0
    assert value["eval_documents_exposed"] == 0
    assert value["candidate_programs"] == 27
    assert value["winner_selected_inside_scorer"] is False
    assert value["calibration_arm_budgets"] == [2, 4, 8, 16]
    assert len(inspect.signature(lifecycle.execute_validation_v1).parameters) == 0


def test_spent_namespace_is_refused(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    lifecycle.TABLE.write_text("{}\n")
    with pytest.raises(RuntimeError, match="namespace is spent"):
        lifecycle.execute_validation_v1()
    assert not lifecycle.LOCK.exists()


def _mock_transaction(monkeypatch, tmp_path, *, loader_error=None, scorer_error=None):
    _redirect(monkeypatch, tmp_path)
    payload, validation = _input()
    freeze, audit, freeze_sha, audit_sha = _freeze_inputs()
    fit_parent = {"binding_sha256": "4" * 64}
    closure = {"commit": "1" * 40, "paths": {"one": "2" * 64}, "sha256": "3" * 64}
    grid_terminal = {"cells": [], "manifest_sha256": freeze["grid_manifest_sha256"]}
    monkeypatch.setattr(lifecycle, "head_commit", lambda: closure["commit"])
    monkeypatch.setattr(lifecycle, "source_closure", lambda _commit: closure)
    monkeypatch.setattr(lifecycle, "focused_tests_report", lambda: {"passed": 1})
    monkeypatch.setattr(
        lifecycle, "stable_freeze_inputs", lambda: (freeze, audit, freeze_sha, audit_sha)
    )
    monkeypatch.setattr(
        lifecycle, "stable_grid_terminal", lambda _freeze: (grid_terminal, "5" * 64)
    )
    monkeypatch.setattr(
        lifecycle.rebinding, "fit_parent_binding_by_content_identity",
        lambda _paths=None: fit_parent,
    )
    monkeypatch.setattr(
        lifecycle.rebinding, "physical_identity_deviation",
        lambda _paths=None: {"artifacts": {}},
    )
    for name in ("FREEZE", "FREEZE_AUDIT"):
        source = getattr(lifecycle, name)
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        monkeypatch.setattr(lifecycle, name, target)
    events = []

    class FakeLoader:
        def load_once(self, **kwargs):
            assert lifecycle.AUTHORITY.exists()
            assert kwargs["expected_validation_authority_artifact_sha256"] == (
                lifecycle.file_sha256(lifecycle.AUTHORITY)
            )
            assert kwargs["candidate_freeze_artifact_sha256"] == freeze_sha
            events.append("loader_after_authority")
            if loader_error is not None:
                raise loader_error
            return validation

    monkeypatch.setattr(
        lifecycle.validation_loader, "OneUseFitValidationLoader", lambda: FakeLoader()
    )
    monkeypatch.setattr(
        lifecycle, "load_frozen_candidates", lambda *_args, **_kwargs: ["candidate"],
    )

    def fake_score(candidates, value, freeze_value):
        events.append("scored")
        assert candidates == ["candidate"] and value is validation and freeze_value is freeze
        if scorer_error is not None:
            raise scorer_error
        return {
            "schema": scorer.TABLE_SCHEMA, "status": scorer.TABLE_STATUS,
            "candidate_count": 27, "candidates": [], "candidate_selected": False,
            "pareto_frontier_formed": False, "candidates_dropped_after_scoring": 0,
        }

    monkeypatch.setattr(lifecycle.scorer, "score_library", fake_score)
    return events, freeze_sha


def test_success_publishes_authority_table_manifest_then_receipt(monkeypatch, tmp_path):
    events, freeze_sha = _mock_transaction(monkeypatch, tmp_path)
    digest = lifecycle.execute_validation_v1()
    assert events == ["loader_after_authority", "scored"]
    authority = json.loads(lifecycle.AUTHORITY.read_text())
    table = json.loads(lifecycle.TABLE.read_text())
    manifest = json.loads(lifecycle.MANIFEST.read_text())
    receipt = json.loads((lifecycle.TERMINAL_DIR / "receipt.json").read_text())
    terminal = json.loads((lifecycle.TERMINAL_DIR / "terminal.json").read_text())
    assert receipt == terminal
    assert hashlib.sha256(
        (lifecycle.TERMINAL_DIR / "terminal.json").read_bytes()
    ).hexdigest() == digest
    assert authority["status"] == lifecycle.validation_loader.AUTHORITY_STATUS
    assert authority["authorized_for_candidate_selection"] is False
    assert authority["candidate_freeze"]["artifact_sha256"] == freeze_sha
    assert authority["self_review"]["independent_audit"] is None
    assert "fit_parent_physical_identity_deviation" in authority["self_review"]
    assert table["candidate_selected"] is False
    assert table["authority_artifact_sha256"] == lifecycle.file_sha256(lifecycle.AUTHORITY)
    assert table["validation_binding"]["shape"] == [2, 4, 4, 1]
    assert manifest["table"]["sha256"] == lifecycle.file_sha256(lifecycle.TABLE)
    assert receipt["kind"] == "receipt"
    assert receipt["payload"]["candidate_selected"] is False
    assert receipt["payload"]["validation_values_read"] is True
    assert receipt["payload"]["eval_values_read"] is False
    assert set(receipt["terminal_snapshot"]) == {
        "authority.json", "table.json", "manifest.json",
        "candidate_freeze_v2.json", "candidate_freeze_v2_audit.json",
    }
    assert receipt["terminal_snapshot"]["candidate_freeze_v2.json"]["sha256"] == freeze_sha
    assert not lifecycle.LOCK.exists()
    with pytest.raises(RuntimeError, match="namespace is spent"):
        lifecycle.execute_validation_v1()


def test_loader_failure_publishes_failure_terminal_without_table(monkeypatch, tmp_path):
    events, _ = _mock_transaction(
        monkeypatch, tmp_path, loader_error=RuntimeError("bundle bytes differ"),
    )
    with pytest.raises(RuntimeError, match="bundle bytes differ"):
        lifecycle.execute_validation_v1()
    assert events == ["loader_after_authority"]
    failure = json.loads((lifecycle.TERMINAL_DIR / "failure.json").read_text())
    assert failure == json.loads((lifecycle.TERMINAL_DIR / "terminal.json").read_text())
    assert failure["kind"] == "failure"
    assert failure["payload"]["status"] == "failed_no_validation_receipt"
    assert failure["payload"]["error_message"] == "bundle bytes differ"
    assert set(failure["terminal_snapshot"]) == {"authority.json"}
    assert not lifecycle.TABLE.exists()
    assert not (lifecycle.TERMINAL_DIR / "receipt.json").exists()
    assert not lifecycle.LOCK.exists()


def test_scorer_failure_after_exposure_is_preserved(monkeypatch, tmp_path):
    events, _ = _mock_transaction(
        monkeypatch, tmp_path, scorer_error=ValueError("panel exploded"),
    )
    with pytest.raises(ValueError, match="panel exploded"):
        lifecycle.execute_validation_v1()
    assert events == ["loader_after_authority", "scored"]
    failure = json.loads((lifecycle.TERMINAL_DIR / "failure.json").read_text())
    assert failure["payload"]["error_type"] == "ValueError"
    assert not lifecycle.TABLE.exists()


def _cell_payload(validation, *, global_rank=1, private_rank=0, seed=1, rms=0.5):
    p, s, t, _ = validation.response.shape
    groups = validation.source_groups
    generator = torch.Generator().manual_seed(seed)

    def factor(rows, rank):
        return torch.randn((rows, rank), dtype=torch.float64, generator=generator).contiguous()

    private = [
        (factor(p, private_rank), factor(int((groups == g).sum()), private_rank), factor(t, private_rank))
        for g in range(int(groups.max()) + 1)
    ]
    program = ResponseProgram(
        factor(p, global_rank), factor(s, global_rank), factor(t, global_rank),
        private_phase=tuple(b[0] for b in private), private_source=tuple(b[1] for b in private),
        private_target=tuple(b[2] for b in private), source_groups=groups.clone(),
    )
    codes = factor(3, program.code_dimension)
    return program, {
        "schema": "causal_response_factorization_v1_grid_cell",
        "status": "complete_training_only",
        "program": {
            "global_phase": program.global_phase, "global_source": program.global_source,
            "global_target": program.global_target, "private_phase": program.private_phase,
            "private_source": program.private_source, "private_target": program.private_target,
            "source_groups": program.source_groups,
        },
        "document_codes": codes,
        "metrics": {},
        "receipt": {
            "global_rank": global_rank, "private_rank_each_owner": private_rank, "seed": seed,
            "validation_values_read": False, "eval_values_read": False,
            "training_response_rms": rms, "persistent_values": program.persistent_values,
            "per_document_values": program.code_dimension,
        },
    }


def test_load_frozen_candidates_binds_bytes_receipt_and_grid_cell(monkeypatch, tmp_path):
    _, validation = _input()
    program, payload = _cell_payload(validation)
    path = tmp_path / "g01_p00_s1.pt"
    torch.save(payload, path)
    raw = path.read_bytes()
    monkeypatch.setattr(lifecycle, "ROOT", tmp_path)
    record = {
        "artifact": path.name, "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "global_rank": 1, "private_rank_each_owner": 0, "seed": 1,
        "persistent_values": program.persistent_values,
        "per_document_values": program.code_dimension,
    }
    cell = {
        "artifact": path.name, "artifact_sha256": record["artifact_sha256"],
        "kind": "result", "healthy": True, "validation_values_read": False,
        "eval_values_read": False, "training_response_rms": 0.5,
    }
    freeze = {"candidate_programs": [record]}
    grid = {"cells": [cell]}
    candidates = lifecycle.load_frozen_candidates(
        freeze, grid, source_groups=validation.source_groups,
    )
    assert len(candidates) == 1
    assert candidates[0].training_response_rms == 0.5
    assert torch.equal(candidates[0].program.global_phase, program.global_phase)

    forged = {"candidate_programs": [{**record, "artifact_sha256": "f" * 64}]}
    with pytest.raises(RuntimeError, match="bytes changed"):
        lifecycle.load_frozen_candidates(forged, grid, source_groups=validation.source_groups)
    unhealthy = {"cells": [{**cell, "healthy": False}]}
    with pytest.raises(RuntimeError, match="healthy grid result"):
        lifecycle.load_frozen_candidates(freeze, unhealthy, source_groups=validation.source_groups)
    drifted_rms = {"cells": [{**cell, "training_response_rms": 0.7}]}
    with pytest.raises(RuntimeError, match="receipt does not bind"):
        lifecycle.load_frozen_candidates(freeze, drifted_rms, source_groups=validation.source_groups)
    with pytest.raises(RuntimeError, match="owner topology"):
        lifecycle.load_frozen_candidates(
            freeze, grid, source_groups=torch.zeros_like(validation.source_groups),
        )

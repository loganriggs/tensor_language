from __future__ import annotations

import json

import pytest

import early_mlp_context_cross_v1_lifecycle as lifecycle
import early_mlp_context_cross_v1_statistics as statistics
import run_early_mlp_context_cross_v1 as runner
import score_early_mlp_context_cross_v1 as scorer
from test_run_early_mlp_context_cross_v1 import FakeBackend, _source


def _write_json(path, value):
    path.write_text(json.dumps(
        value, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n")


def _rebind_manifest_in_terminal_receipt(paths):
    receipt = json.loads(paths.receipt.read_text())
    receipt["manifest_file_sha256"] = lifecycle.file_sha256(paths.manifest)
    _write_json(paths.receipt, receipt)


@pytest.fixture
def terminal_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "committed_source_closure", _source)
    monkeypatch.setattr(lifecycle, "verify_source_closure", lambda _source: None)
    paths = lifecycle.output_paths(tmp_path, "score_input")
    runner.run_transaction(backend=FakeBackend(), paths=paths)
    return paths


@pytest.mark.parametrize(
    "rank3_by_role,rank4_by_role,expected_rank,expected_final",
    (
        ({"skip7000": True, "skip11000": True},
         {"skip7000": True, "skip11000": True}, 3, True),
        ({"skip7000": True, "skip11000": False},
         {"skip7000": True, "skip11000": True}, 4, True),
        ({"skip7000": True, "skip11000": True},
         {"skip7000": True, "skip11000": False}, 3, True),
    ),
)
def test_two_role_conjunction_and_minimal_rank_selection(
    terminal_measurement, tmp_path, monkeypatch,
    rank3_by_role, rank4_by_role, expected_rank, expected_final,
):
    def fake_score(discovery, validation, heldout, *, rank):
        assert discovery.role == validation.role
        assert (heldout is None) == (rank == 3)
        passed = (
            rank3_by_role if rank == 3 else rank4_by_role
        )[discovery.role]
        return {
            "role": discovery.role, "rank": rank,
            "ce_useful_pass": passed,
            "targets": {"ce_nats": {}, "top1_pp": {}},
        }

    monkeypatch.setattr(statistics, "score_rank", fake_score)
    paths = scorer.score_paths(tmp_path, f"score_case_{len(list(tmp_path.iterdir()))}")
    receipt = scorer.score_transaction(
        measurement_paths=terminal_measurement, paths=paths,
    )
    results = json.loads(paths.results.read_text())
    assert receipt["selected_minimal_rank"] == expected_rank
    assert receipt["ce_any_registered_pass"] is expected_final
    assert receipt["top1_broad_behavior_pass"] is None
    assert results["two_role_ce_rank3_pass"] is all(rank3_by_role.values())
    assert results["two_role_ce_rank4_pass"] is all(rank4_by_role.values())
    assert paths.receipt.exists() and not paths.failure.exists() and not paths.lock.exists()


def test_missing_terminal_receipt_refuses_to_open_payload(tmp_path):
    paths = lifecycle.output_paths(tmp_path, "incomplete_measurement")
    paths.payload.write_bytes(b"not opened")
    with pytest.raises(RuntimeError, match="terminal receipt"):
        scorer.load_terminal_bundles(paths)


def test_tampered_predecessor_is_rejected_before_scoring(
    terminal_measurement, tmp_path,
):
    terminal_measurement.manifest.write_text("{}\n")
    paths = scorer.score_paths(tmp_path, "score_tampered")
    with pytest.raises(RuntimeError, match="terminal receipt"):
        scorer.score_transaction(
            measurement_paths=terminal_measurement, paths=paths,
        )
    assert paths.failure.exists() and not paths.results.exists() and not paths.receipt.exists()


def test_self_rebound_corrupt_cell_ledger_fails_semantic_replay(
    terminal_measurement,
):
    manifest = json.loads(terminal_measurement.manifest.read_text())
    record = manifest["cell_audit_records"]["skip7000"][0]
    record["call_ledger"]["native_module_calls"][0][1] += 1
    _write_json(terminal_measurement.manifest, manifest)
    _rebind_manifest_in_terminal_receipt(terminal_measurement)
    with pytest.raises((RuntimeError, ValueError), match="ledger|census"):
        scorer.load_terminal_bundles(terminal_measurement)


def test_self_rebound_role_authority_cross_link_fails_semantic_replay(
    terminal_measurement,
):
    authority = json.loads(terminal_measurement.authority.read_text())
    authority["role_authorities"]["skip7000"] = authority[
        "role_authorities"
    ]["skip11000"]
    _write_json(terminal_measurement.authority, authority)
    manifest = json.loads(terminal_measurement.manifest.read_text())
    manifest["authority_file_sha256"] = lifecycle.file_sha256(
        terminal_measurement.authority
    )
    _write_json(terminal_measurement.manifest, manifest)
    receipt = json.loads(terminal_measurement.receipt.read_text())
    receipt["authority_file_sha256"] = manifest["authority_file_sha256"]
    receipt["manifest_file_sha256"] = lifecycle.file_sha256(
        terminal_measurement.manifest
    )
    _write_json(terminal_measurement.receipt, receipt)
    with pytest.raises(RuntimeError, match="authority hash chain"):
        scorer.load_terminal_bundles(terminal_measurement)


def test_self_rebound_stage_manifest_hash_fails_semantic_replay(
    terminal_measurement,
):
    manifest = json.loads(terminal_measurement.manifest.read_text())
    manifest["stage_payload_sha256s"]["skip7000"]["validation"] = "f" * 64
    _write_json(terminal_measurement.manifest, manifest)
    _rebind_manifest_in_terminal_receipt(terminal_measurement)
    with pytest.raises(RuntimeError, match="stage hash chain"):
        scorer.load_terminal_bundles(terminal_measurement)

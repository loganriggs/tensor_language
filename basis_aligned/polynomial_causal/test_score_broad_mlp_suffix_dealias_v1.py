from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

import broad_mlp_suffix_dealias_v1 as assay
import broad_mlp_suffix_dealias_v1_lifecycle as lifecycle
import broad_mlp_suffix_dealias_v1_measurements as measurement
import run_broad_mlp_suffix_dealias_v1 as runner
import score_broad_mlp_suffix_dealias_v1 as scorer
import score_early_mlp_context_cross_v1 as parent_scorer
from test_run_broad_mlp_suffix_dealias_v1 import FakeBackend, _source


def _write_json(path, value):
    path.write_text(json.dumps(
        value, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n")


@pytest.fixture
def terminal_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "committed_source_closure", _source)
    monkeypatch.setattr(lifecycle, "verify_source_closure", lambda _source: None)
    paths = lifecycle.output_paths(tmp_path, "broad_score_input")
    runner.run_transaction(backend=FakeBackend(), paths=paths)
    return paths


def _load_join_inputs(paths):
    old_bundles, _old_receipt = parent_scorer.load_terminal_bundles(
        lifecycle.PARENT_PATHS, require_authoritative=True,
    )
    new_bundles, new_authorities, _new_receipt = scorer.load_new_terminal_bundles(
        paths, require_authoritative=False,
    )
    parent_authority = lifecycle.parent_authority()
    parent_receipt_sha = lifecycle.file_sha256(lifecycle.PARENT_PATHS.receipt)
    return old_bundles, new_bundles, new_authorities, parent_authority, parent_receipt_sha


def test_exact_document_and_denominator_join_builds_ce_role_arrays(
    terminal_measurement,
):
    old, new, authorities, parent_authority, parent_receipt_sha = _load_join_inputs(
        terminal_measurement
    )
    for role in assay.ROLE_NAMES:
        arrays = scorer.join_ce_role_arrays(
            role=role, old_bundle=old[role], new_bundle=new[role],
            new_authority=authorities[role], parent_authority=parent_authority,
            parent_receipt_file_sha256=parent_receipt_sha,
        )
        old_grid = scorer._old_grid(old[role], "ce_sum")
        assert arrays.role == role
        assert arrays.e.shape == arrays.a.shape == arrays.m.shape == arrays.am.shape == (
            measurement.ROLE_DOCUMENT_COUNTS[role], assay.CELL_COUNT,
        )
        assert torch.equal(torch.from_numpy(arrays.e), old_grid[:, :, 0])
        assert torch.equal(torch.from_numpy(arrays.a), old_grid[:, :, 4])
        assert torch.equal(torch.from_numpy(arrays.am), old_grid[:, :, 5])
        assert torch.equal(torch.from_numpy(arrays.m), new[role].statistics.ce_sum)
        assert torch.equal(
            torch.from_numpy(arrays.token_count).long(),
            old[role].discovery.document_token_count,
        )


def test_top1_secondary_reuses_exact_join_and_reports_percentage_points(
    terminal_measurement,
):
    old, new, authorities, parent_authority, parent_receipt_sha = _load_join_inputs(
        terminal_measurement
    )
    role = "skip7000"
    arrays = scorer.join_top1_role_arrays(
        role=role, old_bundle=old[role], new_bundle=new[role],
        new_authority=authorities[role], parent_authority=parent_authority,
        parent_receipt_file_sha256=parent_receipt_sha,
    )
    old_grid = scorer._old_grid(old[role], "top1_correct")
    assert torch.equal(torch.from_numpy(arrays.e), old_grid[:, :, 0].double())
    assert torch.equal(torch.from_numpy(arrays.m), new[role].statistics.top1_correct.double())
    result = scorer.score_top1_secondary(arrays)
    assert result["unit"] == "percentage_points"
    assert result["decision_role"] == "mandatory_secondary_no_gate"
    assert set(result["contrasts_percentage_points"]) == {
        "d_a", "d_m", "d_am", "prediction", "r", "q",
        "standalone_m_marginal",
    }


def test_join_rejects_mixed_document_identity(terminal_measurement):
    old, new, authorities, parent_authority, parent_receipt_sha = _load_join_inputs(
        terminal_measurement
    )
    role = "skip7000"
    forged_statistics = measurement.RoleStatistics(
        role=role, authority_sha256=new[role].statistics.authority_sha256,
        ordered_document_ids_sha256="f" * 64,
        document_token_count=new[role].statistics.document_token_count,
        top1_correct=new[role].statistics.top1_correct,
        ce_sum=new[role].statistics.ce_sum,
    )
    forged_receipt = replace(
        new[role].receipt, statistics_sha256=forged_statistics.sha256,
    )
    forged = measurement.RoleBundle(
        statistics=forged_statistics, receipt=forged_receipt,
    )
    with pytest.raises(RuntimeError, match="provenance|support"):
        scorer.join_ce_role_arrays(
            role=role, old_bundle=old[role], new_bundle=forged,
            new_authority=authorities[role], parent_authority=parent_authority,
            parent_receipt_file_sha256=parent_receipt_sha,
        )


def test_join_rejects_mixed_per_document_token_denominators(terminal_measurement):
    old, new, authorities, parent_authority, parent_receipt_sha = _load_join_inputs(
        terminal_measurement
    )
    role = "skip7000"
    tokens = new[role].statistics.document_token_count.clone()
    tokens[0] += 1
    tokens[1] -= 1
    forged_statistics = measurement.RoleStatistics(
        role=role, authority_sha256=new[role].statistics.authority_sha256,
        ordered_document_ids_sha256=new[role].statistics.ordered_document_ids_sha256,
        document_token_count=tokens,
        top1_correct=new[role].statistics.top1_correct,
        ce_sum=new[role].statistics.ce_sum,
    )
    forged = measurement.RoleBundle(
        statistics=forged_statistics,
        receipt=replace(
            new[role].receipt, statistics_sha256=forged_statistics.sha256,
        ),
    )
    with pytest.raises(RuntimeError, match="token denominators"):
        scorer.join_ce_role_arrays(
            role=role, old_bundle=old[role], new_bundle=forged,
            new_authority=authorities[role], parent_authority=parent_authority,
            parent_receipt_file_sha256=parent_receipt_sha,
        )


def test_scores_both_roles_and_both_conditional_directions_receipt_last(
    terminal_measurement, tmp_path, monkeypatch,
):
    role_calls, cross_calls, publications = [], [], []

    def fake_role(value):
        role_calls.append(value.role)
        return {"role": value.role, "useful_pass": value.role == "skip7000"}

    def fake_cross(source, target):
        cross_calls.append((source.role, target.role))
        return {
            "source_role": source.role, "target_role": target.role,
            "useful_pass": True,
        }

    monkeypatch.setattr(assay, "score_role", fake_role)
    monkeypatch.setattr(assay, "score_cross_role", fake_cross)
    real_publish = lifecycle.publish_json_create_only

    def publish(path, value, lock):
        publications.append(path.name)
        return real_publish(path, value, lock)

    monkeypatch.setattr(lifecycle, "publish_json_create_only", publish)
    paths = scorer.score_paths(tmp_path, "broad_score_output")
    receipt = scorer.score_transaction(
        measurement_paths=terminal_measurement, paths=paths,
    )
    result = json.loads(paths.results.read_text())
    assert role_calls == list(assay.ROLE_NAMES)
    assert cross_calls == [
        ("skip7000", "skip11000"), ("skip11000", "skip7000"),
    ]
    assert publications == [paths.results.name, paths.receipt.name]
    assert receipt["attention_invariance_useful_pass"] is False
    assert result["two_role_within_role_pass"] is False
    assert result["two_direction_conditional_cross_role_pass"] is True
    assert result["authorized_for_global_ledger_credit"] is False
    assert set(result["top1_secondary"]) == set(assay.ROLE_NAMES)
    assert all(
        value["decision_role"] == "mandatory_secondary_no_gate"
        for value in result["top1_secondary"].values()
    )
    assert "no OOD" in result["claim_boundary"]
    assert paths.receipt.exists() and not paths.failure.exists() and not paths.lock.exists()


def test_missing_new_terminal_receipt_refuses_to_open_payload(tmp_path):
    paths = lifecycle.output_paths(tmp_path, "broad_incomplete")
    paths.payload.write_bytes(b"not a torch payload")
    with pytest.raises(RuntimeError, match="terminal receipt"):
        scorer.load_new_terminal_bundles(paths, require_authoritative=False)


def test_self_rebound_wrong_mlp_mask_fails_physical_replay(terminal_measurement):
    authority = json.loads(terminal_measurement.authority.read_text())
    authority["program_descriptors"][0]["installed_compiled_sites"][0] = ["attn", 3]
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
    with pytest.raises(RuntimeError, match="wrong mask"):
        scorer.load_new_terminal_bundles(
            terminal_measurement, require_authoritative=False,
        )


def test_non_authoritative_measurement_cannot_receive_canonical_score(
    terminal_measurement,
):
    canonical = scorer.score_paths()
    with pytest.raises(RuntimeError, match="canonical new measurement namespace"):
        scorer.score_transaction(
            measurement_paths=terminal_measurement, paths=canonical,
        )
    assert not any(path.exists() for path in (
        canonical.results, canonical.receipt, canonical.failure, canonical.lock,
    ))

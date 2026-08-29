from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import validate_mlp2_cmr_v1 as validation


def test_source_closure_is_exact_once_and_contains_runtime_contracts() -> None:
    assert len(validation.SOURCE_CLOSURE) == len(set(validation.SOURCE_CLOSURE))
    names = {path.name for path in validation.SOURCE_CLOSURE}
    assert names == {
        "MLP2_CMR_V1_PREREGISTRATION.md",
        "MLP2_CMR_V1_VALIDATION_ADDENDUM.md",
        "validate_mlp2_cmr_v1.py",
        "test_validate_mlp2_cmr_v1.py",
        "mlp2_cmr_v1_validation_runtime.py",
        "test_mlp2_cmr_v1_validation_runtime.py",
        "mlp2_cmr_v1_validation_statistics.py",
        "test_mlp2_cmr_v1_validation_statistics.py",
        "mlp2_cmr_v1_physical_program.py",
        "test_mlp2_cmr_v1_physical_program.py",
        "mlp2_cmr_v1_suffix_math.py",
        "test_mlp2_cmr_v1_suffix_math.py",
        "project_mlp2_cmr_v1_validation_rows.py",
        "test_project_mlp2_cmr_v1_validation_rows.py",
        "MLP2_CMR_V1_MARGIN_FREQUENCY_ADDENDUM.md",
        "materialize_mlp2_cmr_v1_token_rows.py",
        "test_materialize_mlp2_cmr_v1_token_rows.py",
        "project_mlp2_cmr_v1_fit_selector_rows.py",
        "test_project_mlp2_cmr_v1_fit_selector_rows.py",
        "bilin18_observed_model_facade.py",
        "tt_model.py",
    }


def test_selector_gauge_replay_checks_zero_error_as_a_number_not_false() -> None:
    valid = {
        "gauge_and_permutation_audit": {
            "channel_permutation": {
                "derangement_equivariant": True,
                "hash_random_equivariant": True,
                "suffix_support_equivariant": True,
            },
            "dyadic_reciprocal": {
                "canonical_down_max_abs_error": 0.0,
                "derangement_exact": True,
                "hash_random_exact": True,
            },
            "general_reciprocal_functional": {
                "canonical_down_max_relative_error": 4.8e-16,
                "hash_byte_replay_required": False,
            },
        },
    }
    assert validation._selector_gauge_passes(valid)
    invalid = {
        "gauge_and_permutation_audit": {
            **valid["gauge_and_permutation_audit"],
            "dyadic_reciprocal": {
                **valid["gauge_and_permutation_audit"]["dyadic_reciprocal"],
                "derangement_exact": False,
            },
        },
    }
    assert not validation._selector_gauge_passes(invalid)


def test_runner_is_role_only_and_never_names_combined_or_replication_rows() -> None:
    source = inspect.getsource(validation)
    assert "mlp2_cmr_v1_token_rows.pt" not in source
    assert "mlp2_cmr_v1_replication" not in source
    assert '"authorized_role": "VALIDATION"' in source
    assert '"authorized_for_replication_execution": False' in source
    assert '"raw_logits_published": False' in source
    assert '"per_token_losses_published": False' in source
    assert '"validation_targets_published": False' in source


def test_expected_parent_hashes_are_complete_exact_and_files_exist() -> None:
    assert set(validation.PARENT_PATHS) == set(validation.EXPECTED_PARENTS)
    assert all(path.is_file() for path in validation.PARENT_PATHS.values())
    assert {
        name: validation.file_sha256(path)
        for name, path in validation.PARENT_PATHS.items()
    } == validation.EXPECTED_PARENTS


def test_parent_dag_replays_transitive_validation_license() -> None:
    hashes, captured = validation.protected_inputs()
    assert hashes == validation.EXPECTED_PARENTS
    changed = dict(captured)
    suffix_receipt = json.loads(changed["suffix_receipt"])
    suffix_receipt["authorized_for_validation"] = False
    changed["suffix_receipt"] = validation.canonical_json_bytes(suffix_receipt)
    with pytest.raises(RuntimeError, match="suffix parent/license"):
        validation.validate_parent_dag(changed)


def _passing_call_ledger() -> dict:
    ledger = validation.runtime.new_call_ledger()
    for arm in validation.runtime.ALL_ARMS:
        ledger[arm]["forward_calls"] = validation.runtime.CALLS
        ledger[arm]["forward_returns"] = validation.runtime.CALLS
        ledger[arm]["attention_calls_by_site"] = [validation.runtime.CALLS] * 18
        ledger[arm]["native_mlp_calls_by_site"] = [validation.runtime.CALLS] * 18
        if arm == "ZERO" or arm in validation.runtime.PHYSICAL_ARMS:
            ledger[arm]["native_mlp_calls_by_site"][validation.runtime.SITE] = 0
        ledger[arm]["physical_mlp2_calls"] = (
            validation.runtime.CALLS
            if arm in validation.runtime.PHYSICAL_ARMS
            or arm in validation.runtime.SIGNED_T else 0
        )
        ledger[arm]["zero_mlp2_calls"] = (
            validation.runtime.CALLS if arm == "ZERO" else 0
        )
        ledger[arm]["diagnostic_full_product_evaluations"] = (
            validation.runtime.CALLS if arm == "NATIVE" else 0
        )
    return ledger


def test_protocol_audits_are_derived_from_evidence_not_stored_booleans() -> None:
    suffix_result = json.loads(validation.SUFFIX_RESULT.read_text())
    correction = json.loads(validation.CORRECTION.read_text())
    historical = suffix_result["gauge_and_permutation_audit"]
    evidence = {
        "program_receipts": {
            arm: dict(validation.EXPECTED_PROGRAM_RECEIPT)
            for arm in validation.runtime.PHYSICAL_ARMS
        },
        "support_hashes": correction["support_hashes"],
        "selector_gauge_and_permutation_audit": historical,
        "selector_gauge_permutation_replay": historical,
        "physical_gauge_permutation_replay": {
            "currency": "CPU float64 copies of materialized owned buffers",
            "tolerance": 5e-12,
            "per_arm": {
                arm: {
                    "permutation_max_absolute_error": 0.0,
                    "dyadic_max_relative_error": 0.0,
                    "general_max_relative_error": 0.0,
                    "passed": True,
                } for arm in validation.runtime.PHYSICAL_ARMS
            },
            "passed": True,
        },
        "physical_materialization": {
            "maximum_absolute_error": 0.0,
            "bit_exact": True,
            "per_arm_maximum_absolute_error": {
                arm: 0.0 for arm in validation.runtime.PHYSICAL_ARMS
            },
        },
        "call_ledger": _passing_call_ledger(),
        "precision_audit": {
            "maximum_native_nll_absolute_error": 0.0,
            "maximum_candidate_nll_absolute_error": 0.0,
            "maximum_teacher_kl_absolute_error": 0.0,
            "maximum_raw_sse_relative_error": 0.0,
            "maximum_centered_sse_relative_error": 0.0,
            "maximum_native_centered_energy_relative_error": 0.0,
            "passed": True,
        },
    }
    audits = validation.derive_protocol_audits(
        evidence, expected_support_hashes=correction["support_hashes"],
        expected_selector_audit=historical,
    )
    assert all(audits.values())
    evidence["program_receipts"]["SUFFIX"]["stored_scalar_values"] += 1
    audits = validation.derive_protocol_audits(
        evidence, expected_support_hashes=correction["support_hashes"],
        expected_selector_audit=historical,
    )
    assert audits["exact_price_and_support_replay"] is False


@pytest.mark.parametrize("bad", [1e99, float("nan"), -1.0, "0.0"])
def test_precision_protocol_rejects_bad_numeric_evidence(bad) -> None:
    suffix_result = json.loads(validation.SUFFIX_RESULT.read_text())
    correction = json.loads(validation.CORRECTION.read_text())
    historical = suffix_result["gauge_and_permutation_audit"]
    # Reuse the fully passing evidence construction above without depending on a
    # stored protocol Boolean.
    precision = {
        "maximum_native_nll_absolute_error": 0.0,
        "maximum_candidate_nll_absolute_error": 0.0,
        "maximum_teacher_kl_absolute_error": 0.0,
        "maximum_raw_sse_relative_error": 0.0,
        "maximum_centered_sse_relative_error": 0.0,
        "maximum_native_centered_energy_relative_error": 0.0,
        "passed": True,
    }
    evidence = {
        "program_receipts": {
            arm: dict(validation.EXPECTED_PROGRAM_RECEIPT)
            for arm in validation.runtime.PHYSICAL_ARMS
        },
        "support_hashes": correction["support_hashes"],
        "selector_gauge_and_permutation_audit": historical,
        "selector_gauge_permutation_replay": historical,
        "physical_gauge_permutation_replay": {
            "currency": "CPU float64 copies of materialized owned buffers",
            "tolerance": 5e-12,
            "per_arm": {
                arm: {
                    "permutation_max_absolute_error": 0.0,
                    "dyadic_max_relative_error": 0.0,
                    "general_max_relative_error": 0.0,
                    "passed": True,
                } for arm in validation.runtime.PHYSICAL_ARMS
            },
            "passed": True,
        },
        "physical_materialization": {
            "maximum_absolute_error": 0.0,
            "bit_exact": True,
            "per_arm_maximum_absolute_error": {
                arm: 0.0 for arm in validation.runtime.PHYSICAL_ARMS
            },
        },
        "call_ledger": _passing_call_ledger(),
        "precision_audit": precision,
    }
    precision["maximum_native_nll_absolute_error"] = bad
    audits = validation.derive_protocol_audits(
        evidence, expected_support_hashes=correction["support_hashes"],
        expected_selector_audit=historical,
    )
    assert audits["float32_cpu_float64_precision_audit"] is False


def test_materialization_protocol_rejects_boolean_zero() -> None:
    assert not validation._materialization_passes({
        "maximum_absolute_error": False,
        "bit_exact": True,
        "per_arm_maximum_absolute_error": {
            arm: 0.0 for arm in validation.runtime.PHYSICAL_ARMS
        },
    })


def test_capability_is_closed_one_use_and_noncopyable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(validation, "LOCK", tmp_path / "lock")
    monkeypatch.setattr(validation, "AUTHORITY", tmp_path / "authority")
    validation.projection.base.write_create_only(
        validation.LOCK, validation.canonical_json_bytes({"nonce": "n"}),
    )
    stat = validation.LOCK.stat(follow_symlinks=False)
    inode = (stat.st_dev, stat.st_ino)
    authority = {
        "status": "authority_frozen_before_validation_model_access",
        "authorized_role": "VALIDATION",
        "authorized_forward_calls": {
            arm: validation.runtime.CALLS for arm in validation.runtime.ALL_ARMS
        },
    }
    validation.projection.base.write_create_only(
        validation.AUTHORITY, validation.canonical_json_bytes(authority),
    )
    authority_hash = validation.file_sha256(validation.AUTHORITY)
    capability = validation._mint_capability("n", inode, authority_hash)
    with pytest.raises(TypeError, match="constructible"):
        validation._ValidationCapability(object(), "n", inode, authority_hash)
    validation._consume_capability(capability)
    with pytest.raises(RuntimeError, match="fresh"):
        validation._consume_capability(capability)


def _transaction_fixture(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: tmp_path / name for name in (
            "authority.json", "ledger.pt", "result.json", "receipt.json",
            "failure.json", "lock",
        )
    }
    for attribute, name in (
        ("AUTHORITY", "authority.json"), ("LEDGER", "ledger.pt"),
        ("RESULT", "result.json"), ("RECEIPT", "receipt.json"),
        ("FAILURE", "failure.json"), ("LOCK", "lock"),
    ):
        monkeypatch.setattr(validation, attribute, paths[name])
    parent_hashes = {"role_rows": "parent"}
    parent_bytes = {"role_rows": b"rows"}
    monkeypatch.setattr(validation, "committed_source", lambda: ("commit", {}))
    monkeypatch.setattr(
        validation, "protected_inputs", lambda: (parent_hashes, parent_bytes),
    )
    capability = SimpleNamespace(consumed=True)
    monkeypatch.setattr(validation, "_mint_capability", lambda *_: capability)
    ledger = {"marker": torch.tensor([1], dtype=torch.long)}
    result = {
        "schema": "mlp2_cmr_v1_validation_result",
        "score": {"validation_passed": False, "replication_authorized": False},
        "protocol_audits": {},
    }
    monkeypatch.setattr(validation, "collect", lambda *_: (ledger, result.copy()))
    monkeypatch.setattr(validation, "guard_inputs", lambda *_: None)
    monkeypatch.setattr(validation, "final_guard", lambda *_: None)
    monkeypatch.setattr(validation, "validate_output_semantics", lambda *_: None)
    return paths


def test_transaction_success_is_receipt_last(monkeypatch, tmp_path: Path) -> None:
    paths = _transaction_fixture(monkeypatch, tmp_path)
    validation.main()
    assert paths["authority.json"].exists()
    assert paths["ledger.pt"].exists()
    assert paths["result.json"].exists()
    assert paths["receipt.json"].exists()
    assert not paths["failure.json"].exists()
    receipt = json.loads(paths["receipt.json"].read_text())
    assert receipt["authorized_for_replication_implementation"] is False
    assert receipt["authorized_for_replication_execution"] is False


def test_semantic_replay_failure_publishes_failure_not_receipt(
    monkeypatch, tmp_path: Path,
) -> None:
    paths = _transaction_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        validation, "validate_output_semantics",
        lambda *_: (_ for _ in ()).throw(RuntimeError("semantic replay failed")),
    )
    with pytest.raises(RuntimeError, match="semantic replay failed"):
        validation.main()
    assert paths["failure.json"].exists()
    assert not paths["receipt.json"].exists()
    failure = json.loads(paths["failure.json"].read_text())
    assert failure["status"].endswith("no_scientific_decision")
    assert failure["replication_opened"] is False


def test_bidirectional_terminal_races_leave_only_one_terminal(
    monkeypatch, tmp_path: Path,
) -> None:
    paths = _transaction_fixture(monkeypatch, tmp_path)
    original_guarded = validation.projection.write_create_only_guarded

    def failure_wins(path, data, *, before_link):
        if path == paths["receipt.json"]:
            validation.projection.base.write_create_only(paths["failure.json"], b"{}")
        return original_guarded(path, data, before_link=before_link)

    monkeypatch.setattr(
        validation.projection, "write_create_only_guarded", failure_wins,
    )
    with pytest.raises(RuntimeError, match="terminal namespace"):
        validation.main()
    assert paths["failure.json"].exists()
    assert not paths["receipt.json"].exists()

    second = tmp_path / "receipt_wins"
    second.mkdir()
    paths = _transaction_fixture(monkeypatch, second)
    monkeypatch.setattr(
        validation, "validate_output_semantics",
        lambda *_: (_ for _ in ()).throw(RuntimeError("semantic replay failed")),
    )
    original_guarded = validation.projection.write_create_only_guarded

    def receipt_wins(path, data, *, before_link):
        if path == paths["failure.json"]:
            validation.projection.base.write_create_only(paths["receipt.json"], b"{}")
        return original_guarded(path, data, before_link=before_link)

    monkeypatch.setattr(
        validation.projection, "write_create_only_guarded", receipt_wins,
    )
    with pytest.raises(RuntimeError, match="semantic replay failed"):
        validation.main()
    assert paths["receipt.json"].exists()
    assert not paths["failure.json"].exists()

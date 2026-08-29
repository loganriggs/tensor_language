import copy
import json
from pathlib import Path

import pytest

import prepare_terminal_copy_induction_v2_rows as v2
import terminal_copy_registry_recovery_v2 as recovery


def _transaction(tmp_path: Path):
    stem = tmp_path / "toy_v1"
    paths = {
        "authority": stem.with_name(stem.name + "_authority.json"),
        "failure": stem.with_name(stem.name + "_failure.json"),
        "manifest": stem.with_name(stem.name + "_manifest.json"),
        "receipt": stem.with_name(stem.name + "_receipt.json"),
        "rows": stem.with_name(stem.name + "_rows.pt"),
        "lock": tmp_path / ".toy_v1.lock",
    }
    authority = {
        "schema": "toy_v1_authority",
        "status": "frozen_before_any_row_tensor_model_or_outcome_load",
        "authority_sha256": "a" * 64,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    failure = {
        "schema": "toy_v1_failure",
        "status": "terminal_failure_no_receipt",
        "rows_exists": False,
        "manifest_exists": False,
        "receipt_exists": False,
        "exception_type": "RuntimeError",
        "exception_message": "toy failure",
    }
    paths["authority"].write_text(json.dumps(authority))
    paths["failure"].write_text(json.dumps(failure))
    return paths, authority, failure


def _identity(paths, authority, failure):
    return {
        "authority_path": str(paths["authority"].resolve()),
        "authority_sha256": recovery.file_sha256(paths["authority"]),
        "internal_authority_sha256": authority["authority_sha256"],
        "failure_sha256": recovery.file_sha256(paths["failure"]),
    }


def test_exact_failed_unmaterialized_authority_is_accepted(tmp_path):
    paths, authority, _ = _transaction(tmp_path)
    ledger = recovery.validate_failed_unmaterialized_authority(
        paths["authority"], authority, paths["rows"],
        expected_identity=_identity(paths, authority, {}),
    )
    assert ledger["kind"] == "failed_unmaterialized_registry"
    assert ledger["omitted_missing_row_path"] == str(paths["rows"].resolve())
    assert len(ledger["authority_sha256"]) == len(ledger["failure_sha256"]) == 64


@pytest.mark.parametrize("field", ["rows_exists", "manifest_exists", "receipt_exists"])
def test_failure_claiming_any_materialized_output_is_rejected(tmp_path, field):
    paths, authority, failure = _transaction(tmp_path)
    failure[field] = True
    paths["failure"].write_text(json.dumps(failure))
    with pytest.raises(RuntimeError, match="does not prove"):
        recovery.validate_failed_unmaterialized_authority(
            paths["authority"], authority, paths["rows"],
            expected_identity=_identity(paths, authority, failure),
        )


def test_missing_failure_is_rejected(tmp_path):
    paths, authority, _ = _transaction(tmp_path)
    paths["failure"].unlink()
    with pytest.raises(RuntimeError, match="no terminal failure"):
        recovery.validate_failed_unmaterialized_authority(
            paths["authority"], authority, paths["rows"],
            expected_identity={
                "authority_path": str(paths["authority"].resolve()),
                "authority_sha256": recovery.file_sha256(paths["authority"]),
                "internal_authority_sha256": authority["authority_sha256"],
                "failure_sha256": "0" * 64,
            },
        )


@pytest.mark.parametrize("kind", ["rows", "manifest", "receipt"])
def test_any_actually_materialized_output_is_rejected(tmp_path, kind):
    paths, authority, failure = _transaction(tmp_path)
    paths[kind].write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="unexpectedly materialized"):
        recovery.validate_failed_unmaterialized_authority(
            paths["authority"], authority, paths["rows"],
            expected_identity=_identity(paths, authority, failure),
        )


def test_path_mismatch_is_rejected(tmp_path):
    paths, authority, _ = _transaction(tmp_path)
    with pytest.raises(RuntimeError, match="exact output"):
        recovery.validate_failed_unmaterialized_authority(
            paths["authority"], authority, tmp_path / "different_rows.pt",
            expected_identity=_identity(paths, authority, {}),
        )


def test_unrelated_missing_tensor_remains_fatal(tmp_path):
    registry = tmp_path / "ordinary_manifest.json"
    registry.write_text(json.dumps({"row_path": str(tmp_path / "missing_rows.pt")}))
    with pytest.raises(RuntimeError, match="unregistered missing"):
        recovery.load_registry_exclusions((registry,))


def test_failure_drift_changes_bound_ledger(tmp_path):
    paths, authority, failure = _transaction(tmp_path)
    first = recovery.validate_failed_unmaterialized_authority(
        paths["authority"], authority, paths["rows"],
        expected_identity=_identity(paths, authority, failure),
    )
    failure["exception_message"] = "different but still terminal"
    paths["failure"].write_text(json.dumps(failure))
    second = recovery.validate_failed_unmaterialized_authority(
        paths["authority"], authority, paths["rows"],
        expected_identity=_identity(paths, authority, failure),
    )
    assert first["failure_sha256"] != second["failure_sha256"]


def test_v2_configuration_uses_fresh_namespace_and_source_closes_v1_failure():
    v2.configure()
    assert v2.base.CACHE == v2.CACHE
    assert v2.base.RECEIPT == v2.RECEIPT
    assert v2.base.FAILURE == v2.FAILURE
    assert v2.V1_FAILURE in v2.base.SOURCE_PATHS
    assert v2.RECOVERY_ADDENDUM in v2.base.SOURCE_PATHS
    assert v2.FAILED_AUTHORITY_RUNNER in v2.base.SOURCE_PATHS
    assert v2.FAILED_AUTHORITY_TEST in v2.base.SOURCE_PATHS
    assert v2.FAILED_AUTHORITY_PREREG in v2.base.SOURCE_PATHS
    assert v2.base.load_prior_registry is recovery.load_registry_exclusions


def test_terminal_json_default_lock_is_resolved_at_call_time(tmp_path):
    v2.configure()
    lock = tmp_path / "v2.lock"
    target = tmp_path / "terminal.json"
    claim = v2.base.natural.acquire_claim(lock)
    old_lock = v2.base.LOCK
    try:
        v2.base.LOCK = lock
        v2.base._publish_json_terminal({"status": "ok"}, target, claim)
        assert json.loads(target.read_bytes()) == {"status": "ok"}
    finally:
        v2.base.LOCK = old_lock
        v2.base.natural.release_claim(claim, lock)

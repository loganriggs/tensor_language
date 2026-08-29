import json

import pytest

import terminal_copy_fit_mean_lifecycle as life
import terminal_copy_fit_mean_recovery_v2 as recovery


def test_v1_failure_is_exact_and_has_no_success_outputs():
    recovery.validate_v1_failure_lineage()
    failure = json.loads(recovery.V1_FAILURE.read_text())
    assert failure["bank_exists"] is False
    assert failure["result_exists"] is False
    assert failure["manifest_exists"] is False
    assert failure["receipt_exists"] is False


def test_recovery_configuration_is_create_only_and_binds_v1(monkeypatch):
    original_sources = life.SOURCE_PATHS
    names = ("AUTHORITY", "BANK", "RESULT", "MANIFEST", "RECEIPT", "FAILURE", "LOCK")
    original_outputs = {name: getattr(life, name) for name in names}
    try:
        recovery.configure()
        assert life.output_namespace() == (
            recovery.AUTHORITY, recovery.BANK, recovery.RESULT, recovery.MANIFEST,
            recovery.RECEIPT, recovery.FAILURE, recovery.LOCK,
        )
        assert set(recovery.RECOVERY_SOURCE_PATHS) <= set(life.SOURCE_PATHS)
        assert str(recovery.V1_AUTHORITY.relative_to(recovery.ROOT)) in life.SOURCE_PATHS
        assert str(recovery.V1_FAILURE.relative_to(recovery.ROOT)) in life.SOURCE_PATHS
        assert not any(path.exists() for path in life.output_namespace())
    finally:
        life.SOURCE_PATHS = original_sources
        for name, value in original_outputs.items():
            setattr(life, name, value)


def test_freeze_requires_separate_recovery_audit(monkeypatch):
    monkeypatch.setattr(
        recovery, "validate_recovery_audit",
        lambda: (_ for _ in ()).throw(RuntimeError("audit absent")),
    )
    with pytest.raises(RuntimeError, match="audit absent"):
        recovery.freeze_authority()

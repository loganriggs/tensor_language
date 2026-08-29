from pathlib import Path

import pytest

import recover_mlp0_c512_mlp2_full512_composition_v2 as recovery


def test_recovery_namespace_is_disjoint() -> None:
    assert recovery.V2_AUTHORITY != recovery.assay.AUTHORITY
    assert recovery.V2_RECEIPT != recovery.assay.RECEIPT
    assert recovery.V2_FAILURE != recovery.assay.FAILURE
    assert recovery.V2_LOCK != recovery.assay.LOCK


def test_recovery_binds_exact_pre_authority_failure() -> None:
    value = recovery.stable_v1_failure()
    assert value["evaluation_may_have_opened"] is False
    assert value["authority_exists"] is False
    assert value["artifact_hashes"] == {}


def test_recovery_atomic_authority_injects_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "authority.json"
    monkeypatch.setattr(recovery, "V2_AUTHORITY", target)
    monkeypatch.setattr(recovery, "recovery_admission", lambda: {"bound": True})
    seen = {}
    monkeypatch.setattr(recovery, "_BASE_ATOMIC_JSON",
                        lambda path, value, pre_link_check=None: seen.update(value))
    recovery.recovery_atomic_json(target, {"schema": "old"})
    assert seen["schema"] == "mlp0_c512_mlp2_full512_composition_v2_authority"
    assert seen["recovery_admission"] == {"bound": True}

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

import hierarchical_shared_private_rrr as hybrid
import run_hierarchical_shared_private_rrr_real_v1 as v1
import run_hierarchical_shared_private_rrr_real_v2_recovery as recovery


def test_spent_v1_exact_hashes_failure_and_semantic_result_replay():
    lineage = recovery.verify_spent_v1()
    assert lineage["authority_file_sha256"].startswith("558d316e")
    assert lineage["results_file_sha256"].startswith("86315dcc")
    assert lineage["failure_file_sha256"].startswith("054db06c")
    assert lineage["receipt_absent"] is True
    assert lineage["scientific_values_used_for_recovery_selection"] is False


def test_json_normalization_repairs_tuple_list_equality_and_is_idempotent():
    original = {"price": {
        "private_ranks": (1, 2, 3),
        "dense_multiplies_by_site": (8, 16, 24),
    }, "value": 4.0}
    reloaded = json.loads(json.dumps(original))
    assert reloaded != original
    normalized = recovery.json_normalize(original)
    assert normalized == reloaded
    assert recovery.json_normalize(normalized) == normalized
    assert normalized["price"]["private_ranks"] == [1, 2, 3]
    assert normalized["price"]["dense_multiplies_by_site"] == [8, 16, 24]


def _mismatch_paths(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        output = []
        for key in left:
            output.extend(_mismatch_paths(
                left[key], right[key], f"{prefix}.{key}" if prefix else key,
            ))
        return output
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)) and (
        len(left) == len(right)
    ):
        if type(left) is not type(right):
            return [prefix]
        output = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            output.extend(_mismatch_paths(a, b, f"{prefix}.{index}"))
        return output
    return [] if left == right and type(left) is type(right) else [prefix]


def test_all_seven_v1_arms_have_exactly_the_two_registered_container_mismatches():
    published = json.loads(recovery.V1_RESULTS.read_text())
    expected_paths = {
        "deployed_hash_receipt.price.private_ranks",
        "deployed_hash_receipt.price.dense_multiplies_by_site",
    }
    assert len(published["arms"]) == 7
    for payload in published["arms"].values():
        diagnostics = payload["diagnostics"]
        reconstructed = deepcopy(diagnostics)
        q0 = diagnostics["shared_rank"]
        price = hybrid.hierarchical_price(
            v1.N_SITES, v1.D, q0, diagnostics["private_ranks_by_site"],
        )
        reconstructed["deployed_hash_receipt"]["price"] = asdict(price)
        assert set(_mismatch_paths(reconstructed, diagnostics)) == expected_paths
        assert recovery.json_normalize(reconstructed) == diagnostics


def test_fit_wrapper_changes_only_result_safe_diagnostics(monkeypatch):
    program = SimpleNamespace(
        name="arm", descriptor={"name": "arm"}, deployed=object(),
        diagnostics={"nested": {"private_ranks": (3, 1)}, "finite": True},
    )
    monkeypatch.setattr(v1, "fit_program", lambda *_args, **_kwargs: program)
    monkeypatch.setattr(v1, "semantic_validate_diagnostics", lambda *_args: None)
    observed = recovery.fit_program({"name": "arm"}, object())
    assert observed is program
    assert observed.deployed is program.deployed
    assert observed.diagnostics == {"nested": {"private_ranks": [3, 1]}, "finite": True}


def test_configuration_uses_fresh_namespace_and_full_source_input_closure():
    try:
        recovery.configure_base()
        assert recovery.base.AUTHORITY == recovery.AUTHORITY
        assert recovery.base.RESULTS == recovery.RESULTS
        assert recovery.base.FAILURE == recovery.FAILURE
        assert recovery.base.RECEIPT == recovery.RECEIPT
        assert recovery.base.fit_program is recovery.fit_program
        assert recovery.base.verify_frozen_inputs is recovery.verify_frozen_inputs
        assert recovery.V1_HASHES.items() <= recovery.base.FILE_PINS.items()
        for path in (recovery.RUNNER, recovery.TEST, recovery.PREREG):
            assert str(path.relative_to(recovery.ROOT)) in recovery.base.SOURCE_PATHS
    finally:
        recovery.restore_base_defaults()


def test_v2_protocol_is_exact_v1_plus_one_change_and_lineage():
    source = {"commit": "a" * 40, "paths": {}, "sha256": "b" * 64}
    checkpoint = json.loads(recovery.V1_AUTHORITY.read_text())["checkpoint"]
    authority = recovery.authority_payload(source, recovery.FILE_PINS, checkpoint)
    v1_protocol = json.loads(recovery.V1_AUTHORITY.read_text())["protocol"]
    protocol = dict(authority["protocol"])
    lineage = protocol.pop("recovery_parent")
    change = protocol.pop("only_execution_change")
    assert protocol == v1_protocol
    assert lineage["receipt_absent"] is True
    assert "json.loads(json.dumps(program.diagnostics" in change
    assert authority["outputs"] == {
        "results": str(recovery.RESULTS),
        "failure": str(recovery.FAILURE),
        "receipt": str(recovery.RECEIPT),
    }


def test_terminal_hook_rechecks_spent_parent(monkeypatch):
    calls = []
    monkeypatch.setattr(recovery, "_INHERITED_V1_VERIFY_FROZEN_INPUTS",
                        lambda *_args, **_kwargs: calls.append("inputs"))
    monkeypatch.setattr(recovery, "verify_spent_v1", lambda: calls.append("parent"))
    recovery.verify_frozen_inputs({}, verify_checkpoint_hash=True)
    assert calls == ["inputs", "parent"]


def test_missing_v1_receipt_is_protected(tmp_path, monkeypatch):
    receipt = tmp_path / "unexpected_receipt.json"
    receipt.write_text("{}")
    monkeypatch.setattr(recovery, "V1_RECEIPT", receipt)
    with pytest.raises(RuntimeError, match="unexpectedly exists"):
        recovery.verify_spent_v1()


def test_receipt_absence_is_checked_before_and_after_parent_replay(monkeypatch):
    calls = []
    original = recovery._require_v1_receipt_absent

    def counted():
        calls.append("absent")
        original()

    monkeypatch.setattr(recovery, "_require_v1_receipt_absent", counted)
    recovery.verify_spent_v1()
    assert calls == ["absent", "absent"]


def test_cross_parent_terminal_rehash_rejects_late_drift(monkeypatch):
    original = recovery.base.file_sha256
    calls = 0

    def drift_after_individual_reads(path):
        nonlocal calls
        calls += 1
        value = original(path)
        if calls > 6 and path == recovery.V1_FAILURE:
            return "0" * 64
        return value

    monkeypatch.setattr(recovery.base, "file_sha256", drift_after_individual_reads)
    with pytest.raises(RuntimeError, match="terminal parent set changed"):
        recovery.verify_spent_v1()


def test_run_restores_base_after_preflight_refusal(monkeypatch):
    original = recovery.base.AUTHORITY
    monkeypatch.setattr(recovery, "verify_spent_v1", lambda: (_ for _ in ()).throw(
        RuntimeError("spent parent refusal")
    ))
    with pytest.raises(RuntimeError, match="spent parent refusal"):
        recovery.run(device="cpu")
    assert recovery.base.AUTHORITY == original


def test_v2_namespace_is_unopened():
    assert not any(path.exists() for path in (
        recovery.AUTHORITY, recovery.RESULTS, recovery.FAILURE,
        recovery.RECEIPT, recovery.LOCK,
    ))

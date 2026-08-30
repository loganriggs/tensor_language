from dataclasses import asdict
import json
import os

import pytest
import torch

import circuit_campaign_statistics as stats
import induction_equality_tensor_final_ood_v2 as subject


ARMS = tuple(sorted(subject.ARMS))
PAIRS = tuple(sorted(("native", arm) for arm in subject.ARMS[1:]))


def _cell(cell, *, stake=3.0):
    ce = {
        "native": 2.0, "full_replay": 2.0, "heads_deleted": stake,
        "extract_equality": 2.2, "deranged_equality": 2.9,
        "remove_equality": 2.5 if cell == "positive" else (
            2.01 if cell == "matched_negative" else 2.005
        ),
    }
    n = 10
    return stats.DocumentCellSums(
        n=n, support_sha256="a" * 64,
        arms=tuple(stats.ArmCellSums(arm, float(n * ce[arm]), 0) for arm in ARMS),
        directed_kls=tuple(stats.DirectedKLSums(source, target, 0.0) for source, target in PAIRS),
    )


def _role(*, stake=3.0, documents=192):
    ledger = {
        f"doc-{index}": {cell: _cell(cell, stake=stake) for cell in ("positive", "matched_negative", "off_target", "all")}
        for index in range(documents)
    }
    return {
        "ledger": ledger,
        "support": {
            cell: {"tokens": 10 * documents, "documents": documents}
            for cell in ("positive", "matched_negative", "off_target", "all")
        },
        "outer": {arm: {"forwards": 48, "returns": 48, "documents": 192} for arm in subject.ARMS},
        "sites": {arm: [
            ([0, 48, 48, 0] if arm != "native" and site in subject.SELECTED else [48, 0, 48, 0])
            for site in range(18)
        ] for arm in subject.ARMS},
        "replay_max_abs": 0.0,
    }


def test_observed_points_are_exact_pooled_statistics_not_bootstrap_means():
    result = subject.analyze({role: _role() for role in subject.ROLES})
    for role in subject.ROLES:
        point = result["roles"][role]["point"]
        assert point == pytest.approx([0.5, 0.49, 0.005, 0.8, 0.1])
        assert result["roles"][role]["arm_cell_reports"]["remove_equality"]["positive"] == pytest.approx({
            "tokens": 1920, "ce": 2.5, "native_to_arm_kl": 0.0, "top1_accuracy": 0.0,
        })
        assert result["roles"][role]["passed"]
    assert result["passed_both_roles"]


def test_analysis_rejects_a_self_consistent_shortened_document_ledger():
    roles = {role: _role() for role in subject.ROLES}
    roles["final_natural"] = _role(documents=40)
    with pytest.raises(RuntimeError, match="exactly 192"):
        subject.analyze(roles)


def test_document_cell_requires_exact_arms_and_kl_pairs():
    cell = _cell("positive")
    subject._validate_document_cell(cell)
    corrupted = stats.DocumentCellSums(
        n=cell.n, support_sha256=cell.support_sha256,
        arms=cell.arms[:-1], directed_kls=cell.directed_kls,
    )
    with pytest.raises(RuntimeError, match="identity schema"):
        subject._validate_document_cell(corrupted)


def test_reducer_directed_kl_order_is_accepted_by_document_validator():
    rows = torch.tensor([[0, 1, 2]], dtype=torch.long)
    logits = {
        arm: torch.zeros(1, 2, 5, dtype=torch.float32)
        for arm in subject.ARMS
    }
    masks = {"positive": torch.ones(1, 2, dtype=torch.bool)}
    reduced = stats.reduce_document_batch(
        logits, rows, masks, ("doc",),
        kl_pairs=tuple(("native", arm) for arm in subject.ARMS[1:]),
    )
    subject._validate_document_cell(reduced["doc"]["positive"])


def test_terminal_receipt_treats_post_link_directory_fsync_error_as_success(
    tmp_path, monkeypatch,
):
    target = tmp_path / "receipt.json"
    real_fsync = os.fsync
    calls = {"count": 0}

    def fail_directory_fsync(descriptor):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected post-link directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(subject.os, "fsync", fail_directory_fsync)
    subject.write_terminal_receipt(
        {"schema": "test", "value": 7}, target,
        pre_link_check=lambda: None,
    )
    assert json.loads(target.read_text()) == {"schema": "test", "value": 7}
    assert not tuple(tmp_path.glob(".receipt.json.tmp.*"))


def test_nonpositive_point_or_bootstrap_stake_fails_without_clamp():
    with pytest.raises(ValueError, match="stake is not positive"):
        subject.analyze({role: _role(stake=2.0) for role in subject.ROLES})


def test_plan_replaces_exact_selected_attention_sites_in_every_analytical_arm():
    plan = subject._plans()
    assert tuple(arm.name for arm in plan.arms) == subject.ARMS
    for arm in plan.arms:
        replaced = {item.site for item in arm.attention if item.action.value == "replace"}
        assert replaced == (set() if arm.name == "native" else set(subject.SELECTED))
        assert all(item.action.value == "native" for item in arm.mlp)


def test_semantic_validator_rejects_self_consistent_stored_gate_corruption(monkeypatch):
    roles = {role: _role() for role in subject.ROLES}
    ledger_payload = {
        "schema": "induction_equality_tensor_final_ood_v2_ledger",
        "authority_sha256": "f" * 64,
        "raw_payloads_published": False,
        "model_state_sha256_before": "e" * 64, "model_state_sha256_after": "e" * 64,
        "checkpoint_weights_sha256_before": subject.facade.WEIGHTS_SHA256,
        "checkpoint_weights_sha256_after": subject.facade.WEIGHTS_SHA256,
        "roles": {role: {
            "documents_sha256": subject.hashlib.sha256("\0".join(value["ledger"]).encode()).hexdigest(),
            "support": value["support"], "outer": value["outer"], "sites": value["sites"],
            "replay_max_abs": value["replay_max_abs"],
            "ledger": {doc: {cell: asdict(item) for cell, item in cells.items()} for doc, cells in value["ledger"].items()},
        } for role, value in roles.items()},
    }
    monkeypatch.setattr(subject, "file_sha256", lambda _path: "f" * 64)
    result = {"schema": "induction_equality_tensor_final_ood_v2_result", "authority_sha256": "f" * 64, **subject.analyze(roles)}
    subject.semantic_validate(ledger_payload, result, {})
    result["roles"]["final_natural"]["gates"]["target"] = False
    with pytest.raises(RuntimeError, match="semantic replay"):
        subject.semantic_validate(ledger_payload, result, {})


def test_semantic_replay_rejects_exact_call_census_corruption(monkeypatch):
    roles = {role: _role() for role in subject.ROLES}
    ledger_payload = {
        "schema": "induction_equality_tensor_final_ood_v2_ledger",
        "authority_sha256": "f" * 64, "raw_payloads_published": False,
        "model_state_sha256_before": "e" * 64, "model_state_sha256_after": "e" * 64,
        "checkpoint_weights_sha256_before": subject.facade.WEIGHTS_SHA256,
        "checkpoint_weights_sha256_after": subject.facade.WEIGHTS_SHA256,
        "roles": {role: {
            "documents_sha256": subject.hashlib.sha256("\0".join(value["ledger"]).encode()).hexdigest(),
            "support": value["support"], "outer": value["outer"], "sites": value["sites"],
            "replay_max_abs": value["replay_max_abs"],
            "ledger": {doc: {cell: asdict(item) for cell, item in cells.items()} for doc, cells in value["ledger"].items()},
        } for role, value in roles.items()},
    }
    result = {"schema": "induction_equality_tensor_final_ood_v2_result", "authority_sha256": "f" * 64, **subject.analyze(roles)}
    monkeypatch.setattr(subject, "file_sha256", lambda _path: "f" * 64)
    ledger_payload["roles"]["ood_code"]["sites"]["remove_equality"][5] = [1, 47, 48, 0]
    with pytest.raises(RuntimeError, match="call census"):
        subject.semantic_validate(ledger_payload, result, {})


def test_v1_no_go_is_preserved_and_old_roles_are_not_runner_inputs():
    assert subject.rows_v2.V1_AUDIT_SHA256 == "3fae8d163a367c2af600fbe584f457ace7537a9688e3b091c379f7ebc9b043da"
    assert set(subject.ROLES) == {"final_natural", "ood_code"}
    assert all("rowcache_terminal_copy_induction_v2" not in str(path) for path in subject.SOURCE_PATHS)


def test_execution_source_binding_uses_commit_named_by_stable_audit(tmp_path, monkeypatch):
    commit, sources = "c" * 40, {"runner.py": "d" * 64}
    audit = {"schema": "induction_equality_tensor_final_ood_v2_independent_audit", "status": "GO", "outcome_access": False, "audited_source_commit": commit, "audited_source_hashes": sources, "tests_passed": 1, "reviewer": "reviewer"}
    monkeypatch.setattr(subject, "AUDIT", tmp_path / "audit.json")
    subject.AUDIT.write_text(subject.json.dumps(audit))
    monkeypatch.setattr(subject, "source_closure", lambda selected: sources if selected == commit else (_ for _ in ()).throw(AssertionError("moving HEAD selected")))
    assert subject.audited_source_binding() == (commit, sources, audit)

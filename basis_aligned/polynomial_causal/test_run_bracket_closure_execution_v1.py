from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import bracket_closure_canary_v1 as canary
import bracket_closure_execution_lifecycle_v1 as lifecycle
import bracket_closure_execution_v1 as execution
import bracket_closure_rows_v1 as rows_contract
import circuit_campaign_runtime as campaign
import freeze_bracket_closure_rows_v1 as row_freezer
import run_bracket_closure_execution_v1 as subject
from bracket_closure_masks_v1 import BracketDomain
from bracket_closure_tensor_v1 import PRODUCTION_STORED_VALUES, TARGET_SITE
from test_bracket_closure_execution_v1 import _known_score_stats, _masks
from test_bracket_closure_rows_v1 import REGISTRY, _pool


def _registry_payload():
    return {
        "families": [{
            "name": family.name, "opener_ids": list(family.opener_ids),
            "closer_ids": list(family.closer_ids),
        } for family in REGISTRY.families],
        "quote_control_ids": list(REGISTRY.quote_control_ids),
        "punctuation_control_ids": list(REGISTRY.punctuation_control_ids),
    }


def _payload(tmp_path: Path):
    sources = {path: "a" * 64 for path in lifecycle.SOURCE_CLOSURE}
    roles = {
        role: {
            "filename": f"{role}.pt", "file_sha256": "b" * 64,
            "rows_sha256": "c" * 64, "records_sha256": "d" * 64,
            "support_sha256": "e" * 64, "document_ids_sha256": "f" * 64,
        } for role in ("fit", "select", "ood")
    }
    permutation = torch.roll(torch.arange(128), -1).contiguous()
    value = {
        "schema": subject.AUTHORITY_SCHEMA,
        "status": "frozen_before_any_bracket_model_forward", "outcome_access": False,
        "source_commit": "1" * 40, "source_hashes": sources,
        "row_receipt": {"path": str(tmp_path / "rows.json"), "sha256": "2" * 64},
        "row_cache": str(tmp_path / "cache"), "roles": roles,
        "delimiter_registry": _registry_payload(),
        "model": {
            "snapshot": str(subject.facade.DEFAULT_SNAPSHOT),
            "config_sha256": subject.facade.CONFIG_SHA256,
            "weights_sha256": subject.facade.WEIGHTS_SHA256,
        },
        "derangement": {
            "path": str(tmp_path / "derangement.pt"), "file_sha256": "3" * 64,
            "tensor_sha256": canary.tensor_sha256(permutation),
        },
        "programs": [{
            "arm": arm, "state_sha256": format(index + 10, "064x"),
            "stored_values": PRODUCTION_STORED_VALUES, "native_calls_per_forward": 0,
            "token_table_values": 0, "total_input_support": True,
        } for index, arm in enumerate(canary.ARM_NAMES[1:])],
        "outputs": {key: str(tmp_path / f"{key}.json") for key in (
            "result", "receipt", "failure", "lock",
        )},
    }
    return value


def _audit(authority, authority_sha="4" * 64):
    return {
        "schema": subject.AUDIT_SCHEMA, "status": "GO", "outcome_access": False,
        "authority_sha256": authority_sha, "source_commit": authority["source_commit"],
        "source_hashes": authority["source_hashes"], "tests_passed": 1,
        "reviewer": "independent",
    }


def test_authority_and_independent_audit_are_external_exact(monkeypatch, tmp_path) -> None:
    value = _payload(tmp_path)
    authority = subject.validate_authority_payload(value, audit_sha256="5" * 64)
    assert authority.authorized_for_forward
    assert authority.derangement_sha256 == value["derangement"]["tensor_sha256"]
    audit = _audit(value)
    subject.validate_independent_audit(
        audit, authority_sha256="4" * 64, authority_payload=value,
    )
    audit["outcome_access"] = True
    with pytest.raises(RuntimeError, match="outcome-blind GO"):
        subject.validate_independent_audit(
            audit, authority_sha256="4" * 64, authority_payload=value,
        )
    value["programs"][0]["extra"] = 1
    with pytest.raises(RuntimeError, match="program binding schema"):
        subject.validate_authority_payload(value, audit_sha256="5" * 64)


def _install_roles(tmp_path: Path, authority_payload):
    rows, records = _pool()
    frozen = rows_contract.allocate_roles(
        rows, records, REGISTRY, rows_contract.PriorExclusions.empty(), seed="adapter-test",
    )
    cache = tmp_path / "cache"; cache.mkdir()
    receipt_entries = {}
    for role in frozen:
        path = cache / f"{role.role.value}.pt"
        torch.save(row_freezer._role_payload(role), path)
        record_payload = row_freezer._role_payload(role)["records"]
        binding = {
            "filename": path.name, "file_sha256": lifecycle.file_sha256(path),
            "rows_sha256": rows_contract.tensor_sha256(role.rows),
            "records_sha256": hashlib.sha256(json.dumps(
                record_payload, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(),
            "support_sha256": canary.support_sha256(role.rows, role.masks),
            "document_ids_sha256": subject._logical_sha([
                record.document_id for record in role.records
            ]),
        }
        authority_payload["roles"][role.role.value] = binding
        receipt_entries[role.role.value] = {
            key: binding[key] for key in (
                "filename", "file_sha256", "rows_sha256", "records_sha256",
            )
        }
    receipt = {
        "schema": "bracket_closure_rows_v1_receipt",
        "status": "frozen_before_any_model_forward_receipt_last",
        "authority_sha256": "1" * 64, "audit_sha256": "2" * 64,
        "source_commit": "3" * 40, "source_hashes": {},
        "candidate_sha256": "4" * 64, "candidate_source_identity_sha256": "5" * 64,
        "delimiter_registry_sha256": rows_contract.registry_sha256(REGISTRY),
        "historical_registry_hashes": {}, "historical_exclusion_counts": {},
        "entries": receipt_entries, "outcome_access": False,
    }
    receipt_path = tmp_path / "rows.json"
    receipt_path.write_text(json.dumps(receipt))
    authority_payload["row_receipt"] = {
        "path": str(receipt_path), "sha256": lifecycle.file_sha256(receipt_path),
    }
    return frozen


def test_loader_replays_all_role_metadata_but_returns_only_select_ood(tmp_path) -> None:
    payload = _payload(tmp_path)
    frozen = _install_roles(tmp_path, payload)
    authority = subject.validate_authority_payload(payload, audit_sha256="6" * 64)
    roles = subject.load_bound_roles(payload, authority)
    assert tuple(role.role for role in roles) == ("select", "ood")
    assert roles[0].document_ids == tuple(
        record.document_id for record in frozen[1].records
    )
    fit = tmp_path / "cache/fit.pt"
    forged = torch.load(fit, map_location="cpu", weights_only=True)
    forged["records"][0]["document_id"] = frozen[1].records[0].document_id
    torch.save(forged, fit)
    payload["roles"]["fit"]["file_sha256"] = lifecycle.file_sha256(fit)
    with pytest.raises(RuntimeError, match="parent join|semantic binding|overlap"):
        subject.load_bound_roles(payload, authority)


def _closure(arm: str, documents: int):
    sites = []
    for site in range(18):
        replaced = arm != "native" and site == TARGET_SITE
        sites.append(campaign.SiteCallLedger(
            site, 0 if replaced else 1, 1 if replaced else 0, 1, 0,
        ))
    return campaign.ForwardClosure(
        canary.SCHEMA, arm,
        campaign.ArmKind.NATIVE if arm == "native" else campaign.ArmKind.CANDIDATE,
        1, 1, 1, documents, tuple(sites), True, True,
    )


def test_result_recomputes_score_top1_and_call_semantics(tmp_path) -> None:
    payload = _payload(tmp_path)
    authority = subject.validate_authority_payload(payload, audit_sha256="6" * 64)
    select, ood = _known_score_stats("select"), _known_score_stats("ood")
    closures = {
        (role, arm): (_closure(arm, 40),)
        for role in execution.ROLE_ORDER for arm in canary.ARM_NAMES
    }
    result = subject.build_result_payload("7" * 64, select, ood, closures, authority)
    expected_roles = (
        execution.RoleMaterialization(
            "select", torch.zeros(40, 257, dtype=torch.long), select.document_ids, _masks(40),
        ),
        execution.RoleMaterialization(
            "ood", torch.zeros(40, 257, dtype=torch.long), ood.document_ids, _masks(40),
        ),
    )
    ledger = subject.closure_summary(closures, {"select": 40, "ood": 40})
    subject.validate_result_payload(
        result, authority_sha256="7" * 64, authority=authority,
        expected_roles=expected_roles, expected_call_ledger=ledger,
    )
    result["raw_statistics"]["ood"]["ce_sums"][2][0][0] += 1.0
    with pytest.raises(RuntimeError, match="score/top1 does not replay"):
        subject.validate_result_payload(
            result, authority_sha256="7" * 64, authority=authority,
            expected_roles=expected_roles, expected_call_ledger=ledger,
        )


def test_concrete_transaction_hands_only_select_ood_to_forward_owner(monkeypatch, tmp_path) -> None:
    payload = _payload(tmp_path)
    authority_path, audit_path = tmp_path / "authority.json", tmp_path / "audit.json"
    authority_path.write_text(json.dumps(payload))
    authority_sha = lifecycle.file_sha256(authority_path)
    audit_path.write_text(json.dumps(_audit(payload, authority_sha)))
    monkeypatch.setattr(lifecycle, "verify_source_binding", lambda _authority: None)
    guards = []
    monkeypatch.setattr(subject, "_guard_inputs", lambda *_args: guards.append("guard"))
    select_stats, ood_stats = _known_score_stats("select"), _known_score_stats("ood")
    roles = (
        execution.RoleMaterialization(
            "select", torch.zeros(40, 257, dtype=torch.long),
            select_stats.document_ids, _masks(40),
        ),
        execution.RoleMaterialization(
            "ood", torch.zeros(40, 257, dtype=torch.long),
            ood_stats.document_ids, _masks(40),
        ),
    )
    monkeypatch.setattr(subject, "load_bound_roles", lambda *_args: roles)
    permutation = torch.roll(torch.arange(128), -1).contiguous()
    monkeypatch.setattr(subject, "load_derangement", lambda *_args: permutation)
    monkeypatch.setattr(subject.facade, "load_bilin18", lambda **_kwargs: (
        object(), SimpleNamespace(
            config_sha256=subject.facade.CONFIG_SHA256,
            weights_sha256=subject.facade.WEIGHTS_SHA256,
        ),
    ))
    closures = {
        (role, arm): (_closure(arm, 40),)
        for role in execution.ROLE_ORDER for arm in canary.ARM_NAMES
    }
    forwarded = []
    def execute(_model, actual_roles, _permutation, _authority, *, source_guard):
        forwarded.append(tuple(role.role for role in actual_roles)); source_guard()
        return select_stats, ood_stats, closures
    monkeypatch.setattr(execution, "execute_loaded_roles", execute)
    monkeypatch.setattr(subject, "build_result_payload", lambda *_args: {
        "schema": subject.RESULT_SCHEMA, "promoted": False,
    })
    monkeypatch.setattr(subject, "validate_result_payload", lambda *_args, **_kwargs: None)
    result = subject.run(authority_path, audit_path)
    assert result["promoted"] is False and forwarded == [("select", "ood")]
    assert guards
    assert Path(payload["outputs"]["receipt"]).is_file()
    assert not Path(payload["outputs"]["failure"]).exists()


def test_module_has_no_authority_audit_or_row_mint_and_no_import_io() -> None:
    source = Path(subject.__file__).read_text()
    assert "torch.save" not in source
    assert "write_authority" not in source and "write_independent_audit" not in source
    assert "load_bilin18" in source and "load_bound_roles" in source
    assert lifecycle.SOURCE_CLOSURE[-1].endswith("test_run_bracket_closure_execution_v1.py")

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
        "row_authority": {"path": str(tmp_path / "row-authority.json"), "sha256": "6" * 64},
        "row_audit": {"path": str(tmp_path / "row-audit.json"), "sha256": "7" * 64},
        "row_cache": str(tmp_path / "cache"), "roles": roles,
        "delimiter_registry": _registry_payload(),
        "model": {
            "snapshot": str(subject.facade.DEFAULT_SNAPSHOT),
            "config_sha256": subject.facade.CONFIG_SHA256,
            "weights_sha256": subject.facade.WEIGHTS_SHA256,
            "state_sha256": "8" * 64,
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
            "result", "receipt", "failure", "terminal", "lock",
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
    value = _payload(tmp_path)
    value["roles"]["fit"]["filename"] = "../select.pt"
    with pytest.raises(RuntimeError, match="role bindings"):
        subject.validate_authority_payload(value, audit_sha256="5" * 64)


def _install_roles(tmp_path: Path, authority_payload, monkeypatch):
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
    candidate = tmp_path / "candidate.pt"; candidate.write_bytes(b"candidate")
    history = tmp_path / "history.json"
    history_payload = {"records": [{
        "document_id": "old-doc", "source_file": "old.py",
        "source_blob_sha256": "1" * 64, "normalized_python_sha256": "2" * 64,
        "row_sha256": "3" * 64, "prefix32_sha256": "4" * 64,
    }]}
    history.write_text(json.dumps(history_payload))
    row_sources = {path: "9" * 64 for path in row_freezer.SOURCE_CLOSURE}
    monkeypatch.setattr(row_freezer, "source_closure", lambda _commit: row_sources)
    row_authority = {
        "schema": "bracket_closure_rows_v1_authority", "source_commit": "3" * 40,
        "source_hashes": row_sources, "candidate_path": str(candidate),
        "candidate_sha256": lifecycle.file_sha256(candidate),
        "candidate_source_identity_sha256": "5" * 64,
        "delimiter_registry_sha256": rows_contract.registry_sha256(REGISTRY),
        "historical_registries": [{
            "path": str(history), "sha256": lifecycle.file_sha256(history),
        }],
        "allocation_seed": "adapter-test", "cache_path": str(cache),
        "receipt_path": str(tmp_path / "rows.json"),
        "failure_path": str(tmp_path / "row-failure.json"),
        "lock_path": str(tmp_path / "row-lock"), "outcome_access": False,
    }
    row_authority_path = tmp_path / "row-authority.json"
    row_authority_path.write_text(json.dumps(row_authority))
    row_authority_sha = lifecycle.file_sha256(row_authority_path)
    row_audit = {
        "schema": "bracket_closure_rows_v1_independent_audit", "status": "GO",
        "outcome_access": False, "authority_sha256": row_authority_sha,
        "audited_source_commit": row_authority["source_commit"],
        "audited_source_hashes": row_sources, "tests_passed": 1, "reviewer": "test",
    }
    row_audit_path = tmp_path / "row-audit.json"
    row_audit_path.write_text(json.dumps(row_audit))
    prior = rows_contract.historical_exclusions((history_payload,))
    receipt = {
        "schema": "bracket_closure_rows_v1_receipt",
        "status": "frozen_before_any_model_forward_receipt_last",
        "authority_sha256": row_authority_sha,
        "audit_sha256": lifecycle.file_sha256(row_audit_path),
        "source_commit": row_authority["source_commit"], "source_hashes": row_sources,
        "candidate_sha256": row_authority["candidate_sha256"],
        "candidate_source_identity_sha256": "5" * 64,
        "delimiter_registry_sha256": rows_contract.registry_sha256(REGISTRY),
        "historical_registry_hashes": {str(history): lifecycle.file_sha256(history)},
        "historical_exclusion_counts": {name: len(getattr(prior, name)) for name in (
            "documents", "source_files", "source_blobs", "normalized_python",
            "row_sha256", "prefix32_sha256",
        )},
        "entries": receipt_entries, "outcome_access": False,
    }
    receipt_path = tmp_path / "rows.json"
    receipt_path.write_text(json.dumps(receipt))
    authority_payload["row_receipt"] = {
        "path": str(receipt_path), "sha256": lifecycle.file_sha256(receipt_path),
    }
    authority_payload["row_authority"] = {
        "path": str(row_authority_path), "sha256": row_authority_sha,
    }
    authority_payload["row_audit"] = {
        "path": str(row_audit_path), "sha256": lifecycle.file_sha256(row_audit_path),
    }
    return frozen


def test_loader_replays_all_role_metadata_but_returns_only_select_ood(
    tmp_path, monkeypatch,
) -> None:
    payload = _payload(tmp_path)
    frozen = _install_roles(tmp_path, payload, monkeypatch)
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


@pytest.mark.parametrize("field", (
    "authority_sha256", "audit_sha256", "source_commit", "source_hashes",
    "historical_exclusion_counts",
))
def test_row_receipt_lineage_fields_are_semantically_replayed(
    tmp_path, monkeypatch, field,
) -> None:
    payload = _payload(tmp_path)
    _install_roles(tmp_path, payload, monkeypatch)
    receipt_path = Path(payload["row_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = {} if field in {"source_hashes", "historical_exclusion_counts"} else "0" * (
        40 if field == "source_commit" else 64
    )
    receipt_path.write_text(json.dumps(receipt))
    payload["row_receipt"]["sha256"] = lifecycle.file_sha256(receipt_path)
    authority = subject.validate_authority_payload(payload, audit_sha256="6" * 64)
    with pytest.raises(RuntimeError, match="authority/source join|historical exclusion"):
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
    result = subject.build_result_payload(
        "7" * 64, select, ood, closures, authority, live_model_state_sha256="8" * 64,
    )
    expected_roles = (
        execution.RoleMaterialization(
            "select", torch.zeros(40, 257, dtype=torch.long), select.document_ids, _masks(40),
        ),
        execution.RoleMaterialization(
            "ood", torch.zeros(40, 257, dtype=torch.long), ood.document_ids, _masks(40),
        ),
    )
    ledger = subject.closure_summary(closures, {"select": 40, "ood": 40})
    assert len(ledger["select"]["native"]["sites"]) == 18
    assert ledger["select"]["native"]["outer_returns"] == 1
    assert ledger["ood"][canary.ARM_NAMES[1]]["sites"][TARGET_SITE] == {
        "site": TARGET_SITE, "native_attention_calls": 0,
        "replacement_attention_calls": 1, "native_mlp_calls": 1,
        "replacement_mlp_calls": 0,
    }
    subject.validate_result_payload(
        result, authority_sha256="7" * 64, authority=authority,
        expected_roles=expected_roles, expected_call_ledger=ledger,
        expected_live_model_state_sha256="8" * 64,
    )
    result["raw_statistics"]["ood"]["ce_sums"][2][0][0] += 1.0
    with pytest.raises(RuntimeError, match="score/top1 does not replay"):
        subject.validate_result_payload(
            result, authority_sha256="7" * 64, authority=authority,
            expected_roles=expected_roles, expected_call_ledger=ledger,
            expected_live_model_state_sha256="8" * 64,
        )


def test_live_model_state_hash_covers_parameters_buffers_and_scalar_shape() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
            self.register_buffer("scalar", torch.tensor(2.0))
    model = Tiny()
    first = subject.model_state_sha256(model)
    assert first == subject.model_state_sha256(model)
    with torch.no_grad(): model.weight[0, 0] += 1
    second = subject.model_state_sha256(model)
    assert second != first
    model.scalar += 1
    assert subject.model_state_sha256(model) != second


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
    monkeypatch.setattr(subject, "model_state_sha256", lambda _model: "8" * 64)
    closures = {
        (role, arm): (_closure(arm, 40),)
        for role in execution.ROLE_ORDER for arm in canary.ARM_NAMES
    }
    forwarded = []
    def execute(_model, actual_roles, _permutation, _authority, *, source_guard):
        forwarded.append(tuple(role.role for role in actual_roles)); source_guard()
        return select_stats, ood_stats, closures
    monkeypatch.setattr(execution, "execute_loaded_roles", execute)
    monkeypatch.setattr(subject, "build_result_payload", lambda *_args, **_kwargs: {
        "schema": subject.RESULT_SCHEMA, "promoted": False,
    })
    monkeypatch.setattr(subject, "validate_result_payload", lambda *_args, **_kwargs: None)
    result = subject.run(authority_path, audit_path)
    assert result["promoted"] is False and forwarded == [("select", "ood")]
    assert guards
    assert Path(payload["outputs"]["receipt"]).is_file()
    assert Path(payload["outputs"]["terminal"]).is_file()
    assert not Path(payload["outputs"]["failure"]).exists()


def test_module_has_no_authority_audit_or_row_mint_and_no_import_io() -> None:
    source = Path(subject.__file__).read_text()
    assert "torch.save" not in source
    assert "write_authority" not in source and "write_independent_audit" not in source
    assert "load_bilin18" in source and "load_bound_roles" in source
    assert "basis_aligned/polynomial_causal/test_run_bracket_closure_execution_v1.py" in (
        lifecycle.SOURCE_CLOSURE
    )

"""Focused CPU tests for canonical bracket semantic-chain publication."""
from __future__ import annotations

import json

import pytest

import circuit_registry_v2 as registry
import register_bracket_l13h8_semantic_chain as publish


def test_record_preserves_exact_final_test_split_verdicts_and_ood_boundary():
    record = publish.build_record()
    claim = record["claims"][-1]
    assert claim["claim_id"] == "pending_opener_state.v29"
    assert claim["status"] == "site_live"
    assert claim["causal_variable"]["id"] == "pending_opener_exact_semantic_source_term"
    assert "selective necessity" in claim["causal_variable"]["operation"]
    assert "selective necessity remains null" in claim["next_missing"]
    assert "OOD" in claim["next_missing"]
    events = {event["event_id"]: event for event in record["evidence_events"]}
    transfer = events["pending_opener.semantic_chain.pair_centered_final_test.transfer.held.v1"]
    removal = events["pending_opener.semantic_chain.pair_centered_final_test.removal.null.v1"]
    assert transfer["verdict"] == "held" and transfer["evaluation_role"] == "held-out FINAL_TEST"
    assert removal["verdict"] == "null" and removal["failure_kind"] == "scientific_null"
    assert transfer["result_artifact_id"] == removal["result_artifact_id"]
    assert "OOD remains unopened" in transfer["notes"]
    registry.validate_v2(record)


def test_all_omitted_result_chain_artifacts_are_hash_bound():
    artifacts = publish._artifacts()
    assert len(artifacts) == 2 * len(publish.STEMS)
    for key in publish.STEMS:
        assert artifacts[f"bracket_{key}_prior_art"]["kind"] == "preregistration"
        assert artifacts[f"bracket_{key}_result"]["kind"] == "screen_result"
    assert artifacts["bracket_pair_centered_final_result"]["sha256"] == \
        "e64093354428d62eecd268360a79e8ef6549437babdaf9897d853134a44000f6"


def test_artifact_drift_fails_closed(monkeypatch):
    original = publish._sha
    monkeypatch.setattr(
        publish, "_sha",
        lambda path: "0" * 64 if "pair_centered" in path.name else original(path))
    with pytest.raises(publish.PublicationError, match="artifact changed"):
        publish._artifacts()


def test_apply_is_byte_idempotent_without_regeneration(tmp_path, monkeypatch):
    source = registry.circuit_path(publish.TAG)
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    monkeypatch.setattr(publish.registry, "circuit_path", lambda _tag: target)
    monkeypatch.setattr(publish.registry, "CIRCUITS", tmp_path)
    publish.apply(regenerate=False)
    first = target.read_bytes()
    publish.apply(regenerate=False)
    assert target.read_bytes() == first
    value = json.loads(first)
    assert value["claims"][-1]["claim_id"] == "pending_opener_state.v29"
    assert len({event["event_id"] for event in value["evidence_events"]}) == \
        len(value["evidence_events"])

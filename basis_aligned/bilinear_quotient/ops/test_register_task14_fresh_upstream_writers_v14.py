#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_fresh_upstream_writers_v14 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan: dict, fragment: str) -> dict:
    return next(x for x in plan["events"] if fragment in x["event_id"])


def metrics(item: dict) -> dict:
    return {x["name"]: x["estimate"] for x in item["metrics"]}


def test_separate_outcomes_and_exact_scope() -> None:
    plan = publish.build_plan()
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v14"
    assert plan["claim_revision"]["supersedes"] == "grammatical_subject_number.v13"
    expected = {
        "v1_instrument": ("invalid", "implementation_failure"),
        "v2_instrument": ("held", None),
        "embedding_skip_family": ("null", "scientific_null"),
        "earlier_attention_family": ("null", "scientific_null"),
        "earlier_MLP_family": ("held", None),
        "distributed_families": ("null", "scientific_null"),
        "interaction_needed": ("null", "scientific_null"),
        "number_specificity": ("held", None),
        "lexical_collateral": ("null", "scientific_null"),
    }
    assert len(plan["events"]) == len(expected)
    for fragment, verdict in expected.items():
        got = event(plan, fragment)
        assert (got["verdict"], got["failure_kind"]) == verdict
        assert got["evaluation_role"] == "FRESH_LICENSED_HOLDOUT"
    family = plan["claim_revision"]["counterfactual_families"][-1]
    assert family["holds_fixed"] == [
        "licensed HOLDOUT text", "recipient subject score p_8",
        "recipient cached layer-0 value branch", "native non-subject source-term complement",
    ]
    missing = plan["claim_revision"]["next_missing"]
    for phrase in ("broad writer family only", "necessity", "downstream readers", "individual MLP layer"):
        assert phrase in missing


def test_decisive_metrics_are_recomputed() -> None:
    plan = publish.build_plan()
    e = metrics(event(plan, "embedding_skip_family"))
    a = metrics(event(plan, "earlier_attention_family"))
    m = metrics(event(plan, "earlier_MLP_family"))
    interaction = metrics(event(plan, "interaction_needed"))
    specificity = metrics(event(plan, "number_specificity"))
    assert e["E_over_all_donor_margin_range"] == pytest.approx([0.0969838556, 0.1607042120])
    assert a["A_over_all_donor_margin_range"] == pytest.approx([0.0135098410, 0.0661074473])
    assert m["M_over_all_donor_margin_range"] == pytest.approx([0.7734154122, 0.9129719682])
    assert m["M_mean_margin_range"] == pytest.approx([0.2322033644, 0.5335423946])
    assert m["minimum_M_positive_row_fraction"] == 0.75
    assert interaction["total_interaction_over_all_donor_margin_range"] == pytest.approx([-0.0739603335, 0.0826057817])
    assert specificity["maximum_lexical_margin_ratio"] == pytest.approx(0.1151079915)
    assert specificity["maximum_lexical_CE_ratio"] == pytest.approx(0.0284711677)


def test_v1_is_non_evidence_and_v2_supersedes_it() -> None:
    plan = publish.build_plan()
    invalid = event(plan, "v1_instrument")
    valid = event(plan, "v2_instrument")
    assert "not evidence" in invalid["notes"]
    assert valid["supersedes_event_id"] == invalid["event_id"]
    assert metrics(valid)["uncorrected_state_max_absolute_error_diagnostic"] == 0.00146484375


def test_exact_v13_prefix_and_idempotence(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
    before = json.loads(RECORD.read_text())
    if before["claims"][-1]["claim_id"] == publish.NEW_CLAIM:
        expected_v13_claims = before["claims"][:-1]
        expected_v13_event_ids = expected_v13_claims[-1]["evidence_event_ids"]
    else:
        expected_v13_claims = before["claims"]
        expected_v13_event_ids = before["claims"][-1]["evidence_event_ids"]
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    copied = circuits / RECORD.name
    shutil.copy2(RECORD, copied)
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    publish.apply_plan(plan, regenerate=False)
    first = copied.read_bytes()
    publish.apply_plan(plan, regenerate=False)
    assert copied.read_bytes() == first
    after = json.loads(first)
    assert after["claims"][:-1] == expected_v13_claims
    assert after["claims"][-1]["evidence_event_ids"] == (
        expected_v13_event_ids + [x["event_id"] for x in plan["events"]]
    )
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_result_hash_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_upstream_writers_v2_result"]
    changed["task14_upstream_writers_v2_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()

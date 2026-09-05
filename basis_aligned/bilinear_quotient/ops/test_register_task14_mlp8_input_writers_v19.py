#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_mlp8_input_writers_v19 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan: dict, fragment: str) -> dict:
    return next(x for x in plan["events"] if fragment in x["event_id"])


def metrics(item: dict) -> dict:
    return {x["name"]: x["estimate"] for x in item["metrics"]}


def test_all_registered_outcomes_are_separate_and_preserved() -> None:
    plan = publish.build_plan()
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v19"
    expected = {
        "instrument": "held", ".M_dominant": "null", ".E_dominant": "null",
        ".A_dominant": "null", "distributed_additive": "null",
        "source_interaction_needed": "held", "direction_stable": "null",
        "direction_switch": "null", "number_specificity": "held",
        "lexical_collateral": "null",
    }
    assert len(plan["events"]) == len(expected)
    for fragment, verdict in expected.items():
        got = event(plan, fragment)
        assert got["verdict"] == verdict
        assert got["evaluation_role"] == "FRESH_LICENSED_HOLDOUT"


def test_quantitative_ranges_are_recomputed() -> None:
    plan = publish.build_plan()
    assert metrics(event(plan, ".M_dominant"))["M_signed_recovery_range"] == pytest.approx([0.2475741384, 1.0147995123])
    assert metrics(event(plan, ".E_dominant"))["E_signed_recovery_range"] == pytest.approx([-0.0391342501, 0.4347116594])
    assert metrics(event(plan, ".A_dominant"))["A_signed_recovery_range"] == pytest.approx([-0.0712951991, 0.3705033663])
    interaction = metrics(event(plan, "source_interaction_needed"))
    assert interaction["cross_interaction_fraction_range"] == pytest.approx([0.0468226311, 0.0812823205])
    assert interaction["quadratic_interaction_fraction_range"] == pytest.approx([0.4198968647, 0.5261910519])
    assert interaction["full_interaction_fraction_range"] == pytest.approx([0.5807222930, 0.8587656942])
    assert metrics(event(plan, "number_specificity"))["maximum_lexical_ratio"] == pytest.approx(0.1876561331)


def test_null_direction_claims_do_not_imply_stability() -> None:
    plan = publish.build_plan()
    stable = event(plan, "direction_stable")
    switched = event(plan, "direction_switch")
    expected = {
        "plural_to_singular": {"cross": None, "full": None, "quadratic": None},
        "singular_to_plural": {"cross": None, "full": None, "quadratic": None},
    }
    assert metrics(stable)["direction_component_winners"] == expected
    assert metrics(switched)["direction_component_winners"] == expected
    assert "distinct from evidence for stable identity" in switched["notes"]


def test_scope_and_operational_family_limit_are_explicit() -> None:
    plan = publish.build_plan()
    missing = plan["claim_revision"]["next_missing"]
    for phrase in ("operational native writer families", "not unique semantic units", "Individual writers", "OOD replication", "downstream readers", "necessity"):
        assert phrase in missing


def test_exact_v18_prefix_and_idempotence(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
    before = json.loads(RECORD.read_text())
    v19_present = any(c["claim_id"] == publish.NEW_CLAIM for c in before["claims"])
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
    if v19_present:
        assert after == before
    else:
        assert after["claims"][:-1] == before["claims"]
        assert after["claims"][-1]["evidence_event_ids"] == (
            before["claims"][-1]["evidence_event_ids"] + [x["event_id"] for x in plan["events"]])
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_result_hash_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_mlp8_input_writers_result"]
    changed["task14_mlp8_input_writers_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()

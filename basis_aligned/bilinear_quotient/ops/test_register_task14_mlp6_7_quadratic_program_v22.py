#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil
import pytest
import register_task14_mlp6_7_quadratic_program_v22 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan, fragment):
    return next(x for x in plan["events"] if fragment in x["event_id"])


def metrics(item):
    return {x["name"]: x["estimate"] for x in item["metrics"]}


def test_all_registered_outcomes_are_separate_and_scoped():
    plan = publish.build_plan()
    assert len(plan["events"]) == 18
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v22"
    verdicts = {x["event_id"]: x["verdict"] for x in plan["events"]}
    assert verdicts[next(k for k in verdicts if "gain.extrapolation" in k)] == "held"
    assert verdicts[next(k for k in verdicts if "split.number_specificity" in k)] == "null"
    assert verdicts[next(k for k in verdicts if "tangent.endpoint_local" in k)] == "null"


def test_quantitative_program_bounds_are_recomputed():
    plan = publish.build_plan()
    tangent = metrics(event(plan, "tangent.midpoint_quadratic"))
    assert tangent["minimum_midpoint_cosine"] == pytest.approx(0.9999270775281021)
    assert tangent["maximum_midpoint_relative_error"] == pytest.approx(0.013931754446863103)
    head = metrics(event(plan, "gain.quadratic_head_prediction"))
    assert head["minimum_predicted_cosine"] == pytest.approx(0.9988964593582645)
    assert head["maximum_predicted_relative_error"] == pytest.approx(0.04722858481481618)
    assert metrics(event(plan, "gain.task_manipulation"))["task_recovery_range"] == pytest.approx(
        [0.9599328030799572, 1.0352747326652767])


def test_next_missing_preserves_honest_boundary():
    missing = publish.build_plan()["claim_revision"]["next_missing"]
    for phrase in ("must not be repeated", "quadratic readout", "lexical collateral",
                   "OOD/independent-text", "coefficient sharing", "literal program pricing"):
        assert phrase in missing


def test_exact_v21_prefix_and_idempotence(tmp_path, monkeypatch):
    plan = publish.build_plan(); before = json.loads(RECORD.read_text())
    v22_present = any(c["claim_id"] == publish.NEW_CLAIM for c in before["claims"])
    circuits = tmp_path / "circuits"; circuits.mkdir()
    copied = circuits / RECORD.name; shutil.copy2(RECORD, copied)
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    publish.apply_plan(plan, regenerate=False); first = copied.read_bytes()
    publish.apply_plan(plan, regenerate=False)
    assert copied.read_bytes() == first
    after = json.loads(first)
    if v22_present:
        assert after == before
    else:
        assert after["claims"][:-1] == before["claims"]
        assert after["claims"][-1]["evidence_event_ids"] == (
            before["claims"][-1]["evidence_event_ids"] + [x["event_id"] for x in plan["events"]])
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_result_hash_mutation_is_rejected(monkeypatch):
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_mlp6_7_gain_result"]
    changed["task14_mlp6_7_gain_result"] = (path, "0"*64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()

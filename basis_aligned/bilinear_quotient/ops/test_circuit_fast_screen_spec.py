from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import circuit_battery_integration_contract as battery
import circuit_experiment_spec as framework
import circuit_fast_screen_spec as screen


SOURCE = Path(__file__).with_name("circuit_fast_screen_spec.py")


def task_spec(*, c_answer_changes: bool = True) -> battery.BatteryTaskSpec:
    return battery.BatteryTaskSpec(
        task_id="fixture.linked_behavior",
        generator_role="fixture_generator",
        answer_role="paired_answer_token",
        transforms=(
            battery.TransformSpec("A1", "answer_change_one", True, "toward_donor"),
            battery.TransformSpec("A2", "answer_change_two", True, "toward_donor"),
            battery.TransformSpec("P", "state_preserving", False, "invariant"),
            battery.TransformSpec(
                "C", "unrelated_answer_change", c_answer_changes, "registered_active"
            ),
        ),
    )


def authority(groups: int = 1) -> list[dict[str, object]]:
    rows = []
    for group in range(groups):
        for transform_index, transform in enumerate(screen.TRANSFORMS):
            changes = transform in ("A1", "A2", "C")
            token_a = 10 + 2 * transform_index
            token_b = token_a + 1
            rows.append({
                "row_id": f"FIT:g{group:02d}:{transform}",
                "group_id": f"FIT:g{group:02d}",
                "split": "FIT",
                "task_id": "fixture.linked_behavior",
                "transform_id": transform,
                "answer_changes": changes,
                "base_ids": [100 + group, 200 + transform_index, 300],
                "donor_ids": [400 + group, 500 + transform_index, 600],
                "base_answer_id": token_a,
                "base_foil_id": token_b,
                "donor_answer_id": token_b if changes else token_a,
                "donor_foil_id": token_a if changes else token_b,
                "base_semantic_position": 1,
                "donor_semantic_position": 1,
            })
    return rows


def exact_max_price(groups: int = 1) -> battery.ExactPhasePrice:
    evaluations = 264 * groups
    return battery.ExactPhasePrice(
        phase="FIT", forward_calls=264, example_evaluations=evaluations,
        backward_calls=0, model_updates=0, evidence_bytes=8 * evaluations,
    )


def valid_spec(rows: list[dict[str, object]]) -> screen.CircuitFastScreenSpec:
    return screen.CircuitFastScreenSpec(
        experiment_id="fixture-fast-screen",
        hypothesis=screen.CandidateHypothesis(
            behavior="fixture.linked_behavior",
            answer_score=screen.ANSWER_SCORE,
            information_read="the grammatical state at the marked source token",
            proposed_operation="carry the marked state through the residual stream",
            proposed_write="a donor-directed answer-relevant state",
            candidate_sites=screen.CEILING_SITE_IDS,
            alternative_explanation="a lexical token-identity shortcut",
            circuit_prediction="a selective module or residual state moves the paired score",
            opposing_null_prediction="no selective causal site moves the paired score",
        ),
        task=task_spec(),
        authority_sha256=framework.canonical_sha256(rows),
        expected_fit_rows=len(rows),
        batch_size=32,
        semantic_position=screen.SemanticPositionSpec(
            role="marked_source_state",
            recipient_field="base_semantic_position",
            donor_field="donor_semantic_position",
        ),
        fields=screen.AuthorityFieldSpec(),
        bars=screen.ScreenBars(),
        declared_max_price=exact_max_price(len(rows) // 4),
    )


def attention_scores(default_target: float = 0.1) -> dict[str, dict[str, float]]:
    return {
        site_id: {
            "a1_mean_recovery": default_target,
            "a2_mean_recovery": default_target,
            "a1_direction_fraction": 0.9,
            "a2_direction_fraction": 0.9,
            "p_invariance_effect": 0.1,
            "c_absolute_recovery": 0.1,
        }
        for site_id in screen.ATTENTION_SITE_IDS
    }


def test_compile_exact_native_and_55_site_ceiling_manifest() -> None:
    rows = authority()
    compiled = screen.compile_screen(valid_spec(rows), rows)
    assert compiled["conditional_head_plan"]["status"] == "pending"
    assert len(screen.RESIDUAL_SITE_IDS) == 19
    assert len(screen.MODULE_SITE_IDS) == 36
    assert len(screen.CEILING_SITE_IDS) == 55
    assert len(compiled["call_manifest"]) == 228
    assert compiled["price"] == {
        "phase": "FIT", "forward_calls": 228, "example_evaluations": 228,
        "backward_calls": 0, "model_updates": 0, "evidence_bytes": 1824,
    }
    assert compiled["max_price"] == {
        "phase": "FIT", "forward_calls": 264, "example_evaluations": 264,
        "backward_calls": 0, "model_updates": 0, "evidence_bytes": 2112,
    }
    native = compiled["call_manifest"][:8]
    assert [(call["side"], call["transform_id"]) for call in native] == [
        (side, transform) for side in ("base", "donor")
        for transform in screen.TRANSFORMS
    ]
    ceiling = compiled["call_manifest"][8:]
    assert [call["site"]["site_id"] for call in ceiling[::4]] == list(
        screen.CEILING_SITE_IDS
    )
    assert all(call["intervention"] == {
        "direction": "donor_to_recipient",
        "scope": "single_semantic_position",
        "value": "exact_replace",
        "recipient_sequence_role": "base",
        "donor_sequence_role": "donor",
        "recipient_sequence_field": "base_ids",
        "donor_sequence_field": "donor_ids",
    } for call in ceiling)
    assert ceiling[0]["semantic_bindings"] == [{
        "row_id": "FIT:g00:A1", "semantic_role": "marked_source_state",
        "recipient_position": 1, "donor_position": 1,
    }]
    assert compiled["terminal_schema"] == screen.TERMINAL_SCHEMA
    assert compiled["score_contract"]["bars"] == asdict(screen.kernel.FIXED_BARS)
    assert compiled["score_contract"]["family_roles"]["C"] == \
        "answer_changing_unrelated_behavior_control"
    screen.validate_compiled_screen(valid_spec(rows), rows, compiled)


def test_same_answer_c_control_is_explicitly_typed_in_compiled_contract() -> None:
    rows = authority()
    for row in rows:
        if row["transform_id"] == "C":
            row["answer_changes"] = False
            row["donor_answer_id"] = row["base_answer_id"]
            row["donor_foil_id"] = row["base_foil_id"]
    spec = replace(
        valid_spec(rows),
        task=task_spec(c_answer_changes=False),
        authority_sha256=framework.canonical_sha256(rows),
    )
    compiled = screen.compile_screen(spec, rows)
    assert compiled["score_contract"]["family_roles"]["C"] == \
        "same_answer_active_negative_control"
    screen.validate_compiled_screen(spec, rows, compiled)


def test_32_linked_panels_have_exact_tiered_price() -> None:
    rows = authority(32)
    compiled = screen.compile_screen(valid_spec(rows), rows)
    assert len(compiled["call_manifest"]) == 228
    assert compiled["price"]["example_evaluations"] == 7_296
    assert compiled["price"]["evidence_bytes"] == 58_368
    assert compiled["max_price"] == {
        "phase": "FIT", "forward_calls": 264, "example_evaluations": 8_448,
        "backward_calls": 0, "model_updates": 0, "evidence_bytes": 67_584,
    }


def test_conditional_head_expansion_selects_one_parent_by_frozen_tie_break() -> None:
    rows = authority()
    scores = attention_scores()
    scores["attn:03"]["a1_mean_recovery"] = 0.75
    scores["attn:03"]["a2_mean_recovery"] = 0.75
    scores["attn:07"]["a1_mean_recovery"] = 0.75
    scores["attn:07"]["a2_mean_recovery"] = 0.75
    compiled = screen.compile_screen(
        valid_spec(rows), rows, attention_module_scores=scores
    )
    plan = compiled["conditional_head_plan"]
    assert plan["status"] == "expanded"
    assert plan["selected_parent_site_id"] == "attn:03"
    head_calls = compiled["call_manifest"][228:]
    assert len(head_calls) == 36
    assert len({call["site"]["site_id"] for call in head_calls}) == 9
    assert {call["site"]["parent_site_id"] for call in head_calls} == {"attn:03"}
    assert compiled["price"] == compiled["max_price"]
    screen.validate_compiled_screen(
        valid_spec(rows), rows, compiled, attention_module_scores=scores
    )


def test_conditional_head_stage_skips_when_no_attention_module_passes() -> None:
    rows = authority()
    compiled = screen.compile_screen(
        valid_spec(rows), rows, attention_module_scores=attention_scores()
    )
    assert compiled["conditional_head_plan"]["status"] == \
        "skipped_no_passing_attention_module"
    assert compiled["conditional_head_plan"]["selected_parent_site_id"] is None
    assert len(compiled["call_manifest"]) == 228


def test_authority_requires_linked_fit_panel_and_frozen_bytes() -> None:
    rows = authority()
    spec = valid_spec(rows)
    with pytest.raises(screen.FastScreenSpecError, match="complete A1/A2/P/C"):
        screen.compile_screen(replace(spec, expected_fit_rows=3), rows[:3])
    mutated = deepcopy(rows)
    mutated[0]["base_ids"][0] += 1
    with pytest.raises(screen.FastScreenSpecError, match="authority digest changed"):
        screen.compile_screen(spec, mutated)
    future = deepcopy(rows)
    future[0]["split"] = "TEST"
    future_spec = replace(spec, authority_sha256=framework.canonical_sha256(future))
    with pytest.raises(screen.FastScreenSpecError, match="outside authority"):
        screen.compile_screen(future_spec, future)


def test_authority_rejects_position_and_joint_candidate_mutations() -> None:
    rows = authority()
    outside = deepcopy(rows)
    outside[0]["base_semantic_position"] = 3
    spec = replace(valid_spec(rows), authority_sha256=framework.canonical_sha256(outside))
    with pytest.raises(screen.FastScreenSpecError, match="outside its sequence"):
        screen.compile_screen(spec, outside)
    mispaired = deepcopy(rows)
    mispaired[0]["donor_foil_id"] = 999
    spec = replace(valid_spec(rows), authority_sha256=framework.canonical_sha256(mispaired))
    with pytest.raises(screen.FastScreenSpecError, match="jointly aligned"):
        screen.compile_screen(spec, mispaired)


def test_hypothesis_site_grid_bars_terminals_and_price_are_frozen() -> None:
    rows = authority()
    spec = valid_spec(rows)
    bad_hypothesis = replace(
        spec.hypothesis, candidate_sites=spec.hypothesis.candidate_sites[:-1]
    )
    with pytest.raises(screen.FastScreenSpecError, match=r"exact 19\+36"):
        screen.compile_screen(replace(spec, hypothesis=bad_hypothesis), rows)
    same_prediction = replace(
        spec.hypothesis,
        opposing_null_prediction=spec.hypothesis.circuit_prediction,
    )
    with pytest.raises(screen.FastScreenSpecError, match="must be distinct"):
        screen.compile_screen(replace(spec, hypothesis=same_prediction), rows)
    with pytest.raises(screen.FastScreenSpecError, match="terminal schema"):
        screen.compile_screen(
            replace(spec, terminals=replace(spec.terminals, allowed=("screen", "null"))), rows
        )
    with pytest.raises(screen.FastScreenSpecError, match="one-position"):
        screen.compile_screen(
            replace(spec, intervention=replace(spec.intervention, value="approximate_replace")),
            rows,
        )
    with pytest.raises(screen.FastScreenSpecError, match="conditional nine-head"):
        screen.compile_screen(
            replace(spec, head_selection=replace(spec.head_selection, tie_break="lexical")),
            rows,
        )
    drifted_bars = replace(
        screen.kernel.FIXED_BARS, minimum_c_capability_accuracy=0.8
    )
    with pytest.raises(screen.FastScreenSpecError, match="FIXED_BARS"):
        screen.compile_screen(replace(spec, bars=drifted_bars), rows)
    wrong_price = replace(
        spec.declared_max_price,
        evidence_bytes=spec.declared_max_price.evidence_bytes + 8,
    )
    with pytest.raises(screen.FastScreenSpecError, match="maximum price differs"):
        screen.compile_screen(replace(spec, declared_max_price=wrong_price), rows)


def test_attention_scores_require_complete_finite_control_bound_inputs() -> None:
    rows = authority()
    spec = valid_spec(rows)
    incomplete = attention_scores()
    incomplete.pop("attn:17")
    with pytest.raises(screen.FastScreenSpecError, match="all 18"):
        screen.compile_screen(spec, rows, attention_module_scores=incomplete)
    malformed = attention_scores()
    malformed["attn:00"]["a1_mean_recovery"] = float("nan")
    with pytest.raises(screen.FastScreenSpecError, match="finite"):
        screen.compile_screen(spec, rows, attention_module_scores=malformed)
    controlled = attention_scores()
    controlled["attn:04"].update(
        a1_mean_recovery=0.9,
        a2_mean_recovery=0.9,
        p_invariance_effect=0.1,
        c_absolute_recovery=0.4,
    )
    compiled = screen.compile_screen(spec, rows, attention_module_scores=controlled)
    assert compiled["conditional_head_plan"]["status"] == \
        "skipped_no_passing_attention_module"

    p_failed = attention_scores()
    p_failed["attn:04"].update(
        a1_mean_recovery=0.9,
        a2_mean_recovery=0.9,
        p_invariance_effect=0.21,
        c_absolute_recovery=0.1,
    )
    compiled = screen.compile_screen(spec, rows, attention_module_scores=p_failed)
    assert compiled["conditional_head_plan"]["status"] == \
        "skipped_no_passing_attention_module"

    direction_failed = attention_scores()
    direction_failed["attn:04"].update(
        a1_mean_recovery=0.9,
        a2_mean_recovery=0.9,
        a1_direction_fraction=0.79,
        a2_direction_fraction=0.9,
        p_invariance_effect=0.1,
        c_absolute_recovery=0.1,
    )
    compiled = screen.compile_screen(spec, rows, attention_module_scores=direction_failed)
    assert compiled["conditional_head_plan"]["status"] == \
        "skipped_no_passing_attention_module"


def test_c_control_declaration_must_match_every_authority_row() -> None:
    rows = authority()
    spec = valid_spec(rows)
    transforms = tuple(
        replace(item, answer_changes=False) if item.transform_id == "C" else item
        for item in spec.task.transforms
    )
    with pytest.raises(screen.FastScreenSpecError, match="answer-change semantics"):
        screen.compile_screen(replace(spec, task=replace(spec.task, transforms=transforms)), rows)


def test_compiled_manifest_mutation_is_rejected_by_exact_recompilation() -> None:
    rows = authority()
    spec = valid_spec(rows)
    compiled = screen.compile_screen(spec, rows)
    mutated = deepcopy(compiled)
    mutated["call_manifest"][8]["semantic_bindings"][0]["donor_position"] = 0
    mutated["call_summary"]["manifest_sha256"] = framework.canonical_sha256(
        mutated["call_manifest"]
    )
    mutated["compiled_sha256"] = framework.canonical_sha256({
        key: value for key, value in mutated.items() if key != "compiled_sha256"
    })
    with pytest.raises(screen.FastScreenSpecError, match="exact deterministic"):
        screen.validate_compiled_screen(spec, rows, mutated)


@pytest.mark.parametrize("record", [
    {"terminal": "screen", "reason": "selective_causal_site", "selected_site_id": "attn:03"},
    {"terminal": "null", "reason": "native_behavior_incapable", "selected_site_id": None},
    {"terminal": "null", "reason": "no_selective_causal_site", "selected_site_id": None},
    {"terminal": "invalid", "reason": "instrument_invalid", "selected_site_id": None},
])
def test_terminal_schema_accepts_only_typed_screen_null_invalid(record) -> None:
    screen.validate_terminal_record(record)


@pytest.mark.parametrize("record", [
    {"terminal": "screen", "reason": "selective_causal_site", "selected_site_id": None},
    {"terminal": "null", "reason": "selective_causal_site", "selected_site_id": None},
    {"terminal": "invalid", "reason": "instrument_invalid", "selected_site_id": "attn:03"},
    {"terminal": "ok", "reason": "selective_causal_site", "selected_site_id": "attn:03"},
    {"terminal": "screen", "reason": "selective_causal_site", "selected_site_id": "attn:99"},
])
def test_terminal_schema_rejects_ambiguous_records(record) -> None:
    with pytest.raises(screen.FastScreenSpecError):
        screen.validate_terminal_record(record)


def test_dryrun_is_deterministic_and_declares_zero_model_gpu_queue_work() -> None:
    rows = authority()
    spec = valid_spec(rows)
    first = screen.compile_dryrun(spec, rows)
    second = screen.compile_dryrun(spec, rows)
    assert first == second
    assert first["head_stage"] == "pending"
    assert first["model_loaded"] is False
    assert first["gpu_accessed"] is False
    assert first["model_forwards"] == first["model_backwards"] == first["model_updates"] == 0
    assert first["queue_touched"] is False
    source = SOURCE.read_text()
    assert "import torch" not in source
    assert "import numpy" not in source

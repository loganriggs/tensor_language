"""CPU-only schema and binding checks for the prospective R585 v4 handoff."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


OPS = Path(__file__).resolve().parent
V3 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v3_addendum.json"
V4 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v4_addendum.json"
R585 = OPS / "induction_selector_payload_frozen_factor_rung585.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_is_prospective_and_binds_unchanged_v3():
    payload = json.loads(V4.read_text(encoding="utf-8"))
    assert payload["schema"] == "circuit_causal_validity_next_wave_handoff_v4_addendum"
    assert payload["v3_contract_sha256"] == sha256(V3)
    assert [item["lesson"] for item in payload["accepted_lessons"]] == [22]
    assert payload["failure_boundary"]["forbidden_terminal"] == "scientific_null"
    assert payload["failure_boundary"]["required_terminal"] == "preexecution_implementation_failure"


def test_v4_requires_call_census_shape_fixture_and_separate_model_validation():
    payload = json.loads(V4.read_text(encoding="utf-8"))
    assert set(payload["required_test_ids"]) == {
        "every_scientific_forward_call_site_enumerated",
        "validation_mode_explicit_at_every_scientific_forward_call",
        "registered_batch_and_padding_shapes_accepted",
        "checkpoint_and_model_structure_validation_remain_enforced",
    }
    assert set(payload["planted_negative_fixture_ids"]) == {
        "registered_batch32_rejected_by_fixed_4x256_interface_before_enqueue",
        "hidden_scientific_forward_call_with_implicit_or_incompatible_validation_rejected",
    }
    prompt = payload["builder_prompt_addendum"] + payload["critic_prompt_addendum"]
    assert "batch-32" in prompt and "(4,256)" in prompt
    assert "without loading the model" in prompt


def test_repaired_reference_enumerates_all_three_shape_compatible_forward_paths():
    tree = ast.parse(R585.read_text(encoding="utf-8"))
    expected = {
        "collect_capture_replay", "collect_native_comparator", "collect_intervention_arm"
    }
    observed = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in expected:
            continue
        calls = [
            call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "forward_with_dispatch"
        ]
        assert len(calls) == 1
        flags = [item.value for item in calls[0].keywords if item.arg == "require_production"]
        assert len(flags) == 1 and isinstance(flags[0], ast.Constant)
        assert flags[0].value is False
        observed[node.name] = True
    assert set(observed) == expected
    source = R585.read_text(encoding="utf-8")
    assert "BATCH = 32" in source
    assert "facade.load_bilin18(" in source

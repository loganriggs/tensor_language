"""CPU/static fail-closed tests for the full rung-522 runner protocol."""

from __future__ import annotations

from dataclasses import replace
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


OPS = Path(__file__).parent
GUARD_PATH = OPS / "attention8_selective_shared_projector_rung522_state_guard.py"
RUNNER_PATH = OPS / "attention8_selective_shared_projector_rung522_run.py"
SPEC = importlib.util.spec_from_file_location("rung522_state_guard", GUARD_PATH)
GUARD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GUARD
SPEC.loader.exec_module(GUARD)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _complete_state() -> GUARD.ProtocolState:
    state = GUARD.ProtocolState()
    for frame_id, spec in GUARD.EXPECTED_FRAME_SPECS.items():
        state.authorize_training(
            frame_id,
            split="FIT",
            training_targets=spec.training_targets,
            health_targets=spec.health_targets,
        )
        state.register_frozen_frame(GUARD.FrozenFrame(
            spec, _hash("frame:" + frame_id), _hash("scheduler:" + frame_id)
        ))
    state.record_optimization_events(20_600, 20_600)
    return state


def _freeze(state: GUARD.ProtocolState) -> GUARD.PretestFreeze:
    eligible = tuple(
        frame_id for frame_id, spec in GUARD.EXPECTED_FRAME_SPECS.items()
        if spec.family == "all_three"
    )
    return GUARD.PretestFreeze(
        frame_manifest_sha256=state.frame_manifest_sha256(),
        scheduler_manifest_sha256=state.scheduler_manifest_sha256(),
        validation_decisions_sha256=_hash("validation"),
        medoid_selection_sha256=_hash("medoid"),
        fingerprint_definition_sha256=_hash("fingerprint-definition"),
        test_sweep_plan_sha256=_hash("test-sweep"),
        registered_contract_sha256=GUARD.registered_contract_sha256(),
        selected_final_frame_id=eligible[0],
        eligible_all_three_frame_ids=eligible,
        selection_targets=GUARD.FITTED_TARGETS,
        validation_provisional_gates_passed=True,
        medoid_selection_rule="grassmann_medoid_lower_seed_tiebreak",
        test_sweep_plan_frozen=True,
    )


def _runner_main_is_unconditionally_closed(source: str) -> bool:
    tree = ast.parse(source)
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    return any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and any(
            isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and "RUNG522 SCIENCE CLOSED" in argument.value
            for argument in node.exc.args
        )
        for node in ast.walk(main)
    )


def _runner_calls(source: str) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_inventory_is_exactly_15_plus_15_plus_20_plus_48_plus_5():
    specs = tuple(GUARD.EXPECTED_FRAME_SPECS.values())
    counts = {
        family: sum(spec.family == family for spec in specs)
        for family in {spec.family for spec in specs}
    }
    assert len(specs) == len({spec.frame_id for spec in specs}) == 103
    assert counts == {
        "real_leave_one_out": 15,
        "recovery_only": 15,
        "target_oracle": 20,
        "label_null": 48,
        "all_three": 5,
    }


def test_test_access_fails_before_every_frame_and_decision_is_frozen():
    state = GUARD.ProtocolState()
    with pytest.raises(GUARD.ProtocolViolation, match="pre-TEST freeze"):
        state.open_test_once()
    with pytest.raises(GUARD.ProtocolViolation, match="TEST access attempted"):
        state.authorize_split_access("TEST")

    specs = list(GUARD.EXPECTED_FRAME_SPECS.values())
    for spec in specs[:-1]:
        state.register_frozen_frame(GUARD.FrozenFrame(
            spec, _hash(spec.frame_id), _hash("scheduler:" + spec.frame_id)
        ))
    state.record_optimization_events(20_600, 20_600)
    with pytest.raises(GUARD.ProtocolViolation, match="all 103 frames"):
        state.freeze_pretest(_freeze(state))


def test_frames_and_scheduler_fingerprints_require_real_hashes():
    state = GUARD.ProtocolState()
    spec = next(iter(GUARD.EXPECTED_FRAME_SPECS.values()))
    with pytest.raises(GUARD.ProtocolViolation, match="not frozen"):
        state.register_frozen_frame(GUARD.FrozenFrame(
            spec, _hash("frame"), _hash("scheduler"), frozen=False
        ))
    with pytest.raises(GUARD.ProtocolViolation, match="SHA-256"):
        state.register_frozen_frame(GUARD.FrozenFrame(
            spec, "not-a-hash", _hash("scheduler")
        ))


def test_fourth_target_is_excluded_from_shared_training_and_health_but_oracle_is_legal():
    for spec in GUARD.EXPECTED_FRAME_SPECS.values():
        if spec.family in GUARD.SHARED_FAMILIES:
            assert GUARD.RESERVED_FOURTH_TARGET not in spec.training_targets
            assert GUARD.RESERVED_FOURTH_TARGET not in spec.health_targets
    state = GUARD.ProtocolState()
    shared = next(
        spec for spec in GUARD.EXPECTED_FRAME_SPECS.values()
        if spec.family == "real_leave_one_out"
    )
    with pytest.raises(GUARD.ProtocolViolation, match="training target identities"):
        state.authorize_training(
            shared.frame_id,
            split="FIT",
            training_targets=shared.training_targets + (GUARD.RESERVED_FOURTH_TARGET,),
            health_targets=shared.health_targets,
        )
    with pytest.raises(GUARD.ProtocolViolation, match="health target identities"):
        state.authorize_training(
            shared.frame_id,
            split="FIT",
            training_targets=shared.training_targets,
            health_targets=shared.health_targets + (GUARD.RESERVED_FOURTH_TARGET,),
        )
    oracle = GUARD.EXPECTED_FRAME_SPECS[
        f"target_oracle:{GUARD.RESERVED_FOURTH_TARGET}:52200"
    ]
    state.authorize_training(
        oracle.frame_id,
        split="FIT",
        training_targets=oracle.training_targets,
        health_targets=oracle.health_targets,
    )


def test_fourth_target_cannot_enter_medoid_and_validation_failure_keeps_test_closed():
    state = _complete_state()
    with pytest.raises(GUARD.ProtocolViolation, match="fourth target"):
        state.freeze_pretest(replace(
            _freeze(state),
            selection_targets=GUARD.FITTED_TARGETS + (GUARD.RESERVED_FOURTH_TARGET,),
        ))
    state = _complete_state()
    with pytest.raises(GUARD.ProtocolViolation, match="VALIDATION gates failed"):
        state.freeze_pretest(replace(
            _freeze(state), validation_provisional_gates_passed=False
        ))


@pytest.mark.parametrize(
    "change,pattern",
    [
        ({"medoid_selection_rule": "best_validation_score"}, "geometry-only medoid"),
        ({"test_sweep_plan_frozen": False}, "sweep plan is not frozen"),
        ({"registered_contract_sha256": _hash("wrong")}, "contract differs"),
        ({"fingerprint_definition_sha256": "bad"}, "require lowercase SHA-256"),
    ],
)
def test_selection_fingerprint_plan_and_contract_are_frozen(change, pattern):
    state = _complete_state()
    with pytest.raises(GUARD.ProtocolViolation, match=pattern):
        state.freeze_pretest(replace(_freeze(state), **change))


def test_test_opens_once_and_no_training_or_frame_change_can_follow_freeze():
    state = _complete_state()
    state.freeze_pretest(_freeze(state))
    shared = GUARD.EXPECTED_FRAME_SPECS["all_three:52200"]
    with pytest.raises(GUARD.ProtocolViolation, match="permanently closed"):
        state.authorize_training(
            shared.frame_id,
            split="FIT",
            training_targets=shared.training_targets,
            health_targets=shared.health_targets,
        )
    state.open_test_once()
    state.authorize_split_access("TEST")
    with pytest.raises(GUARD.ProtocolViolation, match="forbidden"):
        state.record_optimization_events(1, 1)
    with pytest.raises(GUARD.ProtocolViolation, match="register a frame"):
        state.register_frozen_frame(next(iter(state._frames.values())))
    with pytest.raises(GUARD.ProtocolViolation, match="only once"):
        state.open_test_once()
    state.close_test()
    with pytest.raises(GUARD.ProtocolViolation, match="TEST access attempted"):
        state.authorize_split_access("TEST")


def test_registered_price_and_prediction_contract_is_literal():
    assert GUARD.REGISTERED_PRICE["frame_count_before_test"] == 103
    assert GUARD.REGISTERED_PRICE["updates_per_fit"] == 200
    assert GUARD.REGISTERED_PRICE["optimization_forward_events"] == 20_600
    assert GUARD.REGISTERED_PRICE["optimization_backward_events"] == 20_600
    assert GUARD.REGISTERED_PRICE["optimization_combined_events"] == 41_200
    assert GUARD.REGISTERED_PRICE["registered_worst_case_inference_forwards"] == 9_422
    assert sum(GUARD.INFERENCE_LEDGER.values()) == 9_422
    a = GUARD.REGISTERED_PREDICTIONS["A"]
    assert (a["minimum_signed_cosine"], a["maximum_scaled_relative_residual"]) == (0.75, 0.55)
    assert (a["minimum_member_rms_nat"], a["minimum_concentration"]) == (0.02, 4.0)
    assert a["minimum_full_attention8_concentration_improvement"] == 1.0
    assert a["row_bootstraps"] == 2_000 and a["exact_token_subset_minimum_pairs"] == 32
    b = GUARD.REGISTERED_PREDICTIONS["B"]
    assert b["minimum_recovery_only_concentration_improvement"] == 0.5
    assert b["paired_seed_sign_flips"] == 32
    c = GUARD.REGISTERED_PREDICTIONS["C"]
    assert c["fingerprint_permutations_per_cell"] == 20_000
    assert c["outside_union_to_smallest_quartet_rms_maximum"] == 0.25
    assert GUARD.REGISTERED_PREDICTIONS["D"][
        "quartet_to_median_nonquartet_effect_minimum"
    ] == 2.0


def test_price_counters_fail_closed_at_registered_ceilings():
    state = GUARD.ProtocolState()
    state.record_optimization_events(20_600, 20_600)
    with pytest.raises(GUARD.ProtocolViolation, match="hard ceiling"):
        state.record_optimization_events(2_000, 2_000)
    state.record_inference_events(12_000)
    with pytest.raises(GUARD.ProtocolViolation, match="inference ceiling"):
        state.record_inference_events(1)
    state.record_inference_events(2_000, removal=True)
    with pytest.raises(GUARD.ProtocolViolation, match="removal inference ceiling"):
        state.record_inference_events(1, removal=True)


def test_partial_runner_must_remain_closed_until_state_guard_is_integrated():
    source = RUNNER_PATH.read_text()
    closed = _runner_main_is_unconditionally_closed(source)
    calls = _runner_calls(source)
    required = {
        "authorize_training",
        "register_frozen_frame",
        "freeze_pretest",
        "open_test_once",
        "authorize_split_access",
        "close_test",
    }
    integrated = (
        "attention8_selective_shared_projector_rung522_state_guard" in source
        and required <= calls
    )
    assert closed or integrated, (
        "science runner removed its kill-switch before integrating every protocol-state gate"
    )
    if not integrated:
        assert closed


def test_nonunique_frame_keys_or_unguarded_test_capture_are_safe_only_behind_kill_switch():
    source = RUNNER_PATH.read_text()
    closed = _runner_main_is_unconditionally_closed(source)
    if "def registered_fit_specs" in source:
        key_mentions_fitted_target = "self.fitted_targets" in source[
            source.index("def key"):source.index("def registered_fit_specs")
        ]
    else:
        # The corrected runner delegates its unique 103-ID inventory to the
        # imported state guard rather than maintaining a second FitSpec key.
        key_mentions_fitted_target = (
            "attention8_selective_shared_projector_rung522_state_guard" in source
            and "class FitSpec" not in source
        )
    capture = source[source.index("def _capture_split"):source.index("def _writes_for_rows")]
    guarded_test_capture = "authorize_split_access" in capture
    if not key_mentions_fitted_target or not guarded_test_capture:
        assert closed, "known frame-key/TEST-capture hazard became reachable"


def test_live_runner_must_register_all_four_predictions_and_literal_prices():
    source = RUNNER_PATH.read_text()
    if not _runner_main_is_unconditionally_closed(source):
        for label in ("pred_a", "pred_b", "pred_c", "pred_d"):
            assert label in source
        for literal in ("103", "200", "20600", "41200", "9422", "12000", "2000"):
            assert literal in source


def test_state_guard_has_no_model_data_or_tensor_imports():
    source = GUARD_PATH.read_text()
    assert "import torch" not in source
    assert "census_state_diverse.pt" not in source
    assert "curated_rows.pt" not in source

"""CPU/static tests for the fail-closed rung-522 scientific runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


PATH = Path(__file__).with_name("attention8_selective_shared_projector_rung522_run.py")
SPEC = importlib.util.spec_from_file_location("rung522_partial_runner", PATH)
RUN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUN
SPEC.loader.exec_module(RUN)


def _pair(target: str, member_row: int, control_row: int, position: int):
    return RUN.TargetPairs(
        target,
        torch.tensor([member_row * RUN.TOKENS + position]),
        torch.tensor([control_row * RUN.TOKENS + position]),
        torch.tensor([0]),
    )


class _FakeInstrument:
    def __init__(self, rows):
        self.split_rows = {"fit": rows, "validation": rows}
        row_to_local = torch.full((1000,), -1, dtype=torch.int64)
        row_to_local[rows] = torch.arange(rows.numel())
        self.row_to_local = {"fit": row_to_local, "validation": row_to_local}
        maps = [torch.tensor(index) for index in range(4)]
        self.design = {"donors": {"fit": {"maps": maps}, "validation": {"maps": maps}}}
        self.calls = []

    def projected_delta(
        self,
        split,
        rows,
        donor_map,
        frame,
        *,
        optimization,
        inference_bucket=None,
    ):
        self.calls.append(
            (split, rows.tolist(), int(donor_map), optimization, inference_bucket)
        )
        values = torch.arange(rows.numel() * RUN.TOKENS, dtype=frame.dtype).view(
            rows.numel(), RUN.TOKENS
        )
        return values + frame.sum() * 0, torch.ones(rows.numel())


def test_call_ledger_requires_exact_named_pretest_and_final_buckets():
    ledger = RUN.CallLedger(
        optimization_forwards=20_600,
        optimization_backwards=20_600,
        inference_forwards=5_029,
        inference_by_bucket={
            "native_capture": 131,
            "native_replay": 131,
            "self_donor": 2,
            "fit_d0_full_attention8": 95,
            "fit_health": 206,
            "full_attention8_comparator": 36,
            "prediction_a": 2_988,
            "recovery_only": 540,
            "haar": 720,
            "all_three_selection_and_test": 180,
        },
    )
    ledger.assert_pretest_registered_price()
    with pytest.raises(RuntimeError, match="unregistered inference bucket"):
        ledger.charge("inference_forward", bucket="misc")
    for bucket, increment in {
        "native_capture": 36,
        "native_replay": 36,
        "self_donor": 1,
        "full_attention8_comparator": 36,
        "prediction_a": 2_988,
        "recovery_only": 540,
        "haar": 720,
        "all_three_selection_and_test": 36,
    }.items():
        ledger.charge("inference_forward", increment, bucket=bucket)
    ledger.charge("removal_forward", 36)
    ledger.assert_final_registered_price()


def test_callback_keeps_roles_targets_and_map_axis_explicit():
    omitted = RUN.FITTED_TAGS[0]
    spec = RUN.state_guard.EXPECTED_FRAME_SPECS[
        f"real_leave_one_out:{omitted}:52200"
    ]
    rows = torch.tensor([10, 11, 12, 13])
    first, second = spec.training_targets
    pairs = {
        first: _pair(first, 10, 11, 3),
        second: _pair(second, 12, 13, 7),
    }
    row_mask = torch.zeros(1000, dtype=torch.bool)
    row_mask[rows] = True
    balanced = RUN._make_balanced_scheduler(spec, pairs, row_mask)
    full = torch.empty((4, 4, RUN.TOKENS))
    for map_index in range(4):
        for local in range(4):
            full[map_index, local] = 10_000 * map_index + 100 * local + torch.arange(RUN.TOKENS)
    instrument = _FakeInstrument(rows)
    callback = RUN.ProjectedResponseCallback(
        instrument,
        spec,
        split="fit",
        pairs=pairs,
        balanced=balanced,
        full_by_map=full,
        optimization=True,
    )
    response = callback(torch.eye(RUN.D)[:, : RUN.RANK], 2)
    assert tuple(sorted(response)) == tuple(sorted(spec.training_targets))
    assert instrument.calls[0][2:] == (2, True, None)
    assert response[first].full_member.tolist() == [20_003]
    assert response[second].full_member.tolist() == [20_207]
    assert response[first].projected_member.tolist() == [3]
    assert response[first].projected_control.tolist() == [RUN.TOKENS + 3]
    assert response[second].projected_member.tolist() == [2 * RUN.TOKENS + 7]
    assert response[second].projected_control.tolist() == [3 * RUN.TOKENS + 7]


def test_swap_arm_plan_has_explicit_complete_axes_and_no_reshape_assumption():
    rows = [31, 47, 52, 80, 91, 103]
    arms = RUN._swap_arm_plan(rows)
    assert len(arms) == 96
    assert len(set(arms)) == 96
    assert {arm.cell for arm in arms} == {
        "D0:forward", "D0:reverse", "D1:forward", "D1:reverse"
    }
    for ensemble, offset in (("D0", 0), ("D1", 4)):
        for direction in ("forward", "reverse"):
            subset = [
                arm for arm in arms
                if arm.ensemble == ensemble and arm.direction == direction
            ]
            assert [arm.map_index for arm in subset] == [
                value for value in range(offset, offset + 4) for _ in rows
            ]
            assert [arm.recipient_row for arm in subset] == rows * 4


def test_swap_evaluator_reconstructs_every_physical_arm_explicitly(monkeypatch):
    rows = torch.tensor([31, 47, 52, 80, 91, 103])
    instrument = object.__new__(RUN.Rung522Instrument)
    instrument.state = SimpleNamespace(
        authorize_split_access=lambda _split: None,
        record_inference_events=lambda _count: None,
    )
    instrument.captures = {"validation": torch.zeros((6, RUN.TOKENS, RUN.D))}
    instrument.split_rows = {"validation": rows}
    row_to_local = torch.full((1000,), -1, dtype=torch.int64)
    row_to_local[rows] = torch.arange(6)
    instrument.row_to_local = {"validation": row_to_local}
    instrument.native_ce = {"validation": torch.zeros((6, RUN.TOKENS))}
    instrument.data = {
        "rows": torch.zeros((1000, RUN.TOKENS + 1), dtype=torch.int64)
    }
    maps = tuple(torch.tensor(index + 1) for index in range(8))
    inverse = tuple(torch.tensor(index + 101) for index in range(8))
    instrument.design = {
        "donors": {"validation": {"maps": maps, "inverse_maps": inverse}}
    }
    instrument.device = torch.device("cpu")
    instrument.model = object()
    instrument.ledger = RUN.CallLedger()

    def donor_writes(_split, selected, donor_map):
        return torch.full(
            (selected.numel(), RUN.TOKENS, RUN.D), float(donor_map)
        )

    instrument.donor_writes = donor_writes

    def fake_execute(_model, tokens, *, edit=None, capture=False):
        assert not capture and edit is not None
        changed = edit(torch.zeros((tokens.shape[0], RUN.TOKENS, RUN.D)))
        assert bool((changed != 0).any())
        physical = torch.arange(tokens.shape[0], dtype=torch.float32)
        logits = physical[:, None, None].expand(-1, RUN.TOKENS, 1).clone()
        return logits, None, {"per_sequence_edit_rms": torch.ones(tokens.shape[0])}

    monkeypatch.setattr(RUN, "_execute", fake_execute)
    monkeypatch.setattr(RUN, "_per_token_ce", lambda logits, _targets: logits[..., 0])
    result = RUN.Rung522Instrument.evaluate_swap(
        instrument,
        "validation",
        frame=None,
        inference_bucket="prediction_a",
    )
    for cell_index, cell in enumerate(
        ("D0:forward", "D0:reverse", "D1:forward", "D1:reverse")
    ):
        expected = torch.arange(
            cell_index * 24, (cell_index + 1) * 24, dtype=torch.float32
        ).view(4, 6)
        assert torch.equal(result.map_responses[cell][..., 0], expected)
        assert torch.equal(result.cell_responses[cell][..., 0], expected.mean(0))
    assert instrument.ledger.inference_by_bucket == {"prediction_a": 1}


def test_health_callback_ignores_minus_one_and_uses_validation_d0_map_zero():
    omitted = RUN.FITTED_TAGS[1]
    spec = RUN.state_guard.EXPECTED_FRAME_SPECS[
        f"real_leave_one_out:{omitted}:52201"
    ]
    rows = torch.tensor([20, 21, 22, 23])
    first, second = spec.training_targets
    pairs = {
        first: _pair(first, 20, 21, 1),
        second: _pair(second, 22, 23, 2),
    }
    row_mask = torch.zeros(1000, dtype=torch.bool)
    row_mask[rows] = True
    balanced = RUN._make_balanced_scheduler(spec, pairs, row_mask)
    instrument = _FakeInstrument(rows)
    callback = RUN.ProjectedResponseCallback(
        instrument,
        spec,
        split="validation",
        pairs=pairs,
        balanced=balanced,
        full_by_map=torch.ones((1, 4, RUN.TOKENS)),
        optimization=False,
        fixed_health_batch=True,
    )
    callback(torch.eye(RUN.D)[:, : RUN.RANK], -1)
    assert instrument.calls[0][0] == "validation"
    assert instrument.calls[0][1] == [role.row_index for role in balanced.batch(0).roles]
    assert instrument.calls[0][2:] == (0, False, "fit_health")


def test_response_scoring_uses_rms_bootstrap_full_a8_comparator_and_exact_token_tier():
    positions = torch.arange(40)
    pairs = RUN.TargetPairs(
        "r.2.0.2",
        10 * RUN.TOKENS + positions,
        11 * RUN.TOKENS + positions,
        torch.zeros(40, dtype=torch.int64),
    )
    projected = torch.zeros((2, RUN.TOKENS))
    projected[0, positions] = 1.0
    projected[1, positions] = 0.1
    full = torch.zeros_like(projected)
    full[0, positions] = 1.0
    full[1, positions] = 0.5
    row_to_local = torch.full((1000,), -1, dtype=torch.int64)
    row_to_local[10] = 0
    row_to_local[11] = 1
    score = RUN.score_response_cell(
        projected,
        full,
        pairs,
        row_to_local,
        cell_id="validation:r.2.0.2:D0:forward:seed52200",
    )
    assert score["member_rms"] == 1
    assert score["control_rms"] == pytest.approx(0.1)
    assert score["concentration"] == pytest.approx(10)
    assert score["full_attention8_concentration"] == pytest.approx(2)
    assert score["concentration_improvement_over_full_attention8"] == pytest.approx(8)
    assert score["fourfold_margin_lower95"] == pytest.approx(0.6)
    assert score["exact_token_tier0_or1"]["pair_count"] == 40
    assert score["exact_token_tier0_or1"]["passes"]
    assert score["base_gates_pass"]


def test_response_scoring_pairs_selectivity_bootstrap_against_recovery_control():
    positions = torch.arange(40)
    pairs = RUN.TargetPairs(
        "r.2.0.2",
        10 * RUN.TOKENS + positions,
        11 * RUN.TOKENS + positions,
        torch.zeros(40, dtype=torch.int64),
    )
    real = torch.zeros((2, RUN.TOKENS))
    real[0, positions] = 1.0
    real[1, positions] = 0.1
    recovery = torch.zeros_like(real)
    recovery[0, positions] = 1.0
    recovery[1, positions] = 0.5
    row_to_local = torch.full((1000,), -1, dtype=torch.int64)
    row_to_local[10] = 0
    row_to_local[11] = 1
    score = RUN.score_response_cell(
        real,
        real,
        pairs,
        row_to_local,
        cell_id="paired-control-test",
        selectivity_comparison=recovery,
    )
    assert score["bounded_selectivity_improvement_lower95"] > 0


def test_fit_projection_mean_uses_every_fit_row_and_position_not_circuit_masks():
    writes = torch.zeros((2, 3, RUN.D), dtype=torch.float32)
    writes[..., : RUN.RANK] = torch.arange(
        2 * 3 * RUN.RANK, dtype=torch.float32
    ).view(2, 3, RUN.RANK)
    frame = torch.eye(RUN.D, dtype=torch.float32)[:, : RUN.RANK]
    fake = SimpleNamespace(captures={"fit": writes})
    mean = RUN.Rung522Instrument.fit_projection_mean(fake, frame)
    assert mean.tolist() == pytest.approx(
        writes[..., : RUN.RANK].double().mean((0, 1)).tolist()
    )


def test_fingerprint_uses_same_rms_coordinate_for_all_32_circuits():
    tags = RUN.stage_a.FINGERPRINT_TAGS
    response = torch.zeros((2 * len(tags), RUN.TOKENS))
    row_to_local = torch.full((1000,), -1, dtype=torch.int64)
    pairs = {}
    for index, target in enumerate(tags):
        member_row, control_row = 2 * index, 2 * index + 1
        row_to_local[member_row] = member_row
        row_to_local[control_row] = control_row
        pairs[target] = _pair(target, member_row, control_row, 0)
        response[member_row, 0] = 2.0 if target in RUN.QUARTET_TAGS else 1.0
    coordinates = RUN.fingerprint_coordinates(response, pairs, row_to_local)
    separation = RUN.quartet_separation(coordinates)
    assert len(coordinates) == 32
    assert separation["minimum_quartet_coordinate"] == 2
    assert separation["maximum_nonquartet_coordinate"] == 1
    assert separation["separation"] == 1


def test_fingerprint_null_uses_one_common_bijection_and_strict_gate_fields():
    tags = RUN.stage_a.FINGERPRINT_TAGS
    response = torch.zeros((2 * len(tags), RUN.TOKENS))
    pairs = {}
    for index, target in enumerate(tags):
        member_row, control_row = 2 * index, 2 * index + 1
        pairs[target] = _pair(target, member_row, control_row, 0)
        response[member_row, 0] = 2.0 if target in RUN.QUARTET_TAGS else 1.0
    row_mask = torch.zeros(1000, dtype=torch.bool)
    row_mask[: 2 * len(tags)] = True
    data = {"row_masks": {"test": row_mask}}
    size = 1000 * RUN.TOKENS
    descriptors = {
        "token_class": torch.zeros(size, dtype=torch.int64),
        "position_bin": torch.arange(RUN.TOKENS).repeat(1000) // 32,
        "ce_decile": torch.zeros(size, dtype=torch.int64),
    }
    null = RUN.fingerprint_null_distribution(
        response,
        pairs,
        data,
        descriptors,
        "test",
        cell_id="test:D0:forward",
        replicates=2,
    )
    assert null["replicates"] == 2
    assert null["observed_separation"] == 1
    assert len(null["null_samples_sha256"]) == 64
    assert len(null["first_permutation_sha256"]) == 64


def test_scientific_main_has_complete_fail_closed_orchestration():
    source = PATH.read_text()
    main = source[source.index("def main"):]
    assert "RUNG522 SCIENCE CLOSED" not in main
    required = (
        "train_all_registered_frames(",
        "write_frame_archive(",
        "evaluate_provisional_validation_gates(",
        "assert_pretest_registered_price()",
        "write_pretest_manifest(",
        "open_test_once()",
        "evaluate_test_suite(",
        "evaluate_final_validation_test_gates(",
        "score_prediction_c(",
        "evaluate_removal(",
        "score_prediction_d(",
        "close_test()",
        "assert_final_registered_price(",
    )
    for operation in required:
        assert operation in main
    assert main.index("if not provisional.pretest_passes:") < main.index(
        "open_test_once()"
    )
    assert main.index("write_pretest_manifest(") < main.index("open_test_once()")
    assert main.index("open_test_once()") < main.index("evaluate_test_suite(")
    assert main.index("if abc_pass:") < main.index("evaluate_removal(")
    assert main.index("evaluate_removal(") < main.index("close_test()")
    for label in ("pred_a", "pred_b", "pred_c", "pred_d"):
        assert label in source


def test_materialized_scheduler_is_the_exact_hashed_archive_payload():
    spec = next(
        value for value in RUN.state_guard.EXPECTED_FRAME_SPECS.values()
        if value.family == "real_leave_one_out"
    )
    member = {target: (1, 3, 5) for target in spec.training_targets}
    control = {target: (2, 4, 6) for target in spec.training_targets}
    balanced = RUN.scheduler.two_target_scheduler(
        spec.training_targets, member, control, seed=spec.seed
    )
    payload = RUN._scheduler_payload(balanced)
    assert RUN._sha256_json(payload) == balanced.fingerprint
    assert RUN._batch_zero_rows(payload) == balanced.batch(0).rows_by_role()


def test_one_synthetic_fit_produces_an_archive_revalidated_evidence_record():
    spec = next(
        value for value in RUN.state_guard.EXPECTED_FRAME_SPECS.values()
        if value.family == "real_leave_one_out" and value.seed == 52200
    )
    member = {target: (1, 3, 5) for target in spec.training_targets}
    control = {target: (2, 4, 6) for target in spec.training_targets}
    fit_scheduler = RUN.scheduler.two_target_scheduler(
        spec.training_targets, member, control, seed=spec.seed
    )
    health_scheduler = RUN.scheduler.two_target_scheduler(
        spec.training_targets, member, control, seed=spec.seed
    )
    state = RUN.state_guard.ProtocolState()
    ledger = RUN.CallLedger()
    instrument = SimpleNamespace(
        state=state,
        ledger=ledger,
        device=torch.device("cpu"),
        model=SimpleNamespace(parameters=lambda: ()),
    )

    class Callback:
        def __init__(self, balanced, *, optimization, fixed_health_batch):
            self.spec = spec
            self.balanced = balanced
            self.optimization = optimization
            self.fixed_health_batch = fixed_health_batch

        def __call__(self, frame, _step):
            if self.optimization:
                ledger.charge("optimization_forward")
                state.record_optimization_events(1, 0)
            else:
                ledger.charge("inference_forward", bucket="fit_health")
                state.record_inference_events(1)
            projected = frame[0, 0].repeat(4)
            zero = frame[0, 0].repeat(4) * 0
            return {
                target: RUN.core.TargetResponse(
                    full_member=torch.ones(4),
                    projected_member=projected,
                    projected_control=zero,
                )
                for target in spec.training_targets
            }

    frame, record = RUN.fit_one_registered_frame(
        instrument,
        spec,
        state,
        Callback(fit_scheduler, optimization=True, fixed_health_batch=False),
        Callback(health_scheduler, optimization=False, fixed_health_batch=True),
    )
    artifact = RUN.archive.FrameArtifact(
        spec=spec,
        frame=frame,
        tensor_sha256=record.frame_sha256,
        fit_scheduler_payload=record.fit_scheduler_payload,
        validation_scheduler_payload=record.validation_scheduler_payload,
        fit_record_payload=record.fit_record_payload,
        health_record_payload=record.health_record_payload,
    )
    archived = RUN.archive._validate_artifact(artifact)
    assert archived.tensor_sha256 == record.frame_sha256
    assert archived.fit_record_sha256 == record.fit_record_sha256
    assert archived.health_record_sha256 == record.health_record_sha256
    assert ledger.optimization_forwards == ledger.optimization_backwards == 200
    assert ledger.inference_by_bucket == {"fit_health": 2}

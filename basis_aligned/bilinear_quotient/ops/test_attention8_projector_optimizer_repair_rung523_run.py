"""CPU-only structural tests for the sealed rung-523 runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN = _load("attention8_projector_optimizer_repair_rung523_run")


def test_real_frame_census_is_exactly_three_omissions_times_five_seeds():
    specs = RUN.real_specs()
    assert len(specs) == 15
    assert {spec.omitted_target for spec in specs} == set(RUN.r522_state.FITTED_TARGETS)
    assert {spec.seed for spec in specs} == set(range(52200, 52205))
    assert all(spec.family == "real_leave_one_out" for spec in specs)


def test_split_guard_rejects_test_and_has_exact_terminal_counts():
    state = RUN.FitValidationOnlyState()
    state.authorize_split_access("FIT")
    state.authorize_split_access("VALIDATION")
    with pytest.raises(RuntimeError, match="forbids split access"):
        state.authorize_split_access("TEST")
    state.inference_events = RUN.EXPECTED_INFERENCE_FORWARDS
    state.optimization_forwards = RUN.EXPECTED_OPTIMIZATION_FORWARDS
    state.optimization_backwards = RUN.EXPECTED_OPTIMIZATION_BACKWARDS
    state.assert_terminal()


def test_objective_switches_only_the_denominator():
    response = RUN.r522.core.TargetResponse(
        full_member=torch.tensor([0.1]),
        projected_member=torch.tensor([0.0]),
        projected_control=torch.tensor([0.1]),
    )
    scales = {"target": torch.tensor([1.0, 1.0, 1.0, 1.0])}
    row, row_name, _ = RUN._objective(
        {"target": response}, scales, map_index=0, scale_mode="row_specific"
    )
    fixed, fixed_name, _ = RUN._objective(
        {"target": response}, scales, map_index=0, scale_mode="fixed_target_map"
    )
    assert row_name == fixed_name == "target"
    assert float(row) == pytest.approx(25.0)
    assert float(fixed) == pytest.approx(0.25)


def test_complete_200_update_candidate_path_is_finite_on_cpu():
    state = RUN.FitValidationOnlyState()
    ledger = RUN.r522.CallLedger()
    instrument = SimpleNamespace(
        device=torch.device("cpu"),
        ledger=ledger,
        state=state,
        model=torch.nn.Module(),
    )

    class Callback:
        def __init__(self, optimization):
            self.optimization = optimization
            self.balanced = SimpleNamespace(fingerprint="0" * 64)

        def __call__(self, frame, _step):
            if self.optimization:
                ledger.charge("optimization_forward")
                state.record_optimization_events(1, 0)
            else:
                ledger.charge("inference_forward", bucket="fit_health")
                state.record_inference_events(1)
            member = frame[0, :2].sum().reshape(1)
            control = frame[1, :2].sum().reshape(1)
            response = RUN.r522.core.TargetResponse(
                full_member=torch.ones(1),
                projected_member=member,
                projected_control=control,
            )
            return {"r.2.1.1": response, "r.2.2.1": response}

    spec = RUN.r522_state.FrameSpec(
        frame_id="real_leave_one_out:r.2.0.2:52200",
        family="real_leave_one_out",
        seed=52200,
        training_targets=("r.2.1.1", "r.2.2.1"),
        health_targets=("r.2.1.1", "r.2.2.1"),
        omitted_target="r.2.0.2",
    )
    scales = {
        target: torch.ones(4) for target in spec.training_targets
    }
    frame, record, diagnostics = RUN._fit_candidate(
        instrument,
        spec,
        Callback(True),
        Callback(False),
        scales,
        arm=RUN.repair.FIXED_LOW_LR,
        scale_mode="fixed_target_map",
        learning_rate=0.003,
    )
    assert frame.shape == (1152, 4)
    assert len(record.loss_history) == 200
    assert all(torch.isfinite(torch.tensor(record.loss_history)))
    assert ledger.optimization_forwards == ledger.optimization_backwards == 200
    assert ledger.inference_by_bucket == {"fit_health": 2}
    assert len(diagnostics["per_update_per_target_loss_sha256"]) == 64

"""CPU/fake/planted tests for the narrow Task 14 tensor backend."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import run_task14_head11_3_projector_discovery as program
import task14_program_a_torch_backend as backend


class FakeHeadModel(nn.Module):
    def __init__(self, states):
        super().__init__()
        self.c_proj = nn.Identity()
        self.states = states

    def forward_logits(self, tokens, lengths):
        value = torch.zeros(
            *tokens.shape, backend.MODEL_WIDTH, dtype=torch.float64
        )
        for index, length in enumerate(lengths):
            key = tuple(int(x) for x in tokens[index, :length])
            value[index, length - 1, backend.HEAD_START] = self.states[key]
        output = self.c_proj(value)
        logits = torch.zeros(*tokens.shape, 400, dtype=torch.float64)
        logits[..., 389] = output[..., backend.HEAD_START]
        logits[..., 318] = -output[..., backend.HEAD_START]
        return logits


def _fake_backend(*, enforce_production_contract=True, frozen_parameters=()):
    endpoints = backend.load_discovery_endpoints()
    states = {}
    for endpoint in endpoints.values():
        key = endpoint.token_ids
        state = float(endpoint.cell_metadata[-1])
        if key in states:
            assert states[key] == state
        states[key] = state
    model = FakeHeadModel(states)
    return backend.Task14ProgramATorchBackend(
        forward_logits=model.forward_logits,
        c_proj_module=model.c_proj,
        device="cpu",
        batch_size=32,
        enforce_production_contract=enforce_production_contract,
        frozen_parameters=frozen_parameters,
    )


def test_committed_shard_loads_only_discovery_endpoints() -> None:
    endpoints = backend.load_discovery_endpoints()
    assert len(endpoints) == 128
    assert len({endpoint.cell_metadata[0] for endpoint in endpoints.values()}) == 16
    assert all(endpoint.final_position == len(endpoint.token_ids) - 1
               for endpoint in endpoints.values())


def test_live_parameter_hash_detects_change_not_just_version_metadata() -> None:
    parameter = nn.Parameter(torch.arange(7, dtype=torch.float32), requires_grad=False)
    collector = _fake_backend(frozen_parameters=(parameter,))
    original = collector._checkpoint_tensor_sha256
    with torch.no_grad():
        parameter[2] = -99.0
    assert collector._live_parameter_sha256() != original


def test_wrong_shard_hash_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "shard.json"
    changed.write_text(backend.SHARD_PATH.read_text() + " ")
    with pytest.raises(backend.Task14BackendError, match="hash changed"):
        backend.load_discovery_endpoints(changed)


def test_fake_spectral_collection_has_exact_shapes_effects_and_counts() -> None:
    plan = program.compile_discovery_plan()
    relations = tuple(row for row in plan.fit if row.role == "target")
    collector = _fake_backend(enforce_production_contract=False)
    result = collector.collect_spectral_inputs(relations)
    assert result.ordinals == tuple(row.ordinal for row in relations)
    assert result.head_deltas.shape == (116, 128)
    assert result.downstream_gradients.shape == (116, 128)
    assert result.full_head_effects.shape == (116,)
    assert torch.all(result.full_head_effects > 0)
    assert result.model_counts == {
        "forward_calls": 6,
        "backward_calls": 2,
        "example_evaluations": 180,
    }
    assert result.source_partitions == ("DISCOVERY",)
    assert result.validation_records_seen == 0
    assert result.validation_token_sequences_seen == 0
    assert torch.all(result.downstream_gradients[:, 1:] == 0)


def test_relation_outside_shard_fails_before_forward() -> None:
    relation = program.Relation(
        9999, "bad", "VALIDATION:endpoint", "also:bad", "bad", "target"
    )
    with pytest.raises(backend.Task14BackendError, match="escaped"):
        _fake_backend().collect_spectral_inputs((relation,))


def test_partial_head_hook_keeps_frame_gradient_live() -> None:
    collector = _fake_backend(enforce_production_contract=False)
    endpoints = list(collector.endpoints.values())
    target = endpoints[0]
    donor = next(
        endpoint for endpoint in endpoints
        if endpoint.cell_metadata[-1] == -target.cell_metadata[-1]
    )
    donor_head = torch.zeros(1, 128, dtype=torch.float64)
    donor_head[0, 0] = float(donor.cell_metadata[-1])
    frame = torch.eye(128, dtype=torch.float64)[:, :1].clone().requires_grad_(True)
    logits, _ = collector._forward(
        (target,), donor_heads=donor_head, frame=frame
    )
    gradient = torch.autograd.grad(logits[0, 389] - logits[0, 318], frame)[0]
    assert gradient.shape == (128, 1)
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0


def test_rank0_and_rank128_replays_are_measured_and_cached() -> None:
    plan = program.compile_discovery_plan()
    collector = _fake_backend()
    collector._ensure_baselines(plan.fit + plan.select)
    first = collector._ensure_endpoint_replays(plan.select)
    assert first == {
        "forward_calls": 7, "backward_calls": 0,
        "example_evaluations": 209,
    }
    assert collector.replay_rank0_exact is True
    assert collector.replay_rank128_exact is True
    assert collector._ensure_endpoint_replays(plan.select) == {
        "forward_calls": 0, "backward_calls": 0,
        "example_evaluations": 0,
    }


def test_endpoint_replay_detects_changed_cached_logits() -> None:
    plan = program.compile_discovery_plan()
    collector = _fake_backend()
    collector._ensure_baselines(plan.fit + plan.select)
    relation = plan.select[0]
    collector.full_head_logits[relation.ordinal] = (
        collector.full_head_logits[relation.ordinal] + 1.0
    )
    collector._ensure_endpoint_replays(plan.select)
    assert collector.replay_rank0_exact is True
    assert collector.replay_rank128_exact is False


def test_planted_householder_fit_recovers_projector() -> None:
    initial = torch.eye(8, dtype=torch.float64)[:, :1]
    target = torch.zeros(8, 1, dtype=torch.float64)
    target[0, 0] = target[3, 0] = 2 ** -0.5
    target_projector = target @ target.T

    def objective(frame):
        return torch.sum((frame @ frame.T - target_projector) ** 2)

    result = backend.fit_householder_frame(
        initial,
        objective=objective,
        updates=160,
        optimizer_factory=lambda parameters: torch.optim.Adam(parameters, lr=0.05),
    )
    assert result.gradients_finite
    assert result.maximum_orthonormality_error < 1e-10
    assert torch.allclose(
        result.frame @ result.frame.T, target_projector, atol=2e-4, rtol=0
    )
    assert result.losses[-1] < 1e-7


def test_hierarchical_schedule_and_permutation_labels_are_deterministic() -> None:
    plan = program.compile_discovery_plan()
    backend.Task14ProgramATorchBackend._validate_relation_cells(
        plan.fit, split="FIT"
    )
    backend.Task14ProgramATorchBackend._validate_relation_cells(
        plan.select, split="SELECT"
    )
    with pytest.raises(backend.Task14BackendError, match="coverage"):
        backend.Task14ProgramATorchBackend._validate_relation_cells(
            plan.select[:-1], split="SELECT"
        )
    first = backend.Task14ProgramATorchBackend._training_schedule(
        plan.fit, rank=2, start=1, permutation_id=None, updates=3,
        objective=program.FIT_OBJECTIVE,
    )
    second = backend.Task14ProgramATorchBackend._training_schedule(
        plan.fit, rank=2, start=1, permutation_id=None, updates=3,
        objective=program.FIT_OBJECTIVE,
    )
    assert [[row.ordinal for row in batch] for batch in first] == [
        [row.ordinal for row in batch] for batch in second
    ]
    assert all(sum(row.role == "target" for row in batch) == 16 for batch in first)
    assert all(sum(row.role == "control" for row in batch) == 16 for batch in first)
    for permutation_id in (0, 1):
        labels = backend.Task14ProgramATorchBackend._permutation_labels(
            plan.fit, permutation_id
        )
        for cell in {row.cell_key for row in plan.fit if row.role == "target"}:
            values = [labels[row.ordinal] for row in plan.fit
                      if row.role == "target" and row.cell_key == cell]
            assert sum(value == 1 for value in values) == (len(values) + 1) // 2
            assert set(values) <= {-1.0, 1.0}


def test_short_fake_fit_and_fixed_score_use_injected_frozen_config(monkeypatch) -> None:
    plan = program.compile_discovery_plan()
    targets = [row for row in plan.fit if row.role == "target"][:2]
    controls = [row for row in plan.fit if row.role == "control"][:2]
    select_targets = [row for row in plan.select if row.role == "target"][:2]
    select_controls = [row for row in plan.select if row.role == "control"][:2]
    fit_rows = tuple(targets + controls)
    select_rows = tuple(select_targets + select_controls)
    config = replace(
        program.FIT_OBJECTIVE,
        full_vocabulary_size=400,
        target_draws_per_update=2,
        control_draws_per_update=2,
    )
    monkeypatch.setattr(program, "FIT_OBJECTIVE", config)
    monkeypatch.setattr(program, "UPDATES", 4)
    monkeypatch.setattr(program, "BATCH_SIZE", 4)
    collector = _fake_backend(enforce_production_contract=False)
    collector.collect_spectral_inputs(tuple(targets))
    initial = torch.eye(128, dtype=torch.float64)[:, :1]
    fitted = collector.fit_and_score(
        fit_relations=fit_rows,
        select_relations=select_rows,
        rank=1,
        start=0,
        initial_frame=initial,
        updates=4,
        batch_size=4,
        objective=config,
        permutation_id=None,
    )
    assert fitted.scored_ordinals == tuple(row.ordinal for row in select_rows)
    assert fitted.health.schedule_updates == 4
    assert fitted.model_counts["backward_calls"] == 4
    assert fitted.target_cells and fitted.control_cells
    fixed = collector.score_fixed_frame(
        select_relations=select_rows, frame=initial, control_id="test-haar"
    )
    assert fixed.scored_ordinals == fitted.scored_ordinals
    assert fixed.model_counts == {
        "forward_calls": 1, "backward_calls": 0, "example_evaluations": 4
    }


def test_full_select_score_has_exact_cells_flags_metrics_and_counts(monkeypatch) -> None:
    plan = program.compile_discovery_plan()
    config = replace(program.FIT_OBJECTIVE, full_vocabulary_size=400)
    monkeypatch.setattr(program, "FIT_OBJECTIVE", config)
    collector = _fake_backend()
    targets = tuple(row for row in plan.fit if row.role == "target")
    collector.collect_spectral_inputs(targets)
    setup_counts = collector._ensure_baselines(plan.fit + plan.select)
    assert setup_counts == {
        "forward_calls": 8,
        "backward_calls": 0,
        "example_evaluations": 246,
    }
    collector.fit_control_normalizer = float(torch.median(torch.tensor([
        collector.full_head_effects[row.ordinal]
        for row in plan.fit if row.role == "target"
    ])))
    frame = torch.eye(128, dtype=torch.float64)[:, :1]
    result = collector.score_fixed_frame(
        select_relations=plan.select, frame=frame, control_id="planted"
    )
    assert len(result.target_cells) == 24
    assert len(result.control_cells) == 7
    assert sum(cell.coordinated_subject_cell for cell in result.target_cells.values()) == 2
    assert len(result.normalized_row_effects) == 106
    assert all(cell.full_head_fraction == pytest.approx(1.0)
               for cell in result.target_cells.values())
    assert all(cell.native_donor_recovery == pytest.approx(1.0)
               for cell in result.target_cells.values())
    assert result.model_counts == {
        "forward_calls": 5,
        "backward_calls": 0,
        "example_evaluations": 145,
    }


def test_householder_fit_rejects_nonorthogonal_start() -> None:
    with pytest.raises(backend.Task14BackendError, match="orthonormal"):
        backend.fit_householder_frame(
            torch.ones(8, 2, dtype=torch.float64),
            objective=lambda frame: frame.square().sum(),
            updates=1,
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
        )

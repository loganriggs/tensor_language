"""CPU/fake/planted tests for the narrow Task 14 tensor backend."""

from __future__ import annotations

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


def _fake_backend():
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
    )


def test_committed_shard_loads_only_discovery_endpoints() -> None:
    endpoints = backend.load_discovery_endpoints()
    assert len(endpoints) == 128
    assert len({endpoint.cell_metadata[0] for endpoint in endpoints.values()}) == 16
    assert all(endpoint.final_position == len(endpoint.token_ids) - 1
               for endpoint in endpoints.values())


def test_wrong_shard_hash_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "shard.json"
    changed.write_text(backend.SHARD_PATH.read_text() + " ")
    with pytest.raises(backend.Task14BackendError, match="hash changed"):
        backend.load_discovery_endpoints(changed)


def test_fake_spectral_collection_has_exact_shapes_effects_and_counts() -> None:
    plan = program.compile_discovery_plan()
    relations = tuple(row for row in plan.fit if row.role == "target")
    collector = _fake_backend()
    result = collector.collect_spectral_inputs(relations)
    assert result.ordinals == tuple(row.ordinal for row in relations)
    assert result.head_deltas.shape == (116, 128)
    assert result.downstream_gradients.shape == (116, 128)
    assert result.full_head_effects.shape == (116,)
    assert torch.all(result.full_head_effects > 0)
    assert result.model_counts == {
        "forward_calls": 6,
        "backward_calls": 2,
        "example_evaluations": 179,
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
    collector = _fake_backend()
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


def test_householder_fit_rejects_nonorthogonal_start() -> None:
    with pytest.raises(backend.Task14BackendError, match="orthonormal"):
        backend.fit_householder_frame(
            torch.ones(8, 2, dtype=torch.float64),
            objective=lambda frame: frame.square().sum(),
            updates=1,
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
        )

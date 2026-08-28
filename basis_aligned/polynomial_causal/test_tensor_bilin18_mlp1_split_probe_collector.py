from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import tensor_bilin18_mlp1_split_probe_collector as paired
from test_tensor_bilin18_program import tiny_program


def test_frozen_plan_rows_geometry_and_direct_parents_validate_without_gpu() -> None:
    plan, rows, directions = paired.load_plan_rows_geometry()
    assert plan["plan_fingerprint"] == paired.EXPECTED_PLAN_FINGERPRINT
    assert rows.shape == (16, 256)
    assert directions.shape == (32, 1152)
    assert paired.tensor_sha256(directions) == paired.EXPECTED_MLP1_DIRECTIONS_SHA256
    assert len(set(plan["selection"]["document_ids"])) == 16


def test_source_closure_contains_collector_plan_analysis_and_transitive_program() -> None:
    names = {path.name for path in paired.SOURCES}
    assert {
        Path(paired.__file__).name,
        "mlp1_split_probe_plan.json",
        "MLP1_SPLIT_PROBE_PREREGISTRATION.md",
        "finite_horizon_tangent_bundle.py",
        "tensor_bilin18_tangent_pilot.py",
        "tensor_bilin18_program.py",
        "tensor_preserving_attention.py",
        "tensor_preserving_mlp.py",
        "test_tensor_bilin18_mlp1_split_probe_collector.py",
    } <= names
    assert len(paired.SOURCES) == len(set(paired.SOURCES))


def tiny_transaction(program=None):
    program = tiny_program() if program is None else program
    tokens = torch.tensor([[0, 1, 2], [1, 2, 3]])
    directions = torch.eye(2, dtype=torch.float64)
    return paired.MLP1PairedProbeTransaction(
        program=program, tokens=tokens, row_ids=("r0", "r1"), directions=directions,
        first_probe_seeds=(11, 12), second_probe_seeds=(21, 22),
        injection_position=1, score_start=1, score_stop=3, production=False,
    ), tokens, directions


def test_tiny_paired_transaction_matches_explicit_future_score_vjp() -> None:
    program = tiny_program()
    transaction, tokens, directions = tiny_transaction(program)
    result = transaction.consume()
    assert transaction.aliases_revoked
    assert result.receipt["same_ordered_contexts"] is True
    assert result.receipt["probe_halves_disjoint"] is True
    assert set(result.first) == set(result.second) == {"r0", "r1"}
    assert all(value.shape == (2, 2) and value.dtype == torch.float64
               for value in (*result.first.values(), *result.second.values()))

    logits, leaves, _ = paired.tangent_collector._forward_with_additive_write_leaves(
        program, tokens, source_sites=(1,),
    )
    seeds = (11, 12, 21, 22)
    targets = paired.tangent_collector.stateless_categorical_fisher_targets(
        logits, ("r0", "r1"), seeds, score_start=1, score_stop=3,
    )
    logp = F.log_softmax(logits[:, 1:3].float(), dim=-1)
    for probe, seed in enumerate(seeds):
        selected = torch.gather(logp, -1, targets[probe].unsqueeze(-1)).squeeze(-1)
        gradient = torch.autograd.grad(
            selected.sum(), leaves[1], retain_graph=probe + 1 < len(seeds),
        )[0][:, 1].double() @ directions.T
        half = result.first if probe < 2 else result.second
        index = probe if probe < 2 else probe - 2
        actual = torch.stack([half[row][index] for row in ("r0", "r1")])
        torch.testing.assert_close(actual, gradient, rtol=1e-12, atol=1e-12)


def test_transaction_rejects_overlapping_probe_halves_before_forward() -> None:
    program = tiny_program()
    with pytest.raises(ValueError, match="disjoint"):
        paired.MLP1PairedProbeTransaction(
            program=program, tokens=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            row_ids=("r0", "r1"), directions=torch.eye(2, dtype=torch.float64),
            first_probe_seeds=(11, 12), second_probe_seeds=(12, 22),
            injection_position=1, score_start=1, score_stop=3, production=False,
        )


def test_transaction_revokes_graph_aliases_on_failure() -> None:
    program = tiny_program()
    transaction, _, _ = tiny_transaction(program)
    program.unembedding.fill_(float("nan"))
    with pytest.raises(ValueError, match="logits must be finite"):
        transaction.consume()
    assert transaction.aliases_revoked


def test_transaction_is_one_use() -> None:
    transaction, _, _ = tiny_transaction()
    transaction.consume()
    with pytest.raises(RuntimeError, match="spent"):
        transaction.consume()


def test_authority_paths_are_separate_create_only_namespaces() -> None:
    assert paired.AUTHORITY_RECEIPT != paired.OUTPUT
    assert paired.AUTHORITY_RECEIPT.name.endswith("authority_receipt.json")
    assert paired.OUTPUT.name.endswith("results.json")
    assert not paired.RUN_LOCK.name.endswith(".json")
    assert paired.EXPECTED_RANK640_SHA256 == "639fb8480efee790403113079333100bd63bb61426f6fd6e4dcebd89b21c337d"
    assert paired.EXPECTED_CAUSAL_SHA256 == "73bd18ee81067775680b7d579036e6ec8c04b41116cd3e516b8460a7e7c7ab20"

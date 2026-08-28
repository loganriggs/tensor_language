from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from finite_horizon_tangent_response_bank import TangentResponsePlan
from tensor_bilin18_tangent_collector import (
    PRODUCTION_TOKEN_VOCAB, PRODUCTION_VOCAB, PRODUCTION_WIDTH,
    TensorBilin18TangentTransaction,
    _forward_with_additive_write_leaves,
    collect_write_geometry_bank,
    stateless_categorical_fisher_targets,
    write_covariance_geometry,
)
from test_tensor_bilin18_program import tiny_program


def tiny_plan() -> TangentResponsePlan:
    return TangentResponsePlan(
        experiment_id="tiny-tangent-collector", row_artifact_sha256="c" * 64,
        row_ids=("r0", "r1"), document_ids=("d0", "d1"),
        splits=("primary", "replication"), scored_positions=(64, 65),
        input_dims=((0, 2), (1, 2), (2, 2)), target_site=3,
        probes_per_row=4, direction_seed=41, probe_seed=51, position_seed=61,
    )


def tiny_geometries():
    codes = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    return {
        site: write_covariance_geometry(codes + site * 0.1, site=site, direction_count=2, seed=41)
        for site in (0, 1, 2)
    }


def test_production_embedding_padding_is_distinct_from_valid_token_ids() -> None:
    assert PRODUCTION_WIDTH == 1152
    assert PRODUCTION_TOKEN_VOCAB == 50_257
    assert PRODUCTION_VOCAB == 50_304
    source = Path(__import__("tensor_bilin18_tangent_collector").__file__).read_text()
    assert "program.vocab_size != PRODUCTION_VOCAB" in source
    assert "int(tokens.max()) >= PRODUCTION_TOKEN_VOCAB" in source


def test_zero_write_leaves_preserve_exact_forward_and_all_indirect_paths() -> None:
    program = tiny_program()
    tokens = torch.tensor([[0, 1, 2], [3, 1, 0]])
    baseline = program(tokens)
    logits, leaves, receipt = _forward_with_additive_write_leaves(program, tokens)
    torch.testing.assert_close(logits, baseline, rtol=0, atol=0)
    assert receipt["attention_calls"] == tuple(range(18))
    assert set(leaves) == {0, 1, 2}
    # A late logit reads every earlier additive write through the actual downstream
    # tensor computation; detaching later writes would make this check weaker.
    gradients = torch.autograd.grad(logits[:, -1].square().sum(), tuple(leaves.values()))
    assert all(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_stateless_targets_are_batch_partition_invariant_and_use_full_support() -> None:
    logits = torch.randn(3, 4, 7)
    together = stateless_categorical_fisher_targets(
        logits, ("a", "b", "c"), (3, 5), score_start=0, score_stop=4,
    )
    pieces = torch.cat([
        stateless_categorical_fisher_targets(
            logits[index:index + 1], (row,), (3, 5), score_start=0, score_stop=4,
        )
        for index, row in enumerate(("a", "b", "c"))
    ], dim=1)
    assert torch.equal(together, pieces)
    assert together.shape == (2, 3, 4)
    assert int(together.min()) >= 0 and int(together.max()) < 7


def test_covariance_directions_are_reproducible_unit_rms_and_low_rank_admissible() -> None:
    codes = torch.tensor([
        [-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
    ])
    first = write_covariance_geometry(codes, site=0, direction_count=3, seed=7)
    second = write_covariance_geometry(codes, site=0, direction_count=3, seed=7)
    assert first.support_rank == 1
    assert first.covariance_sha256 == second.covariance_sha256
    assert first.directions_sha256 == second.directions_sha256
    torch.testing.assert_close(
        torch.sqrt(torch.mean(first.directions.square(), dim=1)),
        torch.ones(3, dtype=torch.float64), rtol=0, atol=1e-14,
    )
    assert torch.equal(first.directions[:, 1:], torch.zeros(3, 2, dtype=torch.float64))


def test_geometry_bank_collects_every_early_write_without_returning_raw_codes() -> None:
    program = tiny_program()
    plan = tiny_plan()
    rows = torch.tensor([
        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
    ])
    bank = collect_write_geometry_bank(
        program, rows, plan, batch_size=2, score_start=1, score_stop=4,
        production=False,
    )
    assert set(bank.geometries) == {0, 1, 2}
    assert bank.receipt["write_samples_per_site"] == 12
    assert bank.receipt["raw_write_codes_returned"] is False
    assert all(geometry.count == 12 and geometry.directions.shape == (2, 2)
               for geometry in bank.geometries.values())


def test_transaction_rejects_posthoc_short_code_contract_before_forward() -> None:
    program = tiny_program()
    with pytest.raises(ValueError, match="wrong shape"):
        TensorBilin18TangentTransaction(
            program=program, plan=tiny_plan(), row_ids=("r0", "r1"),
            tokens=torch.tensor([[0, 1, 2], [1, 2, 3]]), geometries=tiny_geometries(),
            production=True,
        )


def test_transaction_rejects_geometry_mutated_under_stale_hash() -> None:
    program = tiny_program()
    geometries = tiny_geometries()
    geometries[0].directions[0, 0] += 1
    with pytest.raises(ValueError, match="frozen hash"):
        TensorBilin18TangentTransaction(
            program=program, plan=tiny_plan(), row_ids=("r0", "r1"),
            tokens=torch.tensor([[0, 1, 2], [1, 2, 3]]), geometries=geometries,
            production=False,
        )


def test_tiny_transaction_emits_only_projected_cpu_rows_and_revokes() -> None:
    program = tiny_program()
    plan = tiny_plan()
    transaction = TensorBilin18TangentTransaction(
        program=program, plan=plan, row_ids=plan.row_ids,
        tokens=torch.tensor([[0, 1, 2], [1, 2, 3]]), geometries=tiny_geometries(),
        production=False,
    )
    result = transaction.consume()
    assert transaction.closed and transaction.aliases_revoked
    assert result.receipt["graph_aliases_revoked"]
    assert result.receipt["future_output_mask"]
    assert set(result.responses) == set(plan.row_ids)
    for rows in result.responses.values():
        assert set(rows) == {0, 1, 2}
        assert all(value.shape == (4, 2) and value.dtype == torch.float64
                   and value.device.type == "cpu" and not value.requires_grad
                   for value in rows.values())
    with pytest.raises(RuntimeError, match="spent"):
        transaction.consume()


def test_nonzero_injection_vjp_matches_explicit_future_only_score() -> None:
    program = tiny_program()
    plan = tiny_plan()
    tokens = torch.tensor([[0, 1, 2], [1, 2, 3]])
    geometries = tiny_geometries()
    positions = (1, 2)
    transaction = TensorBilin18TangentTransaction(
        program=program, plan=plan, row_ids=plan.row_ids, tokens=tokens,
        geometries=geometries, production=False,
        injection_positions_for_test=positions,
    )
    result = transaction.consume()

    logits, leaves, _ = _forward_with_additive_write_leaves(program, tokens)
    seeds = tuple(plan.probe_seed + index for index in range(plan.probes_per_row))
    targets = stateless_categorical_fisher_targets(
        logits, plan.row_ids, seeds, score_start=0, score_stop=tokens.shape[1],
    )
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    absolute = torch.arange(tokens.shape[1])
    mask = absolute.unsqueeze(0) >= torch.tensor(positions).unsqueeze(1)
    row_index = torch.arange(len(tokens))
    for probe in range(plan.probes_per_row):
        selected = torch.gather(
            log_probabilities, -1, targets[probe].unsqueeze(-1),
        ).squeeze(-1)
        gradients = torch.autograd.grad(
            (selected * mask).sum(), tuple(leaves.values()),
            retain_graph=probe + 1 < plan.probes_per_row,
        )
        for site, gradient in zip((0, 1, 2), gradients, strict=True):
            chosen = gradient[row_index, torch.tensor(positions)].double()
            expected = chosen @ geometries[site].directions.T
            actual = torch.stack([
                result.responses[row_id][site][probe] for row_id in plan.row_ids
            ])
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_transaction_revokes_on_injected_graph_failure() -> None:
    program = tiny_program()
    plan = tiny_plan()
    transaction = TensorBilin18TangentTransaction(
        program=program, plan=plan, row_ids=plan.row_ids,
        tokens=torch.tensor([[0, 1, 2], [1, 2, 3]]), geometries=tiny_geometries(),
        production=False,
    )
    program.unembedding.fill_(float("nan"))
    with pytest.raises(ValueError, match="logits must be finite"):
        transaction.consume()
    assert transaction.aliases_revoked

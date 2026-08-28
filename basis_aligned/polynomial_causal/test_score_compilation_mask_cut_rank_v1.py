from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import compilation_mask_cut_rank_v1 as cut
import compilation_mask_cut_rank_v1_measurements as measurement
import score_compilation_mask_cut_rank_v1 as score


def _interaction_batch(draws: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(991)
    left = torch.randn(draws, 7, 2, generator=generator, dtype=torch.float64)
    right = torch.randn(draws, 7, 2, generator=generator, dtype=torch.float64)
    value = torch.zeros((draws, 8, 8), dtype=torch.float64)
    value[:, 1:, 1:] = left @ right.transpose(1, 2)
    value[:, 1:, 1:] += 0.02 * torch.randn(
        (draws, 7, 7), generator=generator, dtype=torch.float64,
    )
    return value


def test_batched_fixed_rank_matches_exact_scalar_eight_restart_als():
    interactions = _interaction_batch()
    ridge = cut.RIDGE_GRID[4]
    observed = score._batched_fixed_rank_predictions(
        interactions, rank=2, ridge=ridge,
    )
    for draw in range(len(interactions)):
        values = {
            cell: float(interactions[draw, cell[0], cell[1]])
            for cell in (*cut.TRAIN_CELLS, *cut.VALIDATION_CELLS)
        }
        scale = math.sqrt(sum(values[cell] ** 2 for cell in cut.TRAIN_CELLS) / 28)
        model, _objective, _restart, _iterations = cut._fit_one_rank_model(
            values, rank=2, ridge=ridge, scale=scale,
        )
        expected = torch.tensor([
            [model.predict_interaction((i, j)) for j in range(1, 8)]
            for i in range(1, 8)
        ], dtype=torch.float64)
        assert torch.allclose(observed[draw], expected, atol=2e-10, rtol=2e-10)


def test_bootstrap_realization_and_literal_order_statistics_are_exact():
    first = score.document_bootstrap_weights(7)
    second = score.document_bootstrap_weights(7)
    assert torch.equal(first, second)
    assert first.shape == (score.BOOTSTRAP_REPETITIONS, 7)
    assert torch.equal(first.sum(1), torch.full(
        (score.BOOTSTRAP_REPETITIONS,), 7.0, dtype=torch.float64,
    ))
    values = torch.arange(score.BOOTSTRAP_REPETITIONS, dtype=torch.float64)
    bounds = score._one_sided_bounds(values)
    assert bounds["lower_95"] == score.LOWER_ORDER_ONE_INDEXED - 1
    assert bounds["upper_95"] == score.UPPER_ORDER_ONE_INDEXED - 1


def test_historical_singletons_are_source_bound_and_converted_to_raw_pp():
    singleton = score.load_historical_top1_singletons(score.REPO)
    assert singleton.target == "top1_pp"
    assert singleton.costs[("mlp", 5)] == pytest.approx(
        score.HISTORICAL_STAKE_PP * 0.6121697020739024
    )
    assert singleton.source_sha256 == score.adapter.file_sha256(
        score.REPO / score.HISTORICAL_SINGLETON_SOURCE
    )


def _synthetic_bundle() -> measurement.FinalizedMeasurementBundle:
    generator = torch.Generator().manual_seed(44)
    base = torch.full((3, 64), 850, dtype=torch.long)
    base -= torch.randint(0, 120, (3, 64), generator=generator)
    tokens = torch.full((3,), 1_000, dtype=torch.long)
    ce = 500.0 + 60.0 * torch.rand(
        (3, 64), generator=generator, dtype=torch.float64,
    )
    authority_sha = "a" * 64
    documents_sha = "b" * 64
    payload = measurement.PerDocumentSufficientStatistics(
        authority_sha256=authority_sha,
        ordered_document_ids_sha256=documents_sha,
        document_row_count=torch.ones(3, dtype=torch.long),
        document_token_count=tokens,
        top1_correct=base.contiguous(), ce_sum=ce.contiguous(),
    )
    vectors = tuple(f"{index:064x}" for index in range(1, 65))
    receipt = measurement.MeasurementReceipt(
        schema_version=measurement.SCHEMA_VERSION,
        preregistration_sha256=measurement.PREREGISTRATION_SHA256,
        request_plan_sha256=measurement.REQUEST_PLAN_SHA256,
        authority_sha256=authority_sha, source_commit="1" * 40,
        source_receipt_sha256="c" * 64, row_tensor_sha256="d" * 64,
        row_provenance_sha256="e" * 64, ordered_row_identity_sha256="f" * 64,
        ordered_row_to_document_sha256="1" * 64,
        ordered_document_ids_sha256=documents_sha, row_token_count_sha256="2" * 64,
        common_support_sha256="3" * 64, model_realization_sha256="4" * 64,
        component_tree_sha256="5" * 64, program_bank_sha256="6" * 64,
        source_closure_sha256="7" * 64, wave_nonce_sha256="8" * 64,
        b0_request_sha256=measurement.REQUESTS[0].sha256,
        b0_cell_receipt_sha256=vectors[0], cell_receipt_sha256s=vectors,
        top1_correct_row_sha256s=vectors, ce_sum_row_sha256s=vectors,
        statistics_sha256s=vectors, per_document_payload_sha256=payload.sha256,
        row_count=3, document_count=3, total_scored_token_count=3_000,
        batch_count=1, cell_count=64,
    )
    return measurement.FinalizedMeasurementBundle(payload=payload, receipt=receipt)


def test_development_is_published_before_one_shot_heldout_and_pass_stays_none(
    monkeypatch,
):
    events = []
    original_finalize = cut.finalize_heldout

    def finalize_after_publication(development, heldout):
        assert events == ["development"]
        events.append("heldout")
        return original_finalize(development, heldout)

    monkeypatch.setattr(cut, "finalize_heldout", finalize_after_publication)
    monkeypatch.setattr(
        score, "document_bootstrap_weights",
        lambda _documents: torch.ones(
            (score.BOOTSTRAP_REPETITIONS, 3), dtype=torch.float64,
        ),
    )

    def fixed_metrics(*_args, **_kwargs):
        values = torch.linspace(
            0.1, 0.2, score.BOOTSTRAP_REPETITIONS, dtype=torch.float64,
        )
        return {
            "interaction_nre": values, "heldout_r2": values,
            "rmse_ratio": values,
            "full_grid_rank2_spectral_tail_nre": values,
        }

    monkeypatch.setattr(score, "fixed_selection_bootstrap_metrics", fixed_metrics)
    result = score.score_bundle(
        _synthetic_bundle(), repo=score.REPO,
        development_publisher=lambda _value: events.append("development"),
    )
    assert events == ["development", "heldout"]
    assert result["registered_ce_baseline_family_complete"] is False
    assert result["useful_pass"] is None and result["promotive"] is False

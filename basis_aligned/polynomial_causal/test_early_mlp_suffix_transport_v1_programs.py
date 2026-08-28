from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest
import torch

import early_mlp_suffix_transport_v1_fit as fit
import early_mlp_suffix_transport_v1_programs as programs
import early_mlp_suffix_transport_v1_runtime as runtime


def _state(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "grammar": "affine", "interface": "state_complete_p",
        "mean": torch.randn(runtime.D_MODEL, generator=generator) / 50,
        "scale": torch.rand(runtime.D_MODEL, generator=generator) + 0.5,
        "left": torch.randn(
            runtime.D_MODEL, runtime.CODE_DIM, generator=generator,
        ) / 50,
        "right": torch.randn(
            runtime.CODE_DIM, runtime.CODE_DIM, generator=generator,
        ) / 50,
        "bias": torch.randn(runtime.CODE_DIM, generator=generator) / 50,
    }


def _fit(route: str, trial: int, seed: int) -> fit.FitCandidate:
    program = runtime.JointAffineProgram.from_v21_states(
        {0: _state(seed), 1: _state(seed + 1)}, route=route,
    )
    state = MappingProxyType({
        name: value.detach().cpu().clone() for name, value in program.state_dict().items()
    })
    return fit.FitCandidate(
        route=route, trial=trial, learning_rate=runtime.LEARNING_RATES[trial],
        completed_steps=288, loss_sum=1.0, loss_min=0.0, loss_max=1.0,
        final_program_sha256=runtime.program_snapshot_sha256(program),
        transaction_history_sha256="6" * 64, state_dict=state,
    )


def _score(candidate: fit.FitCandidate, metric: float, *, copy: float = 0.0):
    return programs.ScoredCandidate(
        fit_candidate=candidate,
        validation=programs.ValidationScore(
            route=candidate.route, trial=candidate.trial,
            learning_rate=candidate.learning_rate,
            program_sha256=candidate.final_program_sha256,
            metric_name=programs.METRIC_BY_ROUTE[candidate.route],
            primary_metric=metric, copy_worsening=copy,
            scored_token_count=programs.VALIDATION_SCORED_TOKENS,
            common_support_sha256="7" * 64,
            sufficient_statistics_sha256="8" * 64,
            student_original_calls=programs.ZERO_NATIVE_CALLS,
            hook_restored=True, hook_inert=True,
        ),
    )


def test_selector_applies_metric_lr_hash_order_and_copy_gate() -> None:
    low_lr = _score(_fit("R", 0, 1), 0.2)
    middle_lr = _score(_fit("R", 1, 3), 0.2)
    lower_metric = _score(_fit("R", 2, 5), 0.1)
    assert programs.select_candidate(
        [low_lr, middle_lr, lower_metric], route="R",
    ) is lower_metric
    rejected = _score(_fit("R", 2, 7), 0.01, copy=0.0100001)
    assert programs.select_candidate([middle_lr, rejected, low_lr], route="R") is low_lr

    same_trial_a = _score(_fit("R", 0, 9), 0.2)
    same_trial_b = _score(_fit("R", 0, 11), 0.2)
    with pytest.raises(ValueError, match="exactly once"):
        programs.select_candidate([same_trial_a, same_trial_b, lower_metric], route="R")
    with pytest.raises(RuntimeError, match="copy bound"):
        programs.select_candidate([
            replace(low_lr, validation=replace(low_lr.validation, copy_worsening=0.02)),
            replace(middle_lr, validation=replace(middle_lr.validation, copy_worsening=0.02)),
            rejected,
        ], route="R")


def test_validation_receipt_rejects_wrong_support_calls_and_metric() -> None:
    candidate = _fit("L", 0, 1)
    valid = _score(candidate, 0.5).validation
    with pytest.raises(ValueError, match="support count"):
        replace(valid, scored_token_count=programs.VALIDATION_SCORED_TOKENS - 1)
    with pytest.raises(ValueError, match="original"):
        replace(valid, student_original_calls=((0, 1), (1, 0), (2, 0)))
    with pytest.raises(ValueError, match="route/trial/metric"):
        replace(valid, metric_name="oon_teacher_kl")
    with pytest.raises(ValueError, match="differs"):
        programs.ScoredCandidate(
            fit_candidate=candidate,
            validation=replace(valid, program_sha256="9" * 64),
        )


def test_signed_float64_svd_freeze_replays_and_roundtrips() -> None:
    selected = _score(_fit("S0", 1, 13), 0.25)
    frozen = programs.freeze_selected(selected)
    assert frozen.route == "S0" and max(frozen.svd_max_errors) <= 2e-6
    restored = frozen.make_program()
    source = programs.restore_fit_candidate(selected.fit_candidate)
    for site in (0, 1):
        source_affine = source.site0 if site == 0 else source.site1
        restored_affine = restored.site0 if site == 0 else restored.site1
        torch.testing.assert_close(
            restored_affine.weight, source_affine.weight, atol=2e-6, rtol=0,
        )
        state = frozen.site_states[site]
        u_direction = state["left"].double() / torch.linalg.vector_norm(
            state["left"].double(), dim=0,
        )
        for column in range(runtime.CODE_DIM):
            pivot = int(torch.argmax(torch.abs(u_direction[:, column])))
            assert float(u_direction[pivot, column]) >= 0


def test_transport_initialization_is_selected_l_plus_zero_cross_only() -> None:
    selected = programs.freeze_selected(_score(_fit("L", 2, 17), 0.3))
    local = selected.make_program()
    transport = programs.make_transport_initialization(selected)
    assert transport.route == "T"
    assert transport.trainable_parameter_names == ("cross",)
    assert transport.cross is not None and int(torch.count_nonzero(transport.cross)) == 0
    for site_name in ("site0", "site1"):
        local_site = getattr(local, site_name)
        transport_site = getattr(transport, site_name)
        assert all(not parameter.requires_grad for parameter in transport_site.parameters())
        for key, value in local_site.state_dict().items():
            torch.testing.assert_close(value, transport_site.state_dict()[key], rtol=0, atol=0)


def test_trained_transport_cross_survives_fit_restore_and_svd_freeze() -> None:
    candidate = _fit("T", 1, 19)
    changed_state = dict(candidate.state_dict)
    changed_state["cross"] = torch.arange(
        runtime.CODE_DIM * runtime.CODE_DIM, dtype=torch.float32,
    ).view(runtime.CODE_DIM, runtime.CODE_DIM) / 1000
    transport = runtime.JointAffineProgram.from_v21_states(
        {0: _state(19), 1: _state(20)}, route="T",
    )
    transport.load_state_dict(changed_state)
    candidate = replace(
        candidate, state_dict=MappingProxyType(changed_state),
        final_program_sha256=runtime.program_snapshot_sha256(transport),
    )
    scored = _score(candidate, 0.2)
    restored = programs.restore_fit_candidate(candidate)
    torch.testing.assert_close(restored.cross, changed_state["cross"], rtol=0, atol=0)
    frozen = programs.freeze_selected(scored)
    assert frozen.cross is not None
    roundtrip = frozen.make_program()
    torch.testing.assert_close(roundtrip.cross, changed_state["cross"], rtol=0, atol=0)
    assert roundtrip.trainable_parameter_names == ("cross",)


def test_four_route_freezer_requires_complete_nonmixed_banks() -> None:
    banks = {
        route: [
            _score(_fit(route, trial, 20 + 3 * index + trial), 1.0 - trial / 10)
            for trial in range(3)
        ]
        for index, route in enumerate(fit.TRUE_FIT_ROUTES)
    }
    frozen = programs.select_and_freeze_routes(banks)
    assert set(frozen) == set(fit.TRUE_FIT_ROUTES)
    with pytest.raises(ValueError, match="exactly"):
        programs.select_and_freeze_routes({"L": banks["L"]})


def test_validation_score_is_recomputed_from_raw_row_statistics(monkeypatch) -> None:
    monkeypatch.setattr(programs, "VALIDATION_ROWS", 2)
    monkeypatch.setattr(programs, "VALIDATION_SCORED_TOKENS", 384)
    generator = torch.Generator().manual_seed(81)
    predictions = tuple(
        torch.randn(2, 192, runtime.CODE_DIM, generator=generator) for _ in range(2)
    )
    labels = tuple(value + 0.1 for value in predictions)
    denominators = (2.0, 4.0)
    primary_sum, primary_count = programs.local_primary_rows(
        predictions, labels, denominators,
    )
    expected = runtime.normalized_local_loss(predictions, labels, denominators)
    torch.testing.assert_close(
        primary_sum.sum() / primary_count.sum(), expected.double(), rtol=2e-6, atol=1e-9,
    )

    rows = torch.arange(2 * 513, dtype=torch.long).view(2, 513) % 7
    logits = torch.randn(2, 192, 7, generator=generator)
    ce_sum, ce_count, copy_sum, copy_count = programs.ce_and_copy_rows(logits, rows)
    baseline_copy = copy_sum - 0.001 * copy_count
    candidate = _fit("L", 0, 31)
    statistics = programs.ValidationSufficientStatistics(
        route="L", program_sha256=candidate.final_program_sha256,
        common_support_sha256="a" * 64,
        row_primary_sum=primary_sum, row_primary_count=primary_count,
        row_ce_sum=ce_sum, row_ce_count=ce_count,
        row_copy_ce_sum=copy_sum, row_copy_count=copy_count,
        baseline_row_copy_ce_sum=baseline_copy,
        baseline_row_copy_count=copy_count,
        student_original_calls=programs.ZERO_NATIVE_CALLS,
        hook_restored=True, hook_inert=True,
    )
    score = programs.validation_score_from_statistics(candidate, statistics)
    assert score.primary_metric == float(primary_sum.sum() / primary_count.sum())
    assert score.copy_worsening == pytest.approx(0.001)
    assert score.sufficient_statistics_sha256 == statistics.sha256
    scored = programs.ScoredCandidate(candidate, score)
    assert scored.validation is score


def test_suffix_kl_row_pool_matches_registered_token_weighted_loss() -> None:
    generator = torch.Generator().manual_seed(91)
    teacher = torch.randn(3, 256, 11, generator=generator)
    student = teacher + torch.randn(3, 256, 11, generator=generator) / 10
    row_sum, row_count = programs.suffix_kl_rows(teacher, student)
    expected = runtime.teacher_student_kl(teacher, student)
    torch.testing.assert_close(
        row_sum.sum() / row_count.sum(), expected.double(), rtol=2e-6, atol=1e-9,
    )

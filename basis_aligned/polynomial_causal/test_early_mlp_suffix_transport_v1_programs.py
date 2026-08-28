from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest
import torch

import early_mlp_suffix_transport_v1 as contract
import early_mlp_suffix_transport_v1_capabilities as capabilities
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


def _validation_context() -> capabilities.ValidationRunContext:
    return capabilities.ValidationRunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, validation_role_tensor_sha256="4" * 64,
        identity_teacher_mapping_sha256="5" * 64,
    )


def _mapped_fit(
    control: str, route: str, trial: int, seed: int, *, mapping: str = "9" * 64,
) -> fit.MappedFitCandidate:
    candidate = _fit(route, trial, seed)
    return fit.MappedFitCandidate(
        control=control, mapping_sha256=mapping,
        route=route, trial=trial, learning_rate=candidate.learning_rate,
        completed_steps=candidate.completed_steps, loss_sum=candidate.loss_sum,
        loss_min=candidate.loss_min, loss_max=candidate.loss_max,
        final_program_sha256=candidate.final_program_sha256,
        transaction_history_sha256=candidate.transaction_history_sha256,
        state_dict=candidate.state_dict,
    )


def _mapped_score(candidate: fit.MappedFitCandidate, metric: float, *, copy: float = 0.0):
    base = programs.ValidationScore(
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
    )
    validation = programs.MappedValidationScore(
        control=candidate.control, mapping_sha256=candidate.mapping_sha256,
        base=base, mapped_sufficient_statistics_sha256="a" * 64,
    )
    return programs.MappedScoredCandidate(candidate, validation)


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


def test_mapped_selector_and_freezer_preserve_control_plan_and_separate_type() -> None:
    candidates = [
        _mapped_score(_mapped_fit("A_null_00", "T", trial, 60 + trial), 0.3 - trial / 20)
        for trial in range(3)
    ]
    selected = programs.select_mapped_candidate(
        candidates, control="A_null_00", route="T",
    )
    assert selected.validation.trial == 2
    frozen = programs.freeze_mapped_selected(selected)
    assert isinstance(frozen, programs.FrozenMappedProgram)
    assert not isinstance(frozen, programs.FrozenProgram)
    assert frozen.key == "A_null_00/T" and frozen.mapping_sha256 == "9" * 64
    torch.testing.assert_close(
        frozen.make_program().cross,
        programs.restore_mapped_fit_candidate(selected.fit_candidate).cross,
        rtol=0, atol=0,
    )
    mixed = list(candidates)
    mixed[0] = _mapped_score(
        _mapped_fit("A_null_01", "T", 0, 70, mapping="b" * 64), 0.1,
    )
    with pytest.raises(ValueError, match="mixes"):
        programs.select_mapped_candidate(mixed, control="A_null_00", route="T")
    with pytest.raises(ValueError, match="registered"):
        programs.mapped_control_key("A_null_20", "T")


def test_all_24_mapped_families_are_required_and_plan_distinct(monkeypatch) -> None:
    banks = {}
    document_mapping = "1" * 64
    for key in programs.required_mapped_control_keys():
        control, route = key.split("/", 1)
        mapping = document_mapping if control == "document_shuffle" else (
            f"{int(control[-2:]) + 2:064x}"
        )
        banks[key] = [
            _mapped_score(
                _mapped_fit(control, route, trial, 80 + trial, mapping=mapping),
                0.5 + trial,
            )
            for trial in range(3)
        ]

    templates = {
        route: programs.freeze_selected(_score(_fit(route, 0, 120 + index), 0.1))
        for index, route in enumerate(programs.SELECTABLE_ROUTES)
    }
    monkeypatch.setattr(
        programs, "freeze_mapped_selected",
        lambda candidate: programs.FrozenMappedProgram(
            control=candidate.fit_candidate.control,
            mapping_sha256=candidate.fit_candidate.mapping_sha256,
            mapped_sufficient_statistics_sha256=(
                candidate.validation.mapped_sufficient_statistics_sha256
            ),
            program=templates[candidate.fit_candidate.route],
        ),
    )
    frozen = programs.select_and_freeze_mapped_controls(banks)
    assert tuple(frozen) == programs.required_mapped_control_keys()
    assert len(frozen) == 24
    missing = dict(banks); missing.pop("A_null_19/T")
    with pytest.raises(ValueError, match="exactly 24"):
        programs.select_and_freeze_mapped_controls(missing)

    duplicated = dict(banks)
    duplicate_bank = []
    for item in duplicated["A_null_19/T"]:
        duplicate_bank.append(programs.MappedScoredCandidate(
            replace(item.fit_candidate, mapping_sha256=f"{2:064x}"),
            replace(item.validation, mapping_sha256=f"{2:064x}"),
        ))
    duplicated["A_null_19/T"] = duplicate_bank
    with pytest.raises(RuntimeError, match="duplicated"):
        programs.select_and_freeze_mapped_controls(duplicated)


def test_mapped_raw_statistics_bind_control_and_plan_before_selection(monkeypatch) -> None:
    monkeypatch.setattr(programs, "VALIDATION_ROWS", 2)
    monkeypatch.setattr(programs, "VALIDATION_SCORED_TOKENS", 384)
    candidate = _mapped_fit("A_null_03", "T", 0, 140, mapping="c" * 64)
    count = torch.full((2,), 192, dtype=torch.long)
    copy_count = torch.tensor([2, 3], dtype=torch.long)
    base = programs.ValidationSufficientStatistics(
        route="T", program_sha256=candidate.final_program_sha256,
        common_support_sha256="d" * 64,
        row_primary_sum=torch.tensor([1.0, 2.0], dtype=torch.float64),
        row_primary_count=count,
        row_ce_sum=torch.tensor([3.0, 4.0], dtype=torch.float64),
        row_ce_count=count,
        row_copy_ce_sum=torch.tensor([0.2, 0.3], dtype=torch.float64),
        row_copy_count=copy_count,
        baseline_row_copy_ce_sum=torch.tensor([0.19, 0.285], dtype=torch.float64),
        baseline_row_copy_count=copy_count,
        student_original_calls=programs.ZERO_NATIVE_CALLS,
        hook_restored=True, hook_inert=True,
    )
    statistics = programs.MappedValidationSufficientStatistics(
        control="A_null_03", mapping_sha256="c" * 64, base=base,
    )
    score = programs.mapped_validation_score_from_statistics(candidate, statistics)
    assert score.control == "A_null_03" and score.mapping_sha256 == "c" * 64
    assert score.primary_metric == pytest.approx(3 / 384)
    assert score.mapped_sufficient_statistics_sha256 == statistics.sha256
    with pytest.raises(ValueError, match="differ"):
        programs.mapped_validation_score_from_statistics(
            candidate, replace(statistics, mapping_sha256="e" * 64),
        )


def test_preflight_gauge_bank_is_exact_reproducible_and_orthogonal() -> None:
    first = programs.orthogonal_gauge_bank()
    second = programs.orthogonal_gauge_bank()
    assert tuple(first) == (
        "signed_permutation_0", "signed_permutation_1",
        "signed_permutation_2", "signed_permutation_3",
        "haar_0", "haar_1", "haar_2", "haar_3",
    )
    for name, gauge in first.items():
        assert gauge.dtype == torch.float64 and tuple(gauge.shape) == (64, 64)
        assert torch.equal(gauge, second[name])
        contract.validate_orthogonal_gauge(name, gauge)
        if name.startswith("signed"):
            assert torch.equal(
                torch.count_nonzero(gauge, dim=0), torch.ones(64, dtype=torch.long),
            )
            assert torch.equal(
                torch.count_nonzero(gauge, dim=1), torch.ones(64, dtype=torch.long),
            )


def test_intervention_assignments_are_role_specific_reproducible_and_balanced() -> None:
    validation = programs.intervention_assignments("validation")
    replay = programs.intervention_assignments("validation")
    final = programs.intervention_assignments("final")
    assert all(torch.equal(value, replay[name]) for name, value in validation.items())
    assert not torch.equal(validation["positions"], final["positions"])
    assert not torch.equal(validation["row_permutation"], final["row_permutation"])
    for assignment in (validation, final):
        assert bool(((assignment["positions"] >= 64) & (
            assignment["positions"] <= 255
        )).all())
        assert torch.equal(
            torch.bincount(assignment["direction_indices"], minlength=32),
            torch.full((32,), 6, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="validation or final"):
        programs.intervention_assignments("fit")


def test_transport_intervention_geometry_matches_frozen_fit_covariance() -> None:
    generator = torch.Generator().manual_seed(611)
    codes = torch.randn(
        capabilities.FIT_ROW_COUNT, 192, runtime.CODE_DIM, generator=generator,
    )
    geometry = programs.build_transport_intervention_geometry(
        codes, selected_l_program_sha256="a" * 64,
        fit_role_tensor_sha256="b" * 64,
    )
    flat = codes.double().reshape(-1, runtime.CODE_DIM)
    centered = flat - flat.mean(dim=0)
    expected_covariance = centered.T @ centered / (len(flat) - 1)
    torch.testing.assert_close(geometry.mean, flat.mean(dim=0), rtol=0, atol=0)
    torch.testing.assert_close(
        geometry.covariance, expected_covariance, rtol=2e-13, atol=2e-13,
    )
    assert geometry.code_count == capabilities.FIT_ROW_COUNT * 192
    assert geometry.natural_rms == float(torch.sqrt(torch.mean(centered.square())))
    assert torch.allclose(
        torch.sqrt(torch.mean(geometry.normalized_directions.square(), dim=1)),
        torch.ones(32, dtype=torch.float64), rtol=2e-12, atol=2e-12,
    )
    first_draw = 2 * torch.randint(
        0, 2, (64,), dtype=torch.long,
        generator=torch.Generator().manual_seed(2026083200),
    ) - 1
    assert torch.equal(geometry.raw_rademacher_signs[0], first_draw)
    before = geometry.sha256
    codes.zero_()
    assert geometry.sha256 == before


def test_transport_intervention_geometry_rejects_degenerate_or_partial_fit_codes() -> None:
    with pytest.raises(ValueError, match="trajectory is malformed"):
        programs.build_transport_intervention_geometry(
            torch.zeros(4, 192, 64), selected_l_program_sha256="a" * 64,
            fit_role_tensor_sha256="b" * 64,
        )
    with pytest.raises(RuntimeError, match="covariance is degenerate"):
        programs.build_transport_intervention_geometry(
            torch.zeros(capabilities.FIT_ROW_COUNT, 192, 64),
            selected_l_program_sha256="a" * 64,
            fit_role_tensor_sha256="b" * 64,
        )


def _canonical_bank_inputs():
    support = "7" * 64
    true_programs = {
        route: programs.freeze_selected(_score(_fit(route, 0, 700 + index), 0.1))
        for index, route in enumerate(programs.SELECTABLE_ROUTES)
    }
    document_mapping = "1" * 64
    mapped_programs = {}
    for key in programs.required_mapped_control_keys():
        control, route = key.split("/", 1)
        mapping = document_mapping if control == "document_shuffle" else (
            f"{int(control[-2:]) + 2:064x}"
        )
        mapped_programs[key] = programs.FrozenMappedProgram(
            control=control, mapping_sha256=mapping,
            mapped_sufficient_statistics_sha256="a" * 64,
            program=true_programs[route],
        )
    count = torch.full((programs.VALIDATION_ROWS,), 192, dtype=torch.long)
    copy_count = torch.ones(programs.VALIDATION_ROWS, dtype=torch.long)
    baseline = programs.ValidationBaselineSufficientStatistics(
        common_support_sha256=support,
        row_ce_sum=torch.ones(programs.VALIDATION_ROWS, dtype=torch.float64),
        row_ce_count=count,
        row_copy_ce_sum=torch.ones(programs.VALIDATION_ROWS, dtype=torch.float64),
        row_copy_count=copy_count,
        literal_early_mlp_calls=programs.ZERO_NATIVE_CALLS,
        native_guard_restored=True, native_guard_inert=True,
    )
    keys = programs.required_validation_candidate_keys()
    statistics = {key: "b" * 64 for key in keys}
    for route, frozen in true_programs.items():
        statistics[f"true/{route}/trial{frozen.trial}"] = (
            frozen.validation_sufficient_statistics_sha256
        )
    for key, frozen in mapped_programs.items():
        statistics[f"{key}/trial{frozen.program.trial}"] = (
            frozen.mapped_sufficient_statistics_sha256
        )
    receipts = tuple("c" * 64 for _ in range(capabilities.VALIDATION_BATCH_COUNT))
    execution = programs.ValidationExecutionManifest(
        validation_role_tensor_sha256="4" * 64,
        common_support_sha256=support,
        baseline_statistics_sha256=baseline.sha256,
        baseline_batch_receipt_sha256s=receipts,
        candidate_batch_receipt_sha256s={key: receipts for key in keys},
        candidate_statistics_sha256s=statistics,
        broker_ledger_sha256s={key: "d" * 64 for key in keys},
    )
    codes = torch.randn(
        capabilities.FIT_ROW_COUNT, 192, runtime.CODE_DIM,
        generator=torch.Generator().manual_seed(811),
    )
    geometry = programs.build_transport_intervention_geometry(
        codes,
        selected_l_program_sha256=true_programs["L"].canonical_tensor_sha256,
        fit_role_tensor_sha256="5" * 64,
    )
    calibration = programs.select_teacher_calibration({
        0.01: 0.005, 0.03: 0.02, 0.1: 0.05, 0.3: 0.19, 1.0: 0.4,
    })
    return true_programs, mapped_programs, baseline, execution, geometry, calibration


def test_validation_execution_manifest_requires_all_87_complete_candidates() -> None:
    assert len(programs.required_validation_candidate_keys()) == 87
    *_, execution, _, _ = _canonical_bank_inputs()
    assert len(execution.candidate_statistics_sha256s) == 87
    incomplete = dict(execution.candidate_statistics_sha256s)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="incomplete"):
        replace(execution, candidate_statistics_sha256s=incomplete)
    bad_receipts = dict(execution.candidate_batch_receipt_sha256s)
    bad_receipts[next(iter(bad_receipts))] = bad_receipts[
        next(iter(bad_receipts))
    ][:-1]
    with pytest.raises(ValueError, match="hash/count"):
        replace(execution, candidate_batch_receipt_sha256s=bad_receipts)


def test_canonical_program_bank_binds_selection_controls_geometry_and_payload(tmp_path) -> None:
    true, mapped, baseline, execution, geometry, calibration = _canonical_bank_inputs()
    bank = programs.build_canonical_program_bank(
        true_programs=true, mapped_programs=mapped,
        validation_baseline=baseline, validation_execution=execution,
        transport_geometry=geometry, teacher_calibration=calibration,
    )
    assert set(bank["true_programs"]) == set(programs.SELECTABLE_ROUTES)
    assert tuple(bank["mapped_programs"]) == programs.required_mapped_control_keys()
    assert len(bank["validation_execution"]["candidate_statistics_sha256s"]) == 87
    assert bank["transport_geometry"]["geometry_sha256"] == geometry.sha256
    assert bank["payload_sha256"] == runtime.logical_identity_sha256(
        programs._payload_identity({
            key: value for key, value in bank.items() if key != "payload_sha256"
        })
    )
    for gauge in bank["gauge_bank"].values():
        contract.validate_orthogonal_gauge("stored", gauge)
    artifact = tmp_path / "programs.pt"
    torch.save(bank, artifact)
    reloaded = torch.load(artifact, map_location="cpu", weights_only=True)
    validated = programs.validate_canonical_program_bank_payload(reloaded)
    assert validated["payload_sha256"] == bank["payload_sha256"]
    assert validated["true_programs"]["L"].canonical_tensor_sha256 == (
        true["L"].canonical_tensor_sha256
    )

    with pytest.raises(ValueError, match="support-mixed"):
        programs.build_canonical_program_bank(
            true_programs=true, mapped_programs=mapped,
            validation_baseline=replace(baseline, common_support_sha256="e" * 64),
            validation_execution=execution, transport_geometry=geometry,
            teacher_calibration=calibration,
        )
    bad_mapped = dict(mapped)
    bad_mapped["A_null_19/T"] = replace(
        bad_mapped["A_null_19/T"],
        mapping_sha256=bad_mapped["A_null_18/T"].mapping_sha256,
    )
    with pytest.raises(RuntimeError, match="duplicated"):
        programs.build_canonical_program_bank(
            true_programs=true, mapped_programs=bad_mapped,
            validation_baseline=baseline, validation_execution=execution,
            transport_geometry=geometry, teacher_calibration=calibration,
        )
    with pytest.raises(RuntimeError, match="selected L"):
        programs.build_canonical_program_bank(
            true_programs=true, mapped_programs=mapped,
            validation_baseline=baseline, validation_execution=execution,
            transport_geometry=replace(
                geometry, selected_l_program_sha256="e" * 64,
            ),
            teacher_calibration=calibration,
        )
    changed_calibration = dict(calibration)
    changed_calibration["selected_amplitude_multiplier"] = 1.0
    with pytest.raises(RuntimeError, match="selection rule"):
        programs.build_canonical_program_bank(
            true_programs=true, mapped_programs=mapped,
            validation_baseline=baseline, validation_execution=execution,
            transport_geometry=geometry, teacher_calibration=changed_calibration,
        )

    changed_tensor = torch.load(artifact, map_location="cpu", weights_only=True)
    changed_tensor["true_programs"]["L"]["site_states"]["0"]["bias"][0] += 1
    with pytest.raises(RuntimeError, match="payload hash"):
        programs.validate_canonical_program_bank_payload(changed_tensor)

    changed_gauge = torch.load(artifact, map_location="cpu", weights_only=True)
    changed_gauge["gauge_bank"]["haar_0"] = torch.eye(runtime.CODE_DIM, dtype=torch.float64)
    changed_gauge["payload_sha256"] = runtime.logical_identity_sha256(
        programs._payload_identity({
            key: value for key, value in changed_gauge.items() if key != "payload_sha256"
        })
    )
    with pytest.raises(RuntimeError, match="gauge bank"):
        programs.validate_canonical_program_bank_payload(changed_gauge)


def test_teacher_only_calibration_selects_in_band_or_fails_closed() -> None:
    values = {0.01: 0.005, 0.03: 0.02, 0.1: 0.05, 0.3: 0.19, 1.0: 0.4}
    selected = programs.select_teacher_calibration(values)
    assert selected["calibration_passed"] is True
    assert selected["selected_amplitude_multiplier"] == 0.1
    assert selected["selected_teacher_median_kl"] == 0.05

    outside = {0.01: 0.001, 0.03: 0.002, 0.1: 0.003, 0.3: 0.4, 1.0: 0.8}
    failed = programs.select_teacher_calibration(outside)
    assert failed["calibration_passed"] is False
    assert failed["selected_amplitude_multiplier"] == 0.1
    with pytest.raises(ValueError, match="five"):
        programs.select_teacher_calibration({0.01: 0.1})


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
    mixed_sum, mixed_count = programs.suffix_kl_rows(
        runtime.scored_positions(teacher), student,
    )
    torch.testing.assert_close(mixed_sum, row_sum)
    torch.testing.assert_close(mixed_count, row_count)


def test_common_validation_support_binds_rows_targets_mask_and_semantics(monkeypatch) -> None:
    monkeypatch.setattr(programs, "VALIDATION_ROWS", 8)
    rows = torch.arange(8 * 257, dtype=torch.long).view(8, 257) % 19
    first = programs.validation_common_support_sha256(rows)
    assert first == programs.validation_common_support_sha256(rows.clone())
    changed = rows.clone()
    changed[4, 100] = (changed[4, 100] + 1) % 19
    assert programs.validation_common_support_sha256(changed) != first
    with pytest.raises(ValueError, match="all frozen role rows"):
        programs.validation_common_support_sha256(rows[:4])


@pytest.mark.parametrize("control,route", [
    ("true", "L"), ("true", "R"), ("document_shuffle", "S1"),
    ("A_null_07", "T"),
])
def test_validation_identity_binds_true_rows_without_erasing_fit_control(
    control, route,
) -> None:
    context = _validation_context()
    program = programs.restore_fit_candidate(_fit(route, 1, 240))
    batch = 7
    indices = tuple(range(batch * 4, batch * 4 + 4))
    inputs = torch.arange(4 * 256, dtype=torch.long).view(4, 256) % 11
    identity = programs.make_validation_identity(
        context=context, program=program, inputs=inputs, indices=indices,
        route=route, control=control, trial=1, batch_ordinal=batch,
    )
    assert identity.role == "early_mlp_suffix_transport_v1_validation"
    assert identity.phase == "validation" and identity.control == control
    assert identity.teacher_mapping_sha256 == context.identity_teacher_mapping_sha256
    assert identity.fit_role_tensor_sha256 == context.validation_role_tensor_sha256
    context.require_identity(identity, inputs, indices)


def test_validation_identity_rejects_noncanonical_rows_and_illegal_control() -> None:
    context = _validation_context()
    program = programs.restore_fit_candidate(_fit("R", 0, 250))
    inputs = torch.zeros(4, 256, dtype=torch.long)
    with pytest.raises(RuntimeError, match="canonical"):
        programs.make_validation_identity(
            context=context, program=program, inputs=inputs, indices=(1, 2, 3, 4),
            route="R", control="true", trial=0, batch_ordinal=0,
        )
    with pytest.raises(ValueError, match="malformed"):
        programs.make_validation_identity(
            context=context, program=program, inputs=inputs, indices=(0, 1, 2, 3),
            route="R", control="A_null_00", trial=0, batch_ordinal=0,
        )


def _collector(monkeypatch):
    monkeypatch.setattr(programs, "VALIDATION_ROWS", 8)
    baseline_count = torch.tensor([1, 2, 3, 4, 4, 3, 2, 1], dtype=torch.long)
    baseline = programs.ValidationBaselineSufficientStatistics(
        common_support_sha256="b" * 64,
        row_ce_sum=torch.ones(8, dtype=torch.float64),
        row_ce_count=torch.full((8,), 192, dtype=torch.long),
        row_copy_ce_sum=baseline_count.double() / 10,
        row_copy_count=baseline_count,
        literal_early_mlp_calls=programs.ZERO_NATIVE_CALLS,
        native_guard_restored=True, native_guard_inert=True,
    )
    collector = programs.ValidationStatisticsCollector(
        route="R", program_sha256="a" * 64, common_support_sha256="b" * 64,
        baseline=baseline,
    )
    return collector, baseline_count


def _add_collector_batch(collector, baseline_count, ordinal):
    start = ordinal * runtime.BATCH_SIZE
    stop = start + runtime.BATCH_SIZE
    collector.add_batch(
        batch_ordinal=ordinal, ordered_row_indices=tuple(range(start, stop)),
        row_primary_sum=torch.arange(start, stop, dtype=torch.float64) + 1,
        row_primary_count=torch.full((4,), 192, dtype=torch.long),
        row_ce_sum=torch.arange(start, stop, dtype=torch.float64) + 10,
        row_ce_count=torch.full((4,), 192, dtype=torch.long),
        row_copy_ce_sum=baseline_count[start:stop].double() / 5,
        row_copy_count=baseline_count[start:stop],
        student_original_calls=programs.ZERO_NATIVE_CALLS,
        hook_restored=True, hook_inert=True,
    )


def test_validation_collector_assembles_every_row_once_and_finalizes(monkeypatch) -> None:
    collector, baseline_count = _collector(monkeypatch)
    _add_collector_batch(collector, baseline_count, 0)
    assert collector.completed_rows == 4
    with pytest.raises(RuntimeError, match="missing"):
        collector.finalize()
    _add_collector_batch(collector, baseline_count, 1)
    statistics = collector.finalize()
    assert statistics.route == "R" and statistics.program_sha256 == "a" * 64
    assert statistics.common_support_sha256 == "b" * 64
    assert torch.equal(statistics.row_copy_count, baseline_count)
    assert statistics.row_primary_sum.tolist() == list(range(1, 9))
    with pytest.raises(RuntimeError, match="already finalized"):
        collector.finalize()


def test_validation_collector_rejects_replay_support_and_closure_drift(monkeypatch) -> None:
    collector, baseline_count = _collector(monkeypatch)
    with pytest.raises(RuntimeError, match="out of order"):
        _add_collector_batch(collector, baseline_count, 1)
    with pytest.raises(RuntimeError, match="row identity"):
        collector.add_batch(
            batch_ordinal=0, ordered_row_indices=(0, 1, 2, 4),
            row_primary_sum=torch.ones(4, dtype=torch.float64),
            row_primary_count=torch.full((4,), 192, dtype=torch.long),
            row_ce_sum=torch.ones(4, dtype=torch.float64),
            row_ce_count=torch.full((4,), 192, dtype=torch.long),
            row_copy_ce_sum=torch.ones(4, dtype=torch.float64),
            row_copy_count=baseline_count[:4],
            student_original_calls=programs.ZERO_NATIVE_CALLS,
            hook_restored=True, hook_inert=True,
        )
    with pytest.raises(RuntimeError, match="closure"):
        collector.add_batch(
            batch_ordinal=0, ordered_row_indices=(0, 1, 2, 3),
            row_primary_sum=torch.ones(4, dtype=torch.float64),
            row_primary_count=torch.full((4,), 192, dtype=torch.long),
            row_ce_sum=torch.ones(4, dtype=torch.float64),
            row_ce_count=torch.full((4,), 192, dtype=torch.long),
            row_copy_ce_sum=torch.ones(4, dtype=torch.float64),
            row_copy_count=baseline_count[:4],
            student_original_calls=((0, 1), (1, 0), (2, 0)),
            hook_restored=True, hook_inert=True,
        )
    wrong_count = baseline_count.clone(); wrong_count[0] += 1
    with pytest.raises(RuntimeError, match="support changed"):
        _add_collector_batch(collector, wrong_count, 0)
    _add_collector_batch(collector, baseline_count, 0)
    with pytest.raises(RuntimeError, match="out of order"):
        _add_collector_batch(collector, baseline_count, 0)


def test_validation_baseline_identity_and_collector_bind_the_complete_role(monkeypatch) -> None:
    monkeypatch.setattr(programs, "VALIDATION_ROWS", 8)
    monkeypatch.setattr(capabilities, "VALIDATION_BATCH_COUNT", 2)
    rows = torch.arange(8 * 513, dtype=torch.long).view(8, 513) % 13
    context = capabilities.ValidationRunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64,
        validation_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
        identity_teacher_mapping_sha256="5" * 64,
    )
    support = programs.validation_common_support_sha256(rows)
    collector = programs.ValidationBaselineCollector(common_support_sha256=support)
    for batch in range(2):
        identity = programs.make_validation_baseline_identity(
            context=context, role_rows=rows, batch_ordinal=batch,
        )
        start = batch * 4
        batch_rows = rows[start:start + 4].contiguous()
        identity.require_batch(batch_rows, tuple(range(start, start + 4)))
        ce_sum = torch.arange(start, start + 4, dtype=torch.float64) + 1
        copy_count = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        collector.add_batch(
            batch_ordinal=batch, ordered_row_indices=tuple(range(start, start + 4)),
            row_ce_sum=ce_sum, row_ce_count=torch.full((4,), 192, dtype=torch.long),
            row_copy_ce_sum=copy_count.double() / 10, row_copy_count=copy_count,
            literal_early_mlp_calls=programs.ZERO_NATIVE_CALLS,
            native_guard_restored=True, native_guard_inert=True,
        )
    baseline = collector.finalize()
    assert baseline.common_support_sha256 == support and baseline.row_ce_sum.tolist() == list(
        range(1, 9)
    )
    candidate = programs.ValidationStatisticsCollector(
        route="R", program_sha256="a" * 64, common_support_sha256=support,
        baseline=baseline,
    )
    assert candidate.completed_rows == 0
    changed = rows[:4].clone(); changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="role/context binding"):
        programs.make_validation_baseline_identity(
            context=context, role_rows=torch.cat((changed, rows[4:])), batch_ordinal=0,
        )


def test_validation_baseline_collector_rejects_partial_and_support_mixing(monkeypatch) -> None:
    monkeypatch.setattr(programs, "VALIDATION_ROWS", 8)
    collector = programs.ValidationBaselineCollector(common_support_sha256="a" * 64)
    with pytest.raises(RuntimeError, match="missing batches"):
        collector.finalize()
    with pytest.raises(RuntimeError, match="frozen support"):
        collector.require_identity(common_support_sha256="b" * 64)
    collector.add_batch(
        batch_ordinal=0, ordered_row_indices=(0, 1, 2, 3),
        row_ce_sum=torch.ones(4, dtype=torch.float64),
        row_ce_count=torch.full((4,), 192, dtype=torch.long),
        row_copy_ce_sum=torch.ones(4, dtype=torch.float64),
        row_copy_count=torch.ones(4, dtype=torch.long),
        literal_early_mlp_calls=programs.ZERO_NATIVE_CALLS,
        native_guard_restored=True, native_guard_inert=True,
    )
    with pytest.raises(RuntimeError, match="out of order"):
        collector.add_batch(
            batch_ordinal=0, ordered_row_indices=(0, 1, 2, 3),
            row_ce_sum=torch.ones(4, dtype=torch.float64),
            row_ce_count=torch.full((4,), 192, dtype=torch.long),
            row_copy_ce_sum=torch.ones(4, dtype=torch.float64),
            row_copy_count=torch.ones(4, dtype=torch.long),
            literal_early_mlp_calls=programs.ZERO_NATIVE_CALLS,
            native_guard_restored=True, native_guard_inert=True,
        )

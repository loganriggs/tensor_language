import pytest

import global_scale_calibration as core


def bank(ce_best=0.8, kl_best=0.5, tokens=1000):
    return [core.ScaleMetrics(
        scale=scale,
        target_ce=2.0 + (scale - ce_best) ** 2,
        teacher_kl=0.2 + (scale - kl_best) ** 2,
        top1_accuracy=0.1,
        teacher_top1_agreement=0.2,
        scored_tokens=tokens,
    ) for scale in core.SCALES]


def test_calibration_selects_ce_and_kl_without_conflating_them():
    assert core.select_calibration_scales(bank()) == {
        "target_ce_selected_scale": 0.8,
        "teacher_kl_selected_scale": 0.5,
    }


def test_tie_prefers_least_intervention_then_smaller_scale():
    values = bank(ce_best=1.0)
    values = [item if item.scale not in (0.8, 1.0) else core.ScaleMetrics(
        item.scale, 2.0, item.teacher_kl, item.top1_accuracy,
        item.teacher_top1_agreement, item.scored_tokens,
    ) for item in values]
    assert core.select_calibration_scales(values)["target_ce_selected_scale"] == 1.0


def test_sealed_role_gates_are_rolewise_and_price_is_foldable():
    # Put the two optima far enough apart to exercise separate selection, but close
    # enough that each satisfies the registered cross-metric non-regression gate.
    passing_bank = bank(kl_best=0.65)
    selected = core.select_calibration_scales(passing_bank)
    result = core.evaluate_sealed_roles(selected, {
        "fresh_a": passing_bank, "fresh_b": bank(kl_best=0.65, tokens=2000),
    })
    assert result["predictive_scale_pass"] is True
    assert result["teacher_faithful_scale_pass"] is True
    assert result["predictive_point_gate"] is True
    assert result["teacher_faithful_point_gate"] is True
    assert result["uncertainty_certified"] is False
    assert result["promotive_pass"] is False
    assert result["selection_reopened_on_sealed_roles"] is False
    assert result["literal_price"] == {
        "structural_fitted_degrees_of_freedom": 1,
        "finite_grid_choice_bits": 3,
        "literal_float32_metadata_bits": 32,
        "extra_deployed_float_values": 0,
        "extra_runtime_multiplies": 0,
        "zero_delta_requires_folded_program_replay": True,
    }


def test_one_role_failure_cannot_be_averaged_away():
    selected = core.select_calibration_scales(bank())
    failed = bank()
    failed = [item if item.scale != selected["target_ce_selected_scale"] else (
        core.ScaleMetrics(item.scale, 2.1, item.teacher_kl, item.top1_accuracy,
                          item.teacher_top1_agreement, item.scored_tokens)
    ) for item in failed]
    result = core.evaluate_sealed_roles(selected, {"good": bank(), "bad": failed})
    assert result["predictive_scale_pass"] is False


def test_one_sealed_role_is_insufficient_for_a_claim():
    selected = core.select_calibration_scales(bank())
    with pytest.raises(ValueError, match="invalid schema"):
        core.evaluate_sealed_roles(selected, {"only_one": bank()})


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "tokens"])
def test_invalid_calibration_bank_fails_closed(mutation):
    values = bank()
    if mutation == "missing":
        values.pop()
    elif mutation == "duplicate":
        values[-1] = values[0]
    else:
        item = values[-1]
        values[-1] = core.ScaleMetrics(
            item.scale, item.target_ce, item.teacher_kl, item.top1_accuracy,
            item.teacher_top1_agreement, item.scored_tokens + 1,
        )
    with pytest.raises(ValueError):
        core.select_calibration_scales(values)

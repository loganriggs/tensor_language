import pytest
import torch

import block3_consequence_fit as fit
import native_gate_subset as subset


def test_capped_simplex_projection_obeys_kkt_and_is_idempotent():
    value = torch.tensor([-4.0, 0.1, 0.4, 0.8, 3.0], dtype=torch.float32)
    projected = fit.project_capped_simplex(value, 2)
    assert float(projected.min()) >= 0
    assert float(projected.max()) <= 1
    assert float(projected.double().sum()) == pytest.approx(2, abs=1e-5)
    torch.testing.assert_close(
        fit.project_capped_simplex(projected, 2), projected, rtol=0, atol=2e-6,
    )
    assert fit.project_capped_simplex(value, 0).sum() == 0
    assert fit.project_capped_simplex(value, len(value)).sum() == len(value)

    free = (projected > 1e-10) & (projected < 1 - 1e-10)
    tau = (value.double()[free] - projected[free]).mean()
    assert bool((value.double()[projected == 0] - tau <= 1e-10).all())
    assert bool((value.double()[projected == 1] - tau >= 1 - 1e-10).all())
    permutation = torch.tensor([4, 1, 3, 0, 2])
    torch.testing.assert_close(
        fit.project_capped_simplex(value[permutation], 2), projected[permutation],
        rtol=0, atol=1e-10,
    )


def test_capped_simplex_projection_1024d_near_boundary_kkt():
    value = torch.linspace(-50, 50, 1024, dtype=torch.float64)
    value[510:514] = torch.tensor([0.0, 1e-13, -1e-13, 0.0])
    projected = fit.project_capped_simplex(value, 512)
    assert float(projected.sum()) == pytest.approx(512, abs=1e-10)
    assert int((projected == 0).sum()) > 400
    assert int((projected == 1).sum()) > 400
    free = (projected > 1e-12) & (projected < 1 - 1e-12)
    tau = (value[free] - projected[free]).mean()
    assert bool((value[projected == 0] - tau <= 1e-10).all())
    assert bool((value[projected == 1] - tau >= 1 - 1e-10).all())


def test_support_ties_use_global_gate_index_and_budgets_are_nested():
    scores = torch.tensor([0.8, 0.8, 0.2, 0.9])
    global_indices = torch.tensor([12, 3, 8, 20])
    supports = fit.stable_nested_supports(scores, global_indices, (2, 3))
    assert supports[2].tolist() == [20, 3]
    assert supports[3].tolist() == [20, 3, 12]


def test_document_weights_make_each_source_document_total_one():
    mapping = torch.tensor([0, 0, 0, 1, 2, 2], dtype=torch.long)
    weights = fit.source_document_row_weights(mapping)
    totals = torch.zeros(3, dtype=torch.float64)
    totals.scatter_add_(0, mapping, weights)
    torch.testing.assert_close(totals, torch.ones_like(totals))
    losses = torch.tensor([1.0, 2.0, 3.0, 7.0, 5.0, 9.0])
    observed = fit.document_balanced_batch_loss(losses, weights, document_count=3)
    assert float(observed) == pytest.approx(((1 + 2 + 3) / 3 + 7 + (5 + 9) / 2) / 3)


def test_document_derangement_never_retains_identity_with_unequal_row_counts():
    mapping = torch.tensor([0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 4], dtype=torch.long)
    donor_rows = fit.document_deranged_row_map(mapping)
    assert len(donor_rows) == len(mapping)
    assert bool((mapping[donor_rows] != mapping).all())
    assert torch.equal(donor_rows, fit.document_deranged_row_map(mapping))


def test_teacher_kl_quotients_logit_translation_and_detects_change():
    generator = torch.Generator().manual_seed(0)
    teacher = torch.randn(2, 3, 7, generator=generator)
    shifted = teacher + torch.tensor([[[17.0]], [[-9.0]]])
    assert float(fit.teacher_kl_by_row(teacher, shifted).abs().max()) < 2e-6
    changed = teacher.clone()
    changed[..., 0] += 2
    assert bool((fit.teacher_kl_by_row(teacher, changed) > 0).all())


def test_raw_logit_softcap_is_applied_once_after_scoring_slice():
    raw = torch.tensor([[[0.0, 60.0], [30.0, -30.0], [3.0, 4.0]]])
    observed = fit.softcap_scored_raw_logits(raw, start=1, stop=3)
    expected = 30 * torch.tanh(raw[:, 1:3] / 30)
    torch.testing.assert_close(observed, expected.float())


def test_four_microbatches_equal_one_full_batch_adam_update():
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    y = torch.randn(8, generator=generator, dtype=torch.float64)
    initial = torch.randn(4, generator=generator, dtype=torch.float64)
    full = torch.nn.Parameter(initial.clone())
    accumulated = torch.nn.Parameter(initial.clone())
    full_optimizer = torch.optim.Adam([full], lr=0.01)
    micro_optimizer = torch.optim.Adam([accumulated], lr=0.01)

    full_optimizer.zero_grad(set_to_none=True)
    full_loss = ((x @ full - y).square()).sum()
    full_loss.backward()
    torch.nn.utils.clip_grad_norm_([full], 1.0)
    full_optimizer.step()

    losses = [((x[start:start + 2] @ accumulated - y[start:start + 2]).square()).sum()
              for start in range(0, 8, 2)]
    fit.logical_batch_adam_step(micro_optimizer, [accumulated], losses, max_grad_norm=1.0)
    torch.testing.assert_close(accumulated, full, rtol=0, atol=1e-14)
    assert micro_optimizer.state[accumulated]["step"] == 1


def test_logical_step_rejects_wrong_optimizer_parameters_or_microbatch_count():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    other = torch.nn.Parameter(torch.tensor(2.0))
    adam = torch.optim.Adam([parameter], lr=0.01)
    four_losses = [parameter.square() / 4 for _ in range(4)]
    with pytest.raises(ValueError):
        fit.logical_batch_adam_step(adam, [other], four_losses)
    with pytest.raises(ValueError):
        fit.logical_batch_adam_step(adam, [parameter], four_losses[:3])
    with pytest.raises(ValueError):
        fit.logical_batch_adam_step(
            torch.optim.SGD([parameter], lr=0.01), [parameter], four_losses,
        )


def test_affine_calibration_folds_into_existing_decoder_and_bias_at_zero_cost():
    generator = torch.Generator().manual_seed(1)
    width, gates = 5, 4
    left = torch.randn(gates, width, generator=generator)
    right = torch.randn(gates, width, generator=generator)
    decoder = torch.randn(width, gates, generator=generator)
    bias = torch.randn(width, generator=generator)
    indices = torch.arange(gates)
    program = subset.build_program(left, right, bias, indices, decoder)
    correction = torch.randn(width, generator=generator)
    calibrated = fit.fold_affine_calibration(program, 1.75, correction)
    value = torch.randn(8, width, generator=generator)
    expected = bias + correction + 1.75 * (program.write(value) - bias)
    torch.testing.assert_close(calibrated.write(value), expected, rtol=2e-6, atol=2e-6)
    assert fit.program_price(calibrated) == fit.program_price(program)


def test_score_write_is_invariant_to_reciprocal_factor_gauge():
    generator = torch.Generator().manual_seed(2)
    width, gates = 6, 9
    left = torch.randn(gates, width, generator=generator)
    right = torch.randn(gates, width, generator=generator)
    down = torch.randn(width, gates, generator=generator)
    bias = torch.randn(width, generator=generator)
    value = torch.randn(3, width, generator=generator)
    indices = torch.tensor([0, 2, 5, 8])
    scores = torch.tensor([0.1, 0.4, 0.7, 1.0])
    from grouped_block_coefficient_screen import balance_product_gauge

    balanced_left, balanced_right, _ = balance_product_gauge(left, right)
    first = fit.consequence_score_write(
        value, balanced_left, balanced_right, down, bias, indices, scores,
    )
    scale = torch.exp(torch.randn(gates, generator=generator))
    gauge_left, gauge_right, _ = balance_product_gauge(
        scale[:, None] * left, right / scale[:, None],
    )
    second = fit.consequence_score_write(
        value, gauge_left, gauge_right, down, bias, indices, scores,
    )
    torch.testing.assert_close(first, second, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("bad,budget", (
    (torch.tensor([float("nan")]), 0),
    (torch.tensor([0.0, 1.0]), 3),
    (torch.tensor([0, 1]), 1),
))
def test_projection_rejects_malformed_values(bad, budget):
    with pytest.raises(ValueError):
        fit.project_capped_simplex(bad, budget)

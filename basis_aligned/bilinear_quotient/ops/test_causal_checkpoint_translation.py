import pytest
import torch

import causal_checkpoint_translation as target
import run_temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1 as a11_executor
import run_temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1 as m11_executor


def test_head_slice_contraction_equals_complete_linear_output_difference():
    torch.manual_seed(1)
    batch, tokens, heads, width, residual = 2, 3, 4, 5, 7
    base = torch.randn(batch, tokens, heads, width)
    writer = base.clone()
    writer[:, :, 2] += torch.randn(batch, tokens, width)
    weight = torch.randn(residual, heads * width)

    complete_delta = (
        writer.reshape(batch, tokens, -1) @ weight.T
        - base.reshape(batch, tokens, -1) @ weight.T
    )
    translated = target.attention_head_write(
        writer[:, :, 2] - base[:, :, 2], weight, head=2, num_heads=heads
    )
    assert torch.allclose(translated, complete_delta, atol=2e-6, rtol=2e-6)


def test_restricted_writer_and_reader_are_exact_weight_contractions():
    torch.manual_seed(101)
    residual, writer_in, reader_out, rank = 7, 5, 4, 3
    basis, _ = torch.linalg.qr(torch.randn(residual, rank))
    writer = torch.randn(residual, writer_in)
    reader = torch.randn(reader_out, residual)
    assert torch.allclose(
        target.restricted_writer_operator(writer, basis), basis.T @ writer
    )
    assert torch.allclose(
        target.restricted_reader_operator(reader, basis), reader @ basis
    )


def test_restricted_bilinear_core_equals_full_checkpoint_computation():
    torch.manual_seed(102)
    residual, hidden, input_rank, output_rank = 8, 11, 3, 2
    input_basis, _ = torch.linalg.qr(torch.randn(residual, input_rank))
    output_basis, _ = torch.linalg.qr(torch.randn(residual, output_rank))
    left, right = torch.randn(hidden, residual), torch.randn(hidden, residual)
    down = torch.randn(residual, hidden)
    coordinates = torch.randn(13, input_rank)
    states = coordinates @ input_basis.T
    full_output = ((states @ left.T) * (states @ right.T)) @ down.T
    core = target.restricted_bilinear_core(
        left, right, down, input_basis, output_basis
    )
    restricted_output = torch.einsum(
        "nb,abc,nc->na", coordinates, core, coordinates
    )
    assert torch.allclose(restricted_output, full_output @ output_basis,
                          atol=2e-5, rtol=2e-5)


def test_restricted_attention_qk_and_ov_cores_are_exact_and_separate():
    torch.manual_seed(103)
    residual, head, input_rank, output_rank = 9, 4, 3, 2
    input_basis, _ = torch.linalg.qr(torch.randn(residual, input_rank))
    output_basis, _ = torch.linalg.qr(torch.randn(residual, output_rank))
    query, key, value = (torch.randn(head, residual) for _ in range(3))
    output = torch.randn(residual, head)
    left, right = torch.randn(7, input_rank), torch.randn(7, input_rank)
    qk = target.restricted_qk_core(query, key, input_basis)
    ov = target.restricted_ov_core(value, output, input_basis, output_basis)
    assert torch.allclose(
        torch.einsum("nb,bc,nc->n", left, qk, right),
        ((left @ input_basis.T @ query.T)
         * (right @ input_basis.T @ key.T)).sum(dim=-1),
        atol=1e-5, rtol=1e-5,
    )
    assert torch.allclose(
        left @ ov.T,
        left @ input_basis.T @ value.T @ output.T @ output_basis,
        atol=1e-5, rtol=1e-5,
    )


def test_weight_operator_and_empirical_reachability_are_distinct_objects():
    torch.manual_seed(104)
    basis, _ = torch.linalg.qr(torch.randn(7, 3))
    states = torch.randn(2, 5, 7)
    coordinates = target.reachable_subspace_coordinates(states, basis)
    assert coordinates.shape == (2, 5, 3)
    assert torch.allclose(coordinates, states @ basis)


def test_restricted_cores_survive_internal_weight_gauges():
    torch.manual_seed(105)
    residual, hidden, head, rank = 8, 10, 4, 3
    basis, _ = torch.linalg.qr(torch.randn(residual, rank))
    left, right = torch.randn(hidden, residual), torch.randn(hidden, residual)
    down = torch.randn(residual, hidden)
    native_mlp = target.restricted_bilinear_core(
        left, right, down, basis, basis
    )
    scale = torch.randn(hidden).sign() * (torch.rand(hidden) + 0.5)
    scaled_mlp = target.restricted_bilinear_core(
        left * scale[:, None], right / scale[:, None], down, basis, basis
    )
    assert torch.allclose(scaled_mlp, native_mlp, atol=2e-5, rtol=2e-5)

    query, key, value = (torch.randn(head, residual) for _ in range(3))
    output = torch.randn(residual, head)
    head_rotation, _ = torch.linalg.qr(torch.randn(head, head))
    native_qk = target.restricted_qk_core(query, key, basis)
    native_ov = target.restricted_ov_core(value, output, basis, basis)
    changed_qk = target.restricted_qk_core(
        head_rotation.T @ query, head_rotation.T @ key, basis
    )
    changed_ov = target.restricted_ov_core(
        head_rotation.T @ value, output @ head_rotation, basis, basis
    )
    assert torch.allclose(changed_qk, native_qk, atol=2e-5, rtol=2e-5)
    assert torch.allclose(changed_ov, native_ov, atol=2e-5, rtol=2e-5)


def test_restricted_core_basis_gauge_changes_coordinates_not_physical_map():
    torch.manual_seed(106)
    residual, hidden, input_rank, output_rank = 7, 9, 3, 2
    input_basis, _ = torch.linalg.qr(torch.randn(residual, input_rank))
    output_basis, _ = torch.linalg.qr(torch.randn(residual, output_rank))
    input_rotation, _ = torch.linalg.qr(torch.randn(input_rank, input_rank))
    output_rotation, _ = torch.linalg.qr(torch.randn(output_rank, output_rank))
    left, right = torch.randn(hidden, residual), torch.randn(hidden, residual)
    down = torch.randn(residual, hidden)
    coordinates = torch.randn(12, input_rank)
    states = coordinates @ input_basis.T
    full_output = ((states @ left.T) * (states @ right.T)) @ down.T
    changed_input = input_basis @ input_rotation
    changed_output = output_basis @ output_rotation
    changed_coordinates = coordinates @ input_rotation
    changed_core = target.restricted_bilinear_core(
        left, right, down, changed_input, changed_output
    )
    evaluated = torch.einsum(
        "nb,abc,nc->na", changed_coordinates, changed_core,
        changed_coordinates
    )
    assert torch.allclose(evaluated, full_output @ changed_output,
                          atol=2e-5, rtol=2e-5)


def test_shared_context_subspace_is_exact_best_fixed_rank_projector():
    torch.manual_seed(107)
    maps = torch.randn(4, 9, 5)
    report = target.optimal_shared_context_subspace(maps, rank=3)
    reconstructed = report["common_maps"] + report["private_tails"]
    assert torch.allclose(reconstructed, maps, atol=2e-6, rtol=2e-6)
    assert torch.allclose(report["basis"].T @ report["basis"], torch.eye(3),
                          atol=2e-6, rtol=2e-6)
    optimum_error = report["private_tails"].square().sum()
    assert torch.allclose(optimum_error,
                          report["optimal_squared_error_certificate"],
                          atol=2e-5, rtol=2e-5)
    assert report["certificate_absolute_error"] < 2e-5
    for _ in range(16):
        competitor, _ = torch.linalg.qr(torch.randn(9, 3))
        competitor_tail = maps - torch.einsum(
            "ck,dk,hdp->hcp", competitor, competitor,
            maps
        )
        assert competitor_tail.square().sum() + 2e-5 >= optimum_error


def test_shared_context_projector_and_error_ignore_private_head_gauges():
    torch.manual_seed(108)
    maps = torch.randn(4, 10, 6)
    changed = maps.clone()
    for head in range(4):
        rotation, _ = torch.linalg.qr(torch.randn(6, 6))
        changed[head] = maps[head] @ rotation
    native = target.optimal_shared_context_subspace(maps, rank=4)
    rotated = target.optimal_shared_context_subspace(changed, rank=4)
    native_projector = native["basis"] @ native["basis"].T
    rotated_projector = rotated["basis"] @ rotated["basis"].T
    assert torch.allclose(native_projector, rotated_projector, atol=2e-5, rtol=2e-5)
    assert torch.allclose(native["relative_squared_error"],
                          rotated["relative_squared_error"], atol=2e-6, rtol=2e-6)


def test_shared_context_subspace_accepts_heterogeneous_reader_widths():
    torch.manual_seed(109)
    maps = (torch.randn(11, 4), torch.randn(11, 7), torch.randn(11, 13))
    report = target.optimal_shared_context_subspace(maps, rank=5)
    assert report["private_widths"] == (4, 7, 13)
    assert tuple(value.shape for value in report["private_adapters"]) == (
        (5, 4), (5, 7), (5, 13))
    for original, common, tail in zip(
            maps, report["common_maps"], report["private_tails"]):
        assert torch.allclose(common + tail, original, atol=2e-6, rtol=2e-6)
    unfolded = torch.cat(maps, dim=1)
    projector = report["basis"] @ report["basis"].T
    assert torch.allclose(torch.cat(report["common_maps"], dim=1),
                          projector @ unfolded, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("rank", (0, 8, 1.5))
def test_shared_context_subspace_rejects_bad_rank(rank):
    with pytest.raises(target.CausalCheckpointTranslationError):
        target.optimal_shared_context_subspace(torch.ones(2, 7, 3), rank=rank)


@pytest.mark.parametrize("maps", ((), (torch.ones(4, 2), torch.ones(5, 3))))
def test_shared_context_subspace_rejects_bad_component_sequence(maps):
    with pytest.raises(target.CausalCheckpointTranslationError):
        target.optimal_shared_context_subspace(maps, rank=1)


def test_head_write_is_invariant_under_paired_orthogonal_head_gauge():
    torch.manual_seed(2)
    delta = torch.randn(11, 4)
    weight = torch.randn(6, 12)
    orthogonal, _ = torch.linalg.qr(torch.randn(4, 4))
    changed_weight = weight.clone()
    changed_weight[:, 4:8] = weight[:, 4:8] @ orthogonal

    native = target.attention_head_write(delta, weight, head=1, num_heads=3)
    changed = target.attention_head_write(
        delta @ orthogonal, changed_weight, head=1, num_heads=3
    )
    assert torch.allclose(changed, native, atol=2e-6, rtol=2e-6)


def test_head_translation_matches_frozen_a11_patch_support():
    torch.manual_seed(21)
    base = torch.randn(2, 3, 12)
    writer = torch.randn(2, 3, 12)
    weight = torch.randn(8, 12)
    positions = [[0, 2], [1]]
    patched = a11_executor.replace_head_slices(
        base, writer, (1,), positions, n_heads=3
    )
    complete_delta = patched @ weight.T - base @ weight.T
    expected = torch.zeros_like(complete_delta)
    translated = target.attention_head_write(
        writer[:, :, 4:8] - base[:, :, 4:8], weight, head=1, num_heads=3
    )
    for row, selected in enumerate(positions):
        expected[row, selected] = translated[row, selected]
    assert torch.allclose(complete_delta, expected, atol=2e-6, rtol=2e-6)


def test_mlp_three_factor_writes_close_exact_checkpoint_output_delta():
    torch.manual_seed(3)
    live_left = torch.randn(2, 3, 5)
    live_right = torch.randn(2, 3, 5)
    writer_left = torch.randn(2, 3, 5)
    writer_right = torch.randn(2, 3, 5)
    down = torch.randn(7, 5)

    writes = target.mlp_factor_writes(
        live_left, live_right, writer_left, writer_right, down
    )
    exact = (
        writer_left * writer_right - live_left * live_right
    ) @ down.T
    assert torch.allclose(writes["all_three"], exact, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        writes["all_three"],
        writes["left"] + writes["right"] + writes["interaction"],
    )


def test_product_factors_match_the_frozen_m11_executor_algebra():
    torch.manual_seed(31)
    tensors = tuple(torch.randn(2, 3, 5) for _ in range(4))
    shared = target.bilinear_product_factors(*tensors)
    frozen = m11_executor.exact_product_factors(*tensors)
    assert shared.keys() == frozen.keys()
    for name in shared:
        assert torch.equal(shared[name], frozen[name])


def test_mlp_factor_writes_survive_reciprocal_scaling_and_hidden_permutation():
    torch.manual_seed(4)
    left0, right0 = torch.randn(8, 5), torch.randn(8, 5)
    left1, right1 = torch.randn(8, 5), torch.randn(8, 5)
    down = torch.randn(6, 5)
    original = target.mlp_factor_writes(left0, right0, left1, right1, down)

    scale = torch.tensor([0.5, -2.0, 3.0, -0.75, 1.25])
    scaled = target.mlp_factor_writes(
        left0 * scale, right0 / scale, left1 * scale, right1 / scale, down
    )
    for name in ("left", "right", "interaction", "all_three"):
        assert torch.allclose(scaled[name], original[name], atol=2e-6, rtol=2e-6)

    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = target.mlp_factor_writes(
        left0[:, permutation], right0[:, permutation],
        left1[:, permutation], right1[:, permutation], down[:, permutation]
    )
    for name in ("left", "right", "interaction", "all_three"):
        assert torch.allclose(permuted[name], original[name], atol=2e-6, rtol=2e-6)


def test_normalized_reader_output_equals_explicit_finite_checkpoint_read():
    torch.manual_seed(5)
    state = torch.randn(2, 3, 7)
    write = torch.randn(2, 3, 7) / 5
    reader = torch.randn(4, 7)
    expected = (
        torch.nn.functional.rms_norm(state + write, (7,))
        - torch.nn.functional.rms_norm(state, (7,))
    ) @ reader.T
    actual = target.normalized_reader_output(state, write, reader)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_normalized_reader_output_is_invariant_under_residual_orthogonal_gauge():
    torch.manual_seed(6)
    state, write = torch.randn(9, 5), torch.randn(9, 5) / 4
    reader = torch.randn(3, 5)
    orthogonal, _ = torch.linalg.qr(torch.randn(5, 5))
    native = target.normalized_reader_output(state, write, reader, eps=0.0)
    changed = target.normalized_reader_output(
        state @ orthogonal, write @ orthogonal,
        reader @ orthogonal, eps=0.0,
    )
    assert torch.allclose(changed, native, atol=3e-6, rtol=3e-6)


def test_normalized_reader_report_reuses_canonical_exact_metric_and_match():
    state = torch.tensor([[[1.0, 2.0, 4.0], [2.0, -1.0, 3.0]]])
    write = torch.tensor([[[.4, -.3, .2], [-.1, .2, .3]]])
    reader = torch.eye(3)
    report = target.normalized_reader_report(state, write, reader)
    assert report["tangent_exact_cosine"] > .9
    output = target.normalized_reader_output(state, write, reader)
    match = target.reader_response_match(output, output)
    assert match == pytest.approx({
        "cosine": 1.0, "relative_l2": 0.0,
        "norm_ratio": 1.0, "sign_agreement": 1.0,
    })


@pytest.mark.parametrize(
    "call",
    (
        lambda: target.attention_head_write(
            torch.zeros(2, 4), torch.zeros(3, 8), head=2, num_heads=2
        ),
        lambda: target.attention_head_write(
            torch.zeros(2, 3), torch.zeros(4, 7), head=0, num_heads=2
        ),
        lambda: target.mlp_factor_write(torch.zeros(2, 3), torch.zeros(4, 2)),
        lambda: target.bilinear_product_factors(
            torch.zeros(2, 3), torch.zeros(2, 3),
            torch.zeros(2, 4), torch.zeros(2, 3)
        ),
        lambda: target.mlp_factor_write(
            torch.tensor([[float("nan")]]), torch.ones(2, 1)
        ),
        lambda: target.normalized_reader_output(
            torch.ones(2, 3), torch.ones(2, 3), torch.ones(4, 2)
        ),
        lambda: target.normalized_reader_output(
            torch.ones(2, 3), torch.ones(2, 3), torch.ones(4, 3), eps=-1
        ),
        lambda: target.reader_response_match(torch.ones(2), torch.ones(3)),
        lambda: target.normalized_reader_output(
            torch.ones(2, 3, dtype=torch.long),
            torch.ones(2, 3, dtype=torch.long), torch.ones(4, 3)
        ),
    ),
)
def test_translation_fails_closed_on_bad_geometry_or_nonfinite_data(call):
    with pytest.raises(target.CausalCheckpointTranslationError):
        call()

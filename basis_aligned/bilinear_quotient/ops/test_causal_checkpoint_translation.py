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

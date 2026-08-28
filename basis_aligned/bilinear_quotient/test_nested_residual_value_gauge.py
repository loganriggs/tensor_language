import torch

from .nested_residual_value_gauge import (canonicalize_nested,
    encode_nested_qk_keyed_heads, generic_combined_gauge_dimension)
from .shared_value_gauge import apply_shared_gauge


def orthogonal(width, seed):
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(
        width, width, generator=generator, dtype=torch.double))
    return q*torch.sign(torch.diag(r))


def fixture():
    generator = torch.Generator().manual_seed(70)
    model, head, layers, heads = 7, 3, 4, 2
    anchor = torch.randn(13, model, generator=generator, dtype=torch.double) \
        @ torch.diag(torch.arange(model, 0, -1, dtype=torch.double))
    values = [[torch.randn(head, model, generator=generator, dtype=torch.double)
               for _ in range(layers)] for _ in range(heads)]
    outputs = [[torch.randn(model, head, generator=generator, dtype=torch.double)
                for _ in range(layers)] for _ in range(heads)]
    return anchor, values, outputs


def test_nested_canonical_tensors_ignore_both_continuous_gauges():
    anchor, values, outputs = fixture()
    expected = canonicalize_nested(anchor, values, outputs)
    rotation = orthogonal(anchor.shape[1], 71)
    rotated_values = []
    rotated_outputs = []
    generator = torch.Generator().manual_seed(72)
    for head_values, head_outputs in zip(values, outputs):
        local = torch.randn(3, 3, generator=generator, dtype=torch.double)+4*torch.eye(3)
        local_values, local_outputs = apply_shared_gauge(
            [value@rotation for value in head_values],
            [rotation.T@output for output in head_outputs], local)
        rotated_values.append(local_values); rotated_outputs.append(local_outputs)
    actual = canonicalize_nested(anchor@rotation, rotated_values, rotated_outputs)
    torch.testing.assert_close(actual["canonical_anchor"], expected["canonical_anchor"],
                               atol=1e-9, rtol=1e-9)
    for expected_head, actual_head in zip(expected["value_maps_by_head"],
                                          actual["value_maps_by_head"]):
        for left, right in zip(expected_head, actual_head):
            torch.testing.assert_close(left, right, atol=1e-8, rtol=1e-8)
    for expected_head, actual_head in zip(expected["output_maps_by_head"],
                                          actual["output_maps_by_head"]):
        for left, right in zip(expected_head, actual_head):
            torch.testing.assert_close(left, right, atol=1e-8, rtol=1e-8)


def test_nested_codec_bytes_ignore_gauges_and_common_head_permutation():
    anchor, values, outputs = fixture()
    keys = [b"route-z", b"route-a"]
    expected = encode_nested_qk_keyed_heads(anchor, values, outputs, keys, 14)
    rotation = orthogonal(anchor.shape[1], 73)
    transformed_values = []; transformed_outputs = []
    for index, (head_values, head_outputs) in enumerate(zip(values, outputs)):
        local = torch.eye(3, dtype=torch.double)*(index+2)
        gv, go = apply_shared_gauge([value@rotation for value in head_values],
                                    [rotation.T@output for output in head_outputs], local)
        transformed_values.append(gv); transformed_outputs.append(go)
    actual = encode_nested_qk_keyed_heads(
        anchor@rotation, transformed_values[::-1], transformed_outputs[::-1],
        keys[::-1], 14)
    assert actual == expected


def test_combined_generic_dimension_is_not_a_bit_discount():
    result = generic_combined_gauge_dimension(1152, 128, 9)
    assert result == {"global_residual_orthogonal": 662976,
                      "shared_value_general_linear": 147456,
                      "generic_combined": 810432}

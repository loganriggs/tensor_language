import torch

from .nested_attention_gauge_codec import (encode_nested_attention_heads,
                                            encode_nested_qk_routes)
from .shared_value_gauge import apply_shared_gauge


def orthogonal(width, seed):
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(
        width, width, generator=generator, dtype=torch.double))
    return q*torch.sign(torch.diag(r))


def fixture():
    generator = torch.Generator().manual_seed(80)
    model, head, layers, heads = 7, 4, 3, 2
    anchor = torch.randn(13, model, generator=generator, dtype=torch.double) \
        @ torch.diag(torch.arange(model, 0, -1, dtype=torch.double))
    qk = [{name: torch.randn(head, model, generator=generator, dtype=torch.double)
           for name in ("q", "k", "q2", "k2")} for _ in range(heads)]
    values = [[torch.randn(head, model, generator=generator, dtype=torch.double)
               for _ in range(layers)] for _ in range(heads)]
    outputs = [[torch.randn(model, head, generator=generator, dtype=torch.double)
                for _ in range(layers)] for _ in range(heads)]
    return anchor, qk, values, outputs


def test_production_qk_route_bytes_ignore_global_residual_rotation():
    anchor, qk, _, _ = fixture()
    expected = encode_nested_qk_routes(anchor, qk, 1e-6, qk_tolerance=1e-8)
    rotation = orthogonal(anchor.shape[1], 81)
    rotated = [{name: value@rotation for name, value in maps.items()} for maps in qk]
    actual = encode_nested_qk_routes(
        anchor@rotation, rotated, 1e-6, qk_tolerance=1e-8)
    torch.testing.assert_close(expected[0]["canonical_anchor"],
                               actual[0]["canonical_anchor"], atol=1e-9, rtol=1e-9)
    assert expected[1] == actual[1]
    expected_lowrank = encode_nested_qk_routes(
        anchor, qk, 1e-7, rank=2, qk_tolerance=1e-8)
    actual_lowrank = encode_nested_qk_routes(
        anchor@rotation, rotated, 1e-7, rank=2, qk_tolerance=1e-8)
    assert expected_lowrank[1] == actual_lowrank[1]


def test_joint_route_bound_bytes_ignore_all_gauges_and_head_permutation():
    anchor, qk, values, outputs = fixture()
    expected = encode_nested_attention_heads(
        anchor, qk, values, outputs, 1e-6, 14, qk_tolerance=1e-8)
    rotation = orthogonal(anchor.shape[1], 82)
    transformed_qk = []
    transformed_values = []
    transformed_outputs = []
    generator = torch.Generator().manual_seed(83)
    for maps, head_values, head_outputs in zip(qk, values, outputs):
        # Existing Q/K codec quotients even sign parity and branch exchange.
        changed = {"q": -(maps["q2"]@rotation),
                   "k": maps["k2"]@rotation,
                   "q2": -(maps["q"]@rotation),
                   "k2": maps["k"]@rotation}
        transformed_qk.append(changed)
        local = torch.randn(4, 4, generator=generator, dtype=torch.double)+5*torch.eye(4)
        gv, go = apply_shared_gauge(
            [value@rotation for value in head_values],
            [rotation.T@output for output in head_outputs], local)
        transformed_values.append(gv); transformed_outputs.append(go)
    actual = encode_nested_attention_heads(
        anchor@rotation, transformed_qk[::-1], transformed_values[::-1],
        transformed_outputs[::-1], 1e-6, 14, qk_tolerance=1e-8)
    assert sorted(expected["routes"]) == sorted(actual["routes"])
    assert expected["route_bound_vo_bundle"] == actual["route_bound_vo_bundle"]

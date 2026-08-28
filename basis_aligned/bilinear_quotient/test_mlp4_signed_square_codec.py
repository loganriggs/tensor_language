import torch

from . import mlp4_signed_square_codec as codec


SOURCES = ("mlp4.rmsnorm_input",)
WIDTHS = (7,)


def factors(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(7, 4, generator=g, dtype=torch.double),
            torch.randn(7, 4, generator=g, dtype=torch.double),
            torch.randn(4, 5, generator=g, dtype=torch.double),
            torch.randn(5, generator=g, dtype=torch.double))


def test_polarization_is_exact_before_quantization():
    A, B, C, bias = factors()
    U, V, canonical_C = codec.signed_square_factors(A, B, C)
    x = torch.randn(11, 7, generator=torch.Generator().manual_seed(9),
                    dtype=torch.double)
    # Compare against the same canonicalized factors because component gauges/order
    # are intentionally normalized by the codec.
    Ac, Bc, Cc = codec.product_codec.canonical_factors(A, B, C)
    expected = bias+((x@Ac)*(x@Bc))@Cc
    actual = bias+((x@U).square()-(x@V).square())@canonical_C
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_stream_is_gauge_invariant_and_decoded_execution_is_finite():
    A, B, C, bias = factors(seed=2)
    encoded, price = codec.encode(A, B, C, bias, 2**-18, SOURCES, WIDTHS)
    alpha = torch.tensor([2., -.5, 3., -4.], dtype=torch.double)
    beta = torch.tensor([-.25, 5., -2., .125], dtype=torch.double)
    permutation = torch.tensor([2, 0, 3, 1])
    A2, B2, C2 = A*alpha, B*beta, C/(alpha*beta)[:, None]
    A2[:, [0, 2]], B2[:, [0, 2]] = B2[:, [0, 2]].clone(), A2[:, [0, 2]].clone()
    encoded2, price2 = codec.encode(A2[:, permutation], B2[:, permutation],
                                    C2[permutation], bias, 2**-18,
                                    SOURCES, WIDTHS)
    assert encoded == encoded2
    assert price["canonical_bytes_hash"] == price2["canonical_bytes_hash"]
    decoded = codec.decode(encoded)
    output = codec.execute_decoded(decoded, torch.randn(3, 7, dtype=torch.double))
    assert output.shape == (3, 5) and torch.isfinite(output).all()

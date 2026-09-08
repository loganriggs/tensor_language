import pytest
import torch

import residual_identity_route as target


def run_suffix(entry, embedding, lambdas, attention_writes, mlp_writes):
    state = entry.clone()
    for coefficient, attention, mlp in zip(lambdas, attention_writes, mlp_writes):
        state = coefficient * state + (1.0 - coefficient) * embedding
        state = state + attention + mlp
    return state


def test_direct_add_and_remove_equal_controlled_factorial_states():
    native_entry = torch.tensor([[1.0, -2.0, 3.0]])
    writer_entry = torch.tensor([[2.5, -1.0, 2.0]])
    embedding = torch.tensor([[0.25, 0.5, -0.75]])
    lambdas = (0.9, 1.1, 0.8)
    native_attention = tuple(torch.full_like(native_entry, value) for value in (0.2, -0.1, 0.4))
    native_mlp = tuple(torch.full_like(native_entry, value) for value in (-0.3, 0.5, 0.1))
    writer_attention = tuple(torch.full_like(native_entry, value) for value in (0.7, -0.4, 0.2))
    writer_mlp = tuple(torch.full_like(native_entry, value) for value in (0.6, 0.3, -0.2))

    native_final = run_suffix(
        native_entry, embedding, lambdas, native_attention, native_mlp)
    writer_final = run_suffix(
        writer_entry, embedding, lambdas, writer_attention, writer_mlp)
    r1m0 = run_suffix(
        writer_entry, embedding, lambdas, native_attention, native_mlp)
    r0m1 = run_suffix(
        native_entry, embedding, lambdas, writer_attention, writer_mlp)
    gain = target.skip_gain(lambdas)

    assert torch.allclose(
        target.direct_add(native_final, native_entry, writer_entry, gain), r1m0)
    assert torch.allclose(
        target.direct_remove(writer_final, native_entry, writer_entry, gain), r0m1)


def test_route_rejects_empty_nonfinite_and_mismatched_inputs():
    with pytest.raises(target.ResidualIdentityRouteError):
        target.skip_gain(())
    with pytest.raises(target.ResidualIdentityRouteError):
        target.skip_gain((1.0, float("nan")))
    with pytest.raises(target.ResidualIdentityRouteError):
        target.direct_add(torch.zeros(2), torch.zeros(2), torch.zeros(3), 1.0)

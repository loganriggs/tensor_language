import pytest
import torch

import causal_port_balancing as balance


def snapshots():
    # Coordinate 0 has enormous reachable variance but is unobservable. Coordinates
    # 1 and 2 carry the actual input-output map.
    reachable = torch.tensor([
        [100., 1., 0.], [-100., 0., 1.], [80., 1., 1.], [-80., 1., -1.],
    ], dtype=torch.float64)
    observable = torch.tensor([
        [0., 2., 0.], [0., 0., 3.], [0., 1., 1.],
    ], dtype=torch.float64)
    return reachable, observable


def test_balancing_discards_reachable_but_unobservable_nuisance():
    reachable, observable = snapshots()
    port = balance.fit_balanced_port(reachable, observable, 2)
    assert port.biorthogonality_max_abs_error < 1e-12
    assert port.response_tail_squared_frobenius == pytest.approx(0.0, abs=1e-20)
    # The observable response is exactly preserved despite the huge nuisance variance.
    raw = observable @ reachable.T / (len(observable) * len(reachable)) ** 0.5
    projected = balance.projected_response(port, reachable, observable)
    assert torch.allclose(projected, raw, atol=1e-12, rtol=1e-12)


def test_hankel_values_and_physical_response_are_gauge_invariant():
    reachable, observable = snapshots()
    transform = torch.tensor([
        [2., 1., 0.], [0., 3., 1.], [1., 0., 2.],
    ], dtype=torch.float64)
    inverse = torch.linalg.inv(transform)
    first = balance.fit_balanced_port(reachable, observable, 2)
    changed_reachable = reachable @ transform.T
    changed_observable = observable @ inverse
    second = balance.fit_balanced_port(changed_reachable, changed_observable, 2)
    assert torch.allclose(
        first.hankel_singular_values, second.hankel_singular_values,
        atol=1e-11, rtol=1e-11,
    )
    assert torch.allclose(
        balance.projected_response(first, reachable, observable),
        balance.projected_response(second, changed_reachable, changed_observable),
        atol=1e-11, rtol=1e-11,
    )
    assert torch.allclose(
        second.projection, transform @ first.projection @ inverse,
        atol=1e-10, rtol=1e-10,
    )


def test_tail_is_eckart_young_response_error():
    generator = torch.Generator().manual_seed(9)
    reachable = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    observable = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    port = balance.fit_balanced_port(reachable, observable, 3)
    raw = observable @ reachable.T / (len(observable) * len(reachable)) ** 0.5
    error = float((raw - balance.projected_response(port, reachable, observable)).square().sum())
    assert error == pytest.approx(port.response_tail_squared_frobenius, rel=1e-10, abs=1e-12)


def test_invalid_or_unsupported_rank_fails_closed():
    reachable, observable = snapshots()
    with pytest.raises(ValueError):
        balance.fit_balanced_port(reachable.float(), observable, 2)
    with pytest.raises(ValueError):
        balance.fit_balanced_port(reachable, observable, 0)
    with pytest.raises(RuntimeError, match="exceeds numerical response support"):
        balance.fit_balanced_port(reachable, observable, 3)


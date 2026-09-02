import torch

import equality_mlp_response_form_rung469 as subject


def test_quadratic_reader_state_identity():
    generator = torch.Generator().manual_seed(469)
    d, hidden = 7, 11
    left = torch.randn(hidden, d, generator=generator, dtype=torch.float64)
    right = torch.randn(hidden, d, generator=generator, dtype=torch.float64)
    down = torch.randn(d, hidden, generator=generator, dtype=torch.float64)
    reader = torch.randn(d, generator=generator, dtype=torch.float64)
    source = torch.randn(5, d, generator=generator, dtype=torch.float64)
    absent = torch.randn(5, d, generator=generator, dtype=torch.float64)
    q = subject.quadratic_reader(left, right, down, reader)
    s = subject.state_form(source, absent)
    direct = sum(
        reader @ down @ ((left @ x) * (right @ x) - (left @ y) * (right @ y))
        for x, y in zip(source, absent)
    )
    assert torch.allclose((q * s).sum(), direct, atol=1e-9, rtol=1e-9)


def test_fit_scale_and_metrics():
    local = torch.tensor([1.0, -2.0, 3.0, -4.0])
    exact = 2.5 * local
    scale = subject._fit_scale(local, exact)
    metrics = subject._metrics(exact, scale * local)
    assert abs(scale - 2.5) < 1e-12
    assert abs(metrics["cosine"] - 1) < 1e-12
    assert metrics["normalized_l2_error"] < 1e-12


def test_reader_and_state_forms_ignore_factor_rescaling():
    generator = torch.Generator().manual_seed(470)
    d, hidden = 5, 9
    left = torch.randn(hidden, d, generator=generator, dtype=torch.float64)
    right = torch.randn(hidden, d, generator=generator, dtype=torch.float64)
    down = torch.randn(d, hidden, generator=generator, dtype=torch.float64)
    reader = torch.randn(d, generator=generator, dtype=torch.float64)
    scales = torch.linspace(.5, 1.5, hidden, dtype=torch.float64)
    original = subject.quadratic_reader(left, right, down, reader)
    rescaled = subject.quadratic_reader(
        scales[:, None] * left, right / scales[:, None], down, reader,
    )
    assert torch.allclose(original, rescaled, atol=1e-10, rtol=1e-10)


def test_metrics_report_projection_not_inverse_projection():
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    prediction = torch.tensor([.25, 0.0, 0.0, 0.0])
    metrics = subject._metrics(target, prediction)
    assert abs(metrics["projection_on_target"] - .25) < 1e-12
    assert abs(metrics["normalized_l2_error"] - .75) < 1e-12

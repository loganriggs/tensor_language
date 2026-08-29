import torch

import mlp2_error_rayleigh_collector_core as core
import mlp2_error_rayleigh_metrics as reference


def test_controls_are_matched_norm_deranged_and_deterministic():
    actual = torch.arange(5 * 3 * 4, dtype=torch.float32).reshape(5, 3, 4) + 1
    first = core.control_error_bank(actual, 17)
    second = core.control_error_bank(actual, 17)
    assert tuple(first) == core.CONTROL_NAMES
    assert torch.equal(first["ACTUAL"], actual)
    assert not any(torch.equal(first["DERANGED"][i], actual[i]) for i in range(5))
    reference_norm = actual.flatten(1).norm(dim=1)
    for name, value in first.items():
        assert torch.allclose(value.flatten(1).norm(dim=1), reference_norm, rtol=1e-5)
        assert torch.equal(value, second[name])


def test_endpoint_interpolation_and_control_write():
    native = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    candidate = torch.randn_like(native)
    assert core.actual_write(native, candidate, 0) is native
    assert core.actual_write(native, candidate, 1) is candidate
    halfway = core.actual_write(native, candidate, 0.5)
    assert torch.allclose(halfway.float(), (native.float() + candidate.float()) / 2,
                          atol=0.01, rtol=0)
    error = candidate.float() - native.float()
    assert torch.allclose(core.control_write(native, error, 0.5).float(), halfway.float())


def test_response_reduction_matches_reference_metrics():
    torch.manual_seed(2)
    documents, positions, vocab, width = 4, 5, 7, 3
    baseline = torch.randn(documents, positions, vocab)
    derivative = torch.randn_like(baseline)
    a = 0.125
    plus, minus = baseline + a * derivative, baseline - a * derivative
    base5 = torch.randn(documents, 6, width)
    d5 = torch.randn_like(base5)
    base6 = torch.randn(documents, 6, width)
    d6 = torch.randn_like(base6)
    targets = torch.randint(0, vocab, (documents, positions))
    out = core.response_statistics(
        baseline, plus, minus, base5, base5 + a*d5, base5 - a*d5,
        base6, base6 + a*d6, base6 - a*d6, targets, a,
    )
    assert torch.allclose(out["qlogit"], reference.categorical_fisher_quadratic(
        baseline, derivative), atol=2e-6, rtol=2e-6)
    assert torch.allclose(out["q5"], reference.normalized_response_energy(d5, base5),
                          atol=2e-6, rtol=2e-6)
    assert torch.allclose(out["q6"], reference.normalized_response_energy(d6, base6),
                          atol=2e-6, rtol=2e-6)
    assert torch.allclose(out["kl_plus"], reference.teacher_kl(baseline, plus),
                          atol=2e-6, rtol=2e-6)


def test_pack_and_exact_replay_statistics():
    n = 3
    local = torch.arange(1., n + 1)
    values = {name: torch.ones(n) for name in (
        "ce_jvp", "qlogit", "q5", "q6", "kl_minus", "kl_plus",
        "dce_minus", "dce_plus",
    )}
    packed = core.pack_features(local, {a: values for a in core.AMPLITUDES})
    assert packed.shape == (n, len(core.FEATURE_NAMES))
    logits = torch.randn(n, 2, 5)
    attn5 = torch.randn(n, 4, 3)
    attn6 = torch.randn(n, 4, 3)
    targets = torch.randint(0, 5, (n, 2))
    replay = core.replay_statistics(
        logits, logits, logits, attn5, attn5, attn5, attn6, attn6, attn6, targets,
    )
    assert replay.shape == (n, len(core.FINITE_NAMES))
    assert torch.equal(replay[:, :5], torch.zeros(n, 5, dtype=torch.float64))
    assert torch.equal(replay[:, 5:], torch.ones(n, 3, dtype=torch.float64))

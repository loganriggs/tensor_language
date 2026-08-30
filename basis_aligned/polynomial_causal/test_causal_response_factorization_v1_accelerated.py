import torch

from causal_response_factorization_v1 import (
    make_program_from_factors,
    predict_from_codes,
)
from causal_response_factorization_v1_accelerated import (
    fit_shared_private_program_accelerated,
    seeded_initial_mse,
)


def _planted():
    generator = torch.Generator().manual_seed(88)
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    program = make_program_from_factors(
        tuple(torch.randn(shape, generator=generator, dtype=torch.float64)
              for shape in ((2, 1), (4, 1), (3, 1))),
        tuple(
            tuple(torch.randn(shape, generator=generator, dtype=torch.float64)
                  for shape in ((2, 1), (2, 1), (3, 1)))
            for _ in range(2)
        ),
        groups,
    )
    codes = torch.randn((12, program.code_dimension), generator=generator,
                        dtype=torch.float64)
    response = predict_from_codes(program.basis(), codes).reshape(2, 4, 3, 12)
    return groups, response, torch.ones_like(response, dtype=torch.bool)


def test_accelerated_cpu_float32_recovers_planted_topology_in_float64_replay():
    groups, response, valid = _planted()
    fitted = fit_shared_private_program_accelerated(
        response, valid, groups, global_rank=1, private_rank=1,
        seed=2026083001, steps=2_000, learning_rate=0.04,
        optimizer_device="cpu",
    )
    replay = predict_from_codes(
        fitted.program.basis(), fitted.document_codes
    ).reshape_as(response)
    assert fitted.improvement_fraction > 0.9999
    assert fitted.final_mse < 1e-8
    assert torch.allclose(replay, response, atol=5e-4, rtol=5e-4)


def test_accelerated_same_seed_replays_exactly_on_cpu():
    groups, response, valid = _planted()
    kwargs = dict(
        global_rank=1, private_rank=1, seed=2026083002,
        steps=50, learning_rate=0.03, optimizer_device="cpu",
    )
    first = fit_shared_private_program_accelerated(response, valid, groups, **kwargs)
    second = fit_shared_private_program_accelerated(response, valid, groups, **kwargs)
    assert first.final_mse == second.final_mse
    assert torch.equal(first.program.basis(), second.program.basis())
    assert torch.equal(first.document_codes, second.document_codes)
    assert first.initial_mse == seeded_initial_mse(
        response, valid, groups, global_rank=1, private_rank=1, seed=2026083002,
    )


def test_accelerated_rejects_unavailable_or_noncompute_device():
    groups, response, valid = _planted()
    try:
        fit_shared_private_program_accelerated(
            response, valid, groups, global_rank=1, private_rank=1,
            seed=1, steps=1, optimizer_device="meta",
        )
    except RuntimeError as error:
        assert "unavailable" in str(error)
    else:
        raise AssertionError("noncompute optimizer device must fail closed")

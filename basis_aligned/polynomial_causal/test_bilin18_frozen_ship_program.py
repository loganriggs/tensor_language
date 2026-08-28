from __future__ import annotations

from types import SimpleNamespace

import torch

import bilin18_frozen_ship_program as frozen
import bilin18_observed_model_facade as facade


def synthetic_program() -> frozen.FrozenShipProgram:
    d_model, vocab = 4, 7
    identity = torch.eye(d_model)
    state = {
        "TWALL": {},
        "SHIP": {
            "t0": torch.arange(vocab * d_model).view(vocab, d_model).float() / 10,
            "r0": (identity, torch.zeros(d_model), torch.ones(d_model)),
            "t1": torch.arange(vocab * d_model).view(vocab, d_model).float() / 20,
            "r1": (torch.ones(2 * d_model, d_model) / 10,
                   torch.zeros(2 * d_model), torch.zeros(d_model)),
            "r2": (torch.ones(2 * d_model, d_model) / 20,
                   torch.zeros(2 * d_model), torch.zeros(d_model)),
        },
        "CORR": {
            "on": True,
            "b": torch.ones(d_model) / 4,
            "V": torch.ones(2 * d_model, 1) / 5,
            "U": torch.ones(d_model, 1) / 6,
        },
        "all_attention": [],
    }
    return frozen.FrozenShipProgram(state, production=False)


def event(site, attention, tokens, prior=()):
    state = torch.zeros((*tokens.shape, 4))
    return facade.EarlyMLPEvent(
        site=site,
        block=SimpleNamespace(),
        state=state,
        attention_write=attention,
        tokens=tokens,
        prior_writes=tuple(prior),
    )


def test_early_ship_is_sequential_and_mlp1_reads_effective_p0() -> None:
    program = synthetic_program()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    attention0 = torch.arange(8).view(1, 2, 4).float() / 7
    n0 = program.mlp(event(0, attention0, tokens))
    expected0 = program.ship["t0"][tokens] + 1 + attention0
    torch.testing.assert_close(n0, expected0)

    attention1 = torch.ones_like(n0) / 3
    n1 = program.mlp(event(1, attention1, tokens, (n0,)))
    shifted = program.mlp(event(1, attention1, tokens, (n0 + 2,)))
    assert not torch.equal(n1, shifted)
    expected_shift = torch.full_like(n1, 0.8)
    torch.testing.assert_close(shifted - n1, expected_shift)


def test_mlp2_applies_frozen_correction_and_rejects_wrong_order() -> None:
    program = synthetic_program()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    prior0 = torch.zeros((1, 2, 4))
    prior1 = torch.ones((1, 2, 4))
    attention2 = torch.ones((1, 2, 4)) * 2
    value = program.mlp(event(2, attention2, tokens, (prior0, prior1)))
    features = torch.cat([attention2, prior1], dim=-1).reshape(-1, 8)
    base = features @ program.ship["r2"][0]
    correction = program.corr["b"] + (features @ program.corr["V"]) @ program.corr["U"].T
    torch.testing.assert_close(value.reshape(-1, 4), base + correction)

    bad = event(2, attention2, tokens, (prior0,))
    try:
        program.mlp(bad)
    except RuntimeError as error:
        assert "nonsequential" in str(error)
    else:
        raise AssertionError("nonsequential frozen MLP event did not fail")

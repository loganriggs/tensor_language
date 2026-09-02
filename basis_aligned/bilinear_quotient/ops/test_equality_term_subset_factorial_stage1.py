from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("equality_term_subset_factorial_stage1.py")
SPEC = importlib.util.spec_from_file_location("equality_stage1", PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_arm_identity_and_term_bits() -> None:
    assert len(module.ARMS) == 33
    assert len(set(module.ARMS)) == 33
    assert module.arm_parts("remove:0001") == ("remove", 1)
    assert module.arm_parts("extract:1100") == ("extract", 12)
    assert module._selected(1, 5, 5)
    assert not module._selected(1, 7, 3)
    assert module._selected(12, 8, 3)
    assert module._selected(12, 8, 4)


def test_mobius_recovers_planted_pair_interaction() -> None:
    values = torch.zeros(16, dtype=torch.float64)
    singleton = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    for mask in range(16):
        values[mask] = sum(singleton[bit] for bit in range(4) if mask & (1 << bit))
        if mask & 0b1100 == 0b1100:
            values[mask] -= .07
    dividends = module._mobius(values)
    assert torch.allclose(dividends[[1, 2, 4, 8]], singleton)
    assert torch.isclose(dividends[0b1100], torch.tensor(-.07, dtype=torch.float64))
    assert torch.allclose(
        dividends[[3, 5, 6, 7, 9, 10, 11, 13, 14, 15]],
        torch.zeros(10, dtype=torch.float64), atol=1e-12,
    )
    assert torch.isclose(dividends.sum(), values[-1])


def test_shapley_allocates_every_dividend_once() -> None:
    dividends = torch.arange(16, dtype=torch.float64) / 100
    allocation = module.shapley(dividends)
    assert torch.isclose(allocation.sum(), dividends[1:].sum())


def test_classification_uses_interval_and_floor() -> None:
    assert module.classify(-.01, -.012, -.007) == "redundant"
    assert module.classify(.01, .007, .014) == "complementary"
    assert module.classify(.01, .004, .016) == "additive_or_unresolved"
    assert module.classify(.003, .002, .004) == "additive_or_unresolved"


def test_frozen_masks_and_code_role_exclusion() -> None:
    payload, masks, metadata = module.validate_inputs()
    assert payload["role"] == "final_natural"
    assert list(masks) == list(module.CELLS)
    assert metadata["support"]["all_positive"]["tokens"] == 3084
    assert metadata["support"]["near_positive"]["tokens"] == 719
    assert torch.equal(
        masks["near_positive"] | masks["far_positive"], masks["all_positive"],
    )
    assert "ood_code" not in str(module.ROWS)

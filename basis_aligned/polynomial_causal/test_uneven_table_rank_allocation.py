import torch

import uneven_table_rank_allocation as allocation


def _spectrum(scale: float = 1.0) -> torch.Tensor:
    value = torch.ones(allocation.WIDTH, dtype=torch.float64)
    value[:64] *= scale
    return value


def test_allocator_respects_budget_and_favors_high_marginal_value():
    spectra = {"a": _spectrum(100), "b": _spectrum(1)}
    floor = 2 * allocation.site_cost(64)
    budget = floor + allocation.site_cost(128) - allocation.site_cost(64)
    result = allocation.allocate(spectra, (64, 128), budget, normalized=False)
    # The first 64 directions are sunk at the floor; the next 64 are equal here, so
    # deterministic lexical tie-breaking chooses a before b.
    assert result.ranks == {"a": 128, "b": 64}
    assert result.cost <= budget


def test_normalized_allocation_is_invariant_to_whole_site_scale():
    base = torch.linspace(4, 1, allocation.WIDTH, dtype=torch.float64)
    spectra = {"a": base, "b": torch.flip(base, (0,))}
    scaled = {"a": 1e8 * base, "b": torch.flip(base, (0,))}
    budget = allocation.site_cost(64) + allocation.site_cost(128)
    one = allocation.allocate(spectra, (64, 128), budget, normalized=True)
    two = allocation.allocate(scaled, (64, 128), budget, normalized=True)
    assert one.ranks == two.ranks


def test_raw_allocation_changes_under_site_scale():
    descending = torch.linspace(4, 1, allocation.WIDTH, dtype=torch.float64)
    flat = torch.ones(allocation.WIDTH, dtype=torch.float64)
    budget = allocation.site_cost(64) + allocation.site_cost(128)
    result = allocation.allocate(
        {"a": 1e6 * descending, "b": flat}, (64, 128), budget, normalized=False,
    )
    assert result.ranks["a"] == 128


def test_type_shift_preserves_rank_multisets_and_cost():
    ranks = {
        **{f"mlp{i}": 64 * (1 + i % 3) for i in range(18)},
        **{f"attn{i}": 64 * (1 + i % 4) for i in range(18)},
    }
    shifted = allocation.type_shifted_null(ranks)
    for kind in ("mlp", "attn"):
        assert sorted(shifted[f"{kind}{i}"] for i in range(18)) == sorted(
            ranks[f"{kind}{i}"] for i in range(18)
        )
    assert allocation.allocation_cost(shifted) == allocation.allocation_cost(ranks)


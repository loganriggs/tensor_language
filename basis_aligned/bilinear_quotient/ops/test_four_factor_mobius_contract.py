import numpy as np
import pytest

import four_factor_mobius_contract as contract


def test_mobius_reconstructs_every_corner_and_shapley_is_efficient():
    corners = {mask: np.asarray([mask, mask * mask, (-1) ** mask], dtype=np.float64)
               for mask in contract.MASKS}
    rebuilt = contract.reconstruct(contract.mobius(corners))
    assert all(np.array_equal(rebuilt[mask], corners[mask]) for mask in contract.MASKS)
    values = contract.shapley(corners)
    assert np.allclose(sum(values.values()), corners[15] - corners[0])


def test_shortest_prefix_uses_fixed_tie_order():
    vectors = {factor: np.asarray([1.0]) for factor in contract.FACTORS}
    selected, norms = contract.shortest_norm_mass_prefix(vectors, 0.5)
    assert selected == ("q", "k")
    assert norms == {factor: 1.0 for factor in contract.FACTORS}


def test_rejects_missing_corner():
    with pytest.raises(ValueError, match="16 masks"):
        contract.mobius({0: np.zeros(1)})

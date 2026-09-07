#!/usr/bin/env python3
"""Focused exact tests for Boolean response-lattice decomposition."""

import numpy as np

import boolean_lattice_mobius as lattice


def main():
    rng = np.random.default_rng(20260907)
    values = rng.normal(size=(32, 3, 2))
    coefficients = lattice.mobius_transform(values)
    assert np.allclose(lattice.zeta_transform(coefficients), values, atol=1e-12)

    # f(x)=2 + 3*x0 - 4*x1 + 7*x0*x2 has exactly the declared terms.
    scalar = []
    for mask in range(8):
        scalar.append(2 + 3 * bool(mask & 1) - 4 * bool(mask & 2)
                      + 7 * bool(mask & 1) * bool(mask & 4))
    exact = lattice.mobius_transform(scalar)
    expected = np.zeros(8); expected[[0, 1, 2, 5]] = [2, 3, -4, 7]
    assert np.allclose(exact, expected)
    energy = lattice.degree_energy(exact)
    assert energy == {0: 4.0, 1: 25.0, 2: 49.0, 3: 0.0}
    assert lattice.top_terms(exact, ("a", "b", "c"), 1)[0]["sites"] == ["a", "c"]
    print("boolean lattice Mobius tests: PASS")


if __name__ == "__main__":
    main()

"""Exact Moebius/Harsanyi interaction analysis for subset factorials (ops lane).

For a set function y(S) measured on ALL subsets S of a ground set N (e.g.
the 2^4 = 16 retain-subsets of the four equality-edge terms), the
Harsanyi dividends
    d(S) = sum_{T subseteq S} (-1)^{|S|-|T|} y(T)
are the UNIQUE additive decomposition y(S) = sum_{T subseteq S} d(T)
(Moebius inversion on the subset lattice; Harsanyi 1963, Rota 1964).
Shapley values are dividend shares: phi_i = sum_{S ni i} d(S)/|S|.

Sign conventions (state which y you feed in!):
- y = task effect RECOVERED by the retained subset (bigger = more task):
  d(pair) < 0  => redundancy (substitutes; one suffices),
  d(pair) > 0  => complementarity (they need each other),
  d(pair) ~ 0  => additivity (independent contributions).
- y = CE DAMAGE from removing the complement flips all signs.

Noise propagation (for preregistering bars): if each y(S) carries iid
noise sigma, then sigma(d(S)) = 2^{|S|/2} * sigma.  With the program's
CUDA wobble sigma ~= .003: order-2 dividends resolve above ~.006,
order-3 above ~.0085, order-4 above ~.012 -- bars on high-order
interactions should clear those floors.

Verified below on synthetic redundant / complementary / additive cases.
"""
from itertools import chain, combinations


def subsets(elements):
    e = list(elements)
    return chain.from_iterable(combinations(e, r) for r in range(len(e) + 1))


def dividends(values):
    """values: {frozenset/tuple -> float} on ALL subsets. Returns {frozenset -> d}."""
    v = {frozenset(k): float(val) for k, val in values.items()}
    ground = max(v, key=len)
    out = {}
    for s in subsets(ground):
        s = frozenset(s)
        out[s] = sum((-1) ** (len(s) - len(t)) * v[frozenset(t)] for t in subsets(s))
    return out


def shapley(divs):
    phi = {}
    for s, d in divs.items():
        for i in s:
            phi[i] = phi.get(i, 0.0) + d / len(s)
    return phi


def reconstruct(divs, s):
    s = frozenset(s)
    return sum(d for t, d in divs.items() if t <= s)


def dividend_noise_floor(order, sigma):
    return (2 ** (order / 2)) * sigma


if __name__ == "__main__":
    # additive: y(S) = |S| -> all interactions zero
    add = {s: float(len(s)) for s in map(frozenset, subsets("ab"))}
    d = dividends(add)
    assert abs(d[frozenset("ab")]) < 1e-12, d
    # redundant (substitutes): y=1 iff S nonempty -> pairwise dividend -1
    red = {frozenset(): 0.0, frozenset("a"): 1.0, frozenset("b"): 1.0, frozenset("ab"): 1.0}
    assert abs(dividends(red)[frozenset("ab")] + 1.0) < 1e-12
    # complementary (AND): y=1 only for full set -> pairwise dividend +1
    comp = {frozenset(): 0.0, frozenset("a"): 0.0, frozenset("b"): 0.0, frozenset("ab"): 1.0}
    assert abs(dividends(comp)[frozenset("ab")] - 1.0) < 1e-12
    # exact reconstruction on a random 4-set function
    import random
    random.seed(0)
    vals = {frozenset(s): random.random() for s in subsets("wxyz")}
    dv = dividends(vals)
    assert all(abs(reconstruct(dv, s) - vals[frozenset(s)]) < 1e-9 for s in map(frozenset, subsets("wxyz")))
    print("mobius.py: all synthetic verifications pass "
          "(additive->0, redundant->-1, complementary->+1, exact reconstruction on 2^4)")

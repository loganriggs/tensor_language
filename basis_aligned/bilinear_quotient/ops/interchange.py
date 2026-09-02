"""Interchange-commutation statistic for task-conditioned groupings (ops lane).

Math review 0110, move #2 (causal abstraction: Geiger et al. 2021/2023;
causal scrubbing 2022): a proposed grouping of terms is a valid
abstraction iff interchange interventions WITHIN a group preserve
behavior while interventions BETWEEN groups do not.  This module turns
that into one frozen statistic, given per-swap behavioral deltas that
the harness measures however it registers (CE, task-mask logit margin,
reader response -- the statistic is agnostic).

    result = commutation(within_deltas, between_deltas, seed=...)

- within_deltas:  list[float], |behavior change| for swaps inside groups
- between_deltas: list[float], |behavior change| for swaps across groups
- separation = between_mean / max(within_mean, eps): >> 1 supports the
  grouping; ~1 kills it.
- p_value: label-permutation test (swap labels shuffled, one-sided on
  separation), exact answer to "could this split arise by chance?".

Advisory instrumentation only; bars/thresholds belong to registrations.
Verified below on synthetic valid and invalid groupings.
"""
import random


def commutation(within_deltas, between_deltas, *, seed=0, permutations=10_000):
    w = [abs(x) for x in within_deltas]
    b = [abs(x) for x in between_deltas]
    if not w or not b:
        raise ValueError("both delta lists must be non-empty")
    eps = 1e-12
    wm = sum(w) / len(w)
    bm = sum(b) / len(b)
    observed = bm / max(wm, eps)
    pooled = w + b
    n_w = len(w)
    rng = random.Random(seed)
    hits = 0
    for _ in range(permutations):
        rng.shuffle(pooled)
        pw = pooled[:n_w]
        pb = pooled[n_w:]
        stat = (sum(pb) / len(pb)) / max(sum(pw) / len(pw), eps)
        if stat >= observed:
            hits += 1
    return {
        "within_mean": wm, "between_mean": bm,
        "separation": observed,
        "p_value": (hits + 1) / (permutations + 1),
        "n_within": n_w, "n_between": len(b),
    }


if __name__ == "__main__":
    rng = random.Random(42)
    # valid grouping: within swaps ~ noise .01, between swaps ~ .5
    valid = commutation([rng.gauss(0, .01) for _ in range(40)],
                        [rng.gauss(.5, .05) for _ in range(40)])
    assert valid["separation"] > 20 and valid["p_value"] < .001, valid
    # invalid grouping: identical distributions
    same = [rng.gauss(.2, .05) for _ in range(80)]
    invalid = commutation(same[:40], same[40:])
    assert .5 < invalid["separation"] < 2 and invalid["p_value"] > .05, invalid
    print("interchange.py: synthetic verifications pass "
          f"(valid sep={valid['separation']:.1f} p={valid['p_value']:.4f}; "
          f"invalid sep={invalid['separation']:.2f} p={invalid['p_value']:.2f})")

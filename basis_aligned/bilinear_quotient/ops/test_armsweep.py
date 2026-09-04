"""Guard for ops/armsweep.py: the saving must not change a single number."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import armsweep


def _stub():
    """Deterministic stand-ins: native depends only on the batch, arm value on both."""
    calls = {"n": 0}

    def native_fn(b):
        calls["n"] += 1
        return 100.0 + b

    def arm_fn(b, a):
        calls["n"] += 1
        return 100.0 + b - (a * 0.5 + b * 0.01)

    return native_fn, arm_fn, calls


def test_damages_are_identical_to_the_naive_loop():
    batches, arms = [1, 2, 3, 4, 5], [0, 1, 2, 3]
    nf, af, _ = _stub()
    naive = {a: [(nf(b), af(b, a)) for b in batches] for a in arms}
    nf2, af2, _ = _stub()
    fast = armsweep.sweep_arms(batches, nf2, af2, arms)
    for a in arms:
        assert [n - x for n, x in naive[a]] == [n - x for n, x in fast[a]], a


def test_the_native_forward_is_evaluated_once_per_batch():
    batches, arms = [1, 2, 3, 4, 5], list(range(36))
    nf, af, calls = _stub()
    armsweep.sweep_arms(batches, nf, af, arms)
    assert calls["n"] == len(batches) * (1 + len(arms))
    assert armsweep.forward_count(5, 36) == calls["n"]


def test_the_saving_is_what_the_docstring_claims():
    n_b, n_a = 5, 36
    fast, naive = armsweep.forward_count(n_b, n_a), armsweep.naive_forward_count(n_b, n_a)
    assert naive == 360 and fast == 185
    saved = 1.0 - fast / naive
    assert 0.48 <= saved <= 0.49, saved      # the 48.6% the EFFICIENCY_LOG row states


def test_on_forward_counts_every_forward():
    seen = []
    armsweep.sweep_arms([1, 2], lambda b: b, lambda b, a: b + a, [0, 1],
                        on_forward=lambda: seen.append(1))
    assert len(seen) == 2 * (1 + 2)

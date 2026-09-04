"""armsweep -- compute one NATIVE forward per batch instead of one per arm. (ops lane, additive, opt-in.)

Written 2026-09-04 08:58Z after measuring where the hour's GPU time went. Five component-sweep rungs
(SS2862, SS2864, SS2866, SS2868, SS2869, SS2871) each ran an inner loop of the shape

    for component in all_36:
        for batch in batches:
            native  = run(model, batch)                      # <-- does NOT depend on `component`
            ablated = run(model, batch, writer=component, ...)
            damage  = margins(native) - margins(ablated)

so the native forward was recomputed 36 times for a value that is identical every time. Exactly half of
every sweep's forwards were natives and only 1/36 of those were needed: **48.6% of the forwards in those
rungs bought nothing.** Measured cost this hour: 33,552 forwards across five sweeps, ~16,300 of them
redundant, ~444 of the hour's 576 GPU-seconds.

`sweep_arms` computes the native margins once per batch and reuses them across every arm. It is a drop-in
for that loop shape and nothing adopts it automatically -- the sweep rungs already run are cited by ledger
sections and their bytes are unchanged.

Equivalence is not asserted, it is tested: `test_armsweep.py` drives both paths with a deterministic stub
and checks the damages agree exactly AND that the forward count drops from n_arms*(1+1) to 1+n_arms per
batch. The numerical equivalence on the real model is registered as an instrument predicate in the first
rung that imports this.
"""
from __future__ import annotations


def sweep_arms(batches, native_fn, arm_fn, arms, on_forward=None):
    """Return {arm: [per-batch native, per-batch arm]} with ONE native evaluation per batch.

    batches    : iterable of whatever `native_fn`/`arm_fn` accept (evaluated once, so pass a list)
    native_fn  : batch -> native value (the arm-independent forward)
    arm_fn     : (batch, arm) -> value under that arm
    arms       : iterable of arm identifiers
    on_forward : optional callable invoked once per model forward, for price accounting

    The caller combines native and arm values however its metric requires; this helper only guarantees
    that the native side is evaluated once per batch rather than once per (batch, arm).
    """
    batches = list(batches)
    arms = list(arms)
    natives = []
    for b in batches:
        natives.append(native_fn(b))
        if on_forward:
            on_forward()
    out = {}
    for a in arms:
        rows = []
        for b, nat in zip(batches, natives):
            rows.append((nat, arm_fn(b, a)))
            if on_forward:
                on_forward()
        out[a] = rows
    return out


def forward_count(n_batches, n_arms):
    """Forwards `sweep_arms` performs, against the naive loop's n_batches * n_arms * 2."""
    return n_batches * (1 + n_arms)


def naive_forward_count(n_batches, n_arms):
    return n_batches * n_arms * 2

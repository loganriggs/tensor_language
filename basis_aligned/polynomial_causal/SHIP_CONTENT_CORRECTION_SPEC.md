# Whole-ship content correction specification

**Status: contingent phase two; do not execute before
`SHIP_CONTENT_ORACLE_SPEC.md` licenses a site.** The optimizer-free screen was
inserted because end-to-end CE training can learn compensation rather than the
missing original computation. If licensed, this draft must inherit the oracle's
covariance/energy-matched null and singleton-site restriction before execution.

## Question

The held-out attention x MLP0-2 x deep factorial assigns `0.7277 / 0.8727`
global ship nats and `1.0776 / 1.1755` novel/rare nats to MLP0-2. This licenses
an early correction, but the 43-64% interaction burden requires fitting it with
the complete ship live.

Does the frozen clean-model content subspace provide a causally useful,
composable interface for that correction, or can an arbitrary subspace do as
well?

## Frozen arms and price

At each of MLP0, MLP1, and MLP2, add

```text
delta(z) = (z W + b) B^T
```

where `z` is that site's live RMS-normalized input under the complete current
ship. `B` is the first 48 columns of the previously frozen 64-dimensional
content basis. `W,b` are optimized directly for end-to-end token CE, with every
attention and MLP replacement and the existing generic rank-32 MLP2 glue live.

The standalone price counts both `W` and `B`: `2*1152*48 + 48 = 110,640`
parameters. This is within 1% of the existing generic MLP2 glue's `111,744`
parameters. Original-model provenance does not make `B` free. The decisive
control replaces `B` with a seeded random orthonormal 48-dimensional basis and
receives the identical optimizer, input map, bias, and training budget.

The three content sites are trained on 192 FineWeb rows at skip 2000 and selected
on 64 rows at skip 5000. The random control is trained only at that
validation-selected site. Discovery (skip 7000) and held-out (skip 11000) each
contain 480 untouched rows. Cell definitions exactly match the completed
factorial.

This is distinct from the failed generic glue cascade: the current ship already
contains the successful generic MLP2 rank-32 glue, and previous added generic
MLP1/MLP0 slots worsened CE. The new arm restricts the output API and asks for
incremental gain on top of that baseline.

## Preregistered decisions

1. Held-out global CE gain over the current ship is at least `0.05` nats.
2. Held-out novel/rare gain is at least `0.11755` nats, 10% of that cell's
   factorial full-ship excess.
3. Copy and novel/frequent CE do not regress by more than `0.01` nats each.
4. The content arm beats the price-matched random-basis arm by at least `0.02`
   held-out global nats.
5. The validation-selected content site is also the best content site on both
   discovery and held-out splits.

Passing all gates promotes the content API to a whole-ship compiler interface,
but not yet to a causal program: output-slice and intervention-family transport
remain separate admission tests. Failure of gate 4 rejects the semantic content
interpretation even if a generic correction improves CE. Failure of gates 1 or 2
demotes this interface and redirects work to an internal MLP0-2 error basis or a
different surrogate grammar rather than adding more glue.

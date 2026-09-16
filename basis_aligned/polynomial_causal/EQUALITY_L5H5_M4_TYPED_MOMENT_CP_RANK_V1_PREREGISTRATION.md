# Equality L5H5 M4 typed-context moment CP-rank V1

The all-position empirical moment differs strongly between natural and code.
This test asks whether an already established structural context slot—use as an
equality-edge query or key—produces a compact metric that transfers, rather
than merely reflecting panel composition.

Using the frozen equality support with no new selection, collect normalized
MLP4-input moments separately at query positions, key positions, their union,
and positions incident to no equality edge. Compute the same reader-weighted
CP-rank lower bounds and Gaussian-energy audit on 192 natural and 192 code
documents.

Predictions fixed before execution:

1. Every cell has at least 128 positions, mean state-square in `[.98,1.02]`,
   PSD error at most `2e-4`, and paired unfolding-energy discrepancy at most
   `2e-4`.
2. Natural/code rank-256 lower bounds differ by at most `.10` for both query
   and key positions.
3. Natural query and key rank-256 lower bounds are each at most `.35`.
4. Both natural typed bounds improve at least 20% on the all-position natural
   bound `.4233676211`.
5. Natural query and key Gaussian-energy errors are each at most `.25`, and
   the worse typed rank-256 bound is at least `.05` below the natural
   off-support bound.

Failure of prediction 1 is invalid. Other failures are valid typed-context
nulls. Passing does not produce a CP executor; it licenses a frozen typed fit
whose exact removal vector must still transfer.

Price: one checkpoint load; 192 natural and 192 code partial prefixes; eight
moment/spectral reports; no gradients, fits, parameter updates, or new text.

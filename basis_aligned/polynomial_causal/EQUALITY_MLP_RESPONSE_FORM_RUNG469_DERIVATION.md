# Derivation: a factorization-independent MLP response object

## Question forced by rung 468

Rung 467 found exact product terms inside MLP8, MLP9, and MLP12 whose removal reproduced the equality-context
correction on held-out code documents. Rung 468 froze those product indices and moved them to natural text. Their
direction remained similar, but their magnitude, matched-control separation, multi-MLP membership, and interaction
did not transfer.

Changing the number of selected terms would only tune coordinates in one particular factorization of the MLP. It
would not tell us whether code and natural text use the same computation. The next object therefore represents the
complete quadratic function of an MLP and separates three possible causes of the transfer failure:

1. the equality signal puts the MLP into different input states;
2. later layers use the MLP output differently; or
3. the input change and downstream use have different correlations.

## Exact computation and dimensions

For one of MLP8, MLP9, or MLP12, let the input be `x` in 1,152-dimensional residual-stream coordinates. Ignoring
the bias, which cancels when two states are compared, the exact MLP output is

`F(x) = Down[(Left x) * (Right x)]`.

`Left` and `Right` each map 1,152 numbers to 4,608 product activations. `*` is elementwise multiplication. `Down`
maps the 4,608 products back to a 1,152-dimensional residual-stream write.

Let `x_s` be the input when the equality matcher is present, `x_0` the input when it is absent, and `g` the gradient
of the selected-token cross-entropy loss with respect to this MLP's output. Define

`Q(g) = sym(Left^T diag(Down^T g) Right)`

and

`S(x_s,x_0) = sym((x_s+x_0)(x_s-x_0)^T)`,

where `sym(A)=(A+A^T)/2`. Both `Q` and `S` are 1,152 by 1,152 matrices. Then

`g^T [F(x_s)-F(x_0)] = <Q(g), S(x_s,x_0)>_F`,

where `<A,B>_F` means multiply corresponding matrix entries and sum them. For the effect of removing the matcher-
induced MLP write, the sign is reversed.

This equality is exact for the MLP's local first-order loss response. It is not an approximation caused by reducing
rank or selecting product terms. The later network remains nonlinear, so the local first-order response must still
be checked against the actual CE change from replacing the complete MLP product vector.

## Why this removes the product-coordinate ambiguity

The native implementation writes the quadratic function as 4,608 products, but another exact factorization could
rescale, swap, or reorganize factors while computing the same output function. `Q(g)` is obtained after contracting
the full MLP with the downstream gradient. Any exact refactorization that leaves `F` unchanged leaves `Q(g)`
unchanged. It is therefore a property of the function and its downstream use, not of the arbitrary product index.

`S` is the corresponding state-side object. It records which quadratic input directions the equality signal changes.
The scalar response is their matrix inner product.

## Mean effect and the coupling term

Across tokens in one context cell,

`E[<Q,S>] = <E[Q], E[S]> + E[<Q-E[Q], S-E[S]>]`.

The first term combines the average downstream reader with the average equality-induced state change. The second is
the covariance between them: whether a particular input change tends to occur exactly when later layers care about
that change. This gives three distinct, measurable failure modes.

- Similar `Q` but different `S`: the same downstream computation is available, but code and natural text drive it
  differently.
- Different `Q` but similar `S`: the equality signal reaches similar MLP states, but later layers use those writes
  differently.
- Similar average `Q` and `S` but different covariance: the computation depends on finer context coupling that the
  average forms erase.

## What would count as progress

A form cosine is only a diagnostic screen. The stronger test freezes the average `Q` on code, combines it with `S`
measured on held-out code or natural inputs, and predicts the four context-specific local responses. A single scale
from code discovery then maps that response to the actual CE effect of removing the complete MLP contribution. The
prediction must work on held-out code and natural text and beat simply copying the four code effects.

Passing that test would give a factorization-independent, cross-register predictive description of what the three
MLPs read and how downstream computation uses it. It would still not be a compressed replacement. Failure would
choose the independent state-level route: model the context-conditioned downstream causal equivalence directly,
rather than selecting more product coordinates.

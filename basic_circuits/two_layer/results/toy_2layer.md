# Toy 2-layer bilinear: 4-wise ANDs in superposition

`python toy_2layer.py`. **Two stacked bilinear layers, no residual / no const**,
computing 4-wise ANDs:

    h = (W1a x) ⊙ (W1b x)     layer 1: degree 2, width h1
    g = (W2a h) ⊙ (W2b h)     layer 2: degree 4 in x, width h2
    logit = Wo g + bo

**Inputs**: 5-hot over m=6 (`C(6,5)=6` inputs, each = "which feature is off").
**Outputs**: the `C(6,4)=15` four-ANDs. 5-hot is the minimum that lets outputs
co-activate (two 4-subsets sharing 3 features have union 5), so each input has
`C(5,4)=5` co-active ANDs — output superposition.

## What hidden widths are needed?

Sweep of `(h1, h2)`, best of 10 seeds:

![h-sweep grid](./fig_toy2L_hsweep.png)

| | h2=1 | 2 | 3 | **4** | 5 | 6 |
|---|---|---|---|---|---|---|
| h1=1 | 78 | 78 | 78 | 78 | 78 | 78 |
| h1=2 | 79 | 92 | 97 | 97 | 97 | 97 |
| **h1=3** | 79 | 89 | 97 | **100** | 100 | 100 |
| h1=4 | 79 | 92 | 97 | 100 | 100 | 100 |

The minimal widths that compute all 15 four-ANDs are **h1=3, h2=4** — and it's a
hard corner: `h1=2` caps at 97%, `h2=3` caps at 97%, `h1=1` at 78%. So both layers
need a floor (`h1≥3` *and* `h2≥4`), and **15 outputs are computed by layer widths
of just 3 and 4** — genuine superposition (3, 4 ≪ 15). The folded exact degree-4
tensor reproduces the forward pass to 2e-12.

## Logit ladder

![logit ladder](./fig_toy2L_ladder.png)

Two case classes only — **positive** (`off ∉ target`, all 4 active) and **neg
shares-3** (`off ∈ target`, 3 of 4 active). There are *no* easy negatives (with
one feature off, any 4-subset keeps ≥3 active), so every negative is the hardest
kind. Separated at threshold 0 (100%).

## Ladder decomposition — the genuine 4-AND term barely matters

Split each logit into `bias + signal + interference`, where **signal** is the
genuine top-degree monomial contribution (`24·T4[a,b,c,d]·x_a x_b x_c x_d`, the
coefficient of the actual 4-AND), and **interference** is everything else (the
lower-degree / cross structure the fold produces):

![ladder decomposition](./fig_toy2L_ladder_decomp.png)

| variant | accuracy |
|---|---|
| full | **100%** |
| no interference (bias + signal) | **62%** |
| no signal (bias + interference) | **97%** |

The surprise (and the point): **removing the genuine 4-AND signal barely hurts
(→97%), while keeping only the signal collapses to 62%.** The detector is almost
entirely in the *interference* — the distributed lower-degree structure — not in
the top-degree monomial that "is" the AND. This is the degree-4, 2-layer version
of the project's recurring theme (the computation isn't where a naive top-order
readout looks), and it's even more extreme here than at one layer. It is amplified
by the small input set: on 5-hot-of-6 a 4-AND is true iff the off-feature lies in
the complementary pair, so the target is naturally expressible through low-order
structure — which is exactly what the network uses.

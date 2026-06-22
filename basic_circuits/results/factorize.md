# `factorize.py` — anatomy of a single bilinear Universal-AND layer

Regenerate with `python factorize.py` (writes the four PNGs in this folder and
prints the interference-factorization numbers quoted below). All results are for
**seed 2** of the single-layer Universal-AND net: `m = 32` boolean features,
3-hot inputs embedded in `d0 = 16` dims, `n_hidden = 64`, `T = C(32,2) = 496`
AND targets, sigmoid + BCE head.

Each output `t = AND(x_a, x_b)` is, after pulling the embedding into the weights,
an **exact quadratic form** in the 32-dim feature basis:

```
logit_t(x) = xᵀ Qf_t x + bias_t ,    x ∈ {0,1}³² (3-hot)
```

`factorize.py` takes those forms (`Qf`, computed in `pullback.py`) and asks: what
is each `Qf_t` actually made of, and where does the cross-target structure come
from?

---

## 1. The quadratic form of one target

![Qf for one target](./fig1_qform.png)

A single target's `Qf_t` (here `AND(x3, x14)`). Three things coexist in this one
32×32 matrix:

- **Signal** — the one bright `(a,b)` cross term (`Qf[3,14]`). On 3-hot inputs the
  relevant contribution is `2·Qf[a,b]`, ≈ **+45** for this target (mean across all
  targets is **+38**). This is the genuine `x_a x_b` detector.
- **Diagonal** — because `x_i² = x_i` on booleans, the diagonal acts as a *linear*
  term, not a quadratic one. It is uniformly negative: ≈ **−4** at the target's own
  two indices, ≈ **−16** everywhere else. This is structured **inhibition**.
- **Everything else** — dense off-diagonal **interference**, mean ≈ 0, large
  variance, with no obvious per-target structure by eye.

## 2. The three-part decomposition

![Decomposition of Qf](./fig2_decomp.png)

The same matrix split additively into `Qf = signal ⊕ diagonal inhibition ⊕
interference`:

| piece | what it is | scale |
|---|---|---|
| **signal** | one `(a,b)` cross-term | `2·Qf[a,b]` ≈ +38 (mean) |
| **diagonal inhibition** | effective linear term (`x_i²=x_i`) | own ≈ −4, others ≈ −16 |
| **interference** | all other off-diagonal entries | mean 0, ≈ ±5 per `Qf` entry (≈ ±11 as `2·Qf`) |

The striking part: by squared mass, **interference carries most of the matrix**,
yet the network classifies correctly. So interference is *tolerated*, not
cancelled — which the next figure explains.

## 3. The logit ladder — why interference is survivable

![Logit ladder](./fig3_ladder.png)

Distribution of the logit over inputs, split by case:

- **positive** (the AND is true) — pushed to the right of 0 by the +38 signal.
- **hardest negative** (input shares **1** of the target's 2 indices) — gets the
  signal's partial activation but is dragged negative by the inhibition.
- **easy negative** (shares **0** indices) — driven far negative by the full
  inhibition ladder.

The diagonal inhibition is what *separates the three populations* along the logit
axis; the ±20-ish interference is just noise riding on top. Because the sigmoid
(dashed) saturates hard away from 0, that interference noise almost never flips a
decision. This is the core "the computation lives below a thresholded readout"
point: a linear/ridge probe sees the messy interface and reports ~no signal,
while the thresholded output is essentially exact.

## 4. Interference factorization — where the off-diagonal comes from

Stack the off-diagonal coefficients into a `T×T` matrix `C = 2·Qf[:, i<j]`
(`C[t,t]` is target `t`'s own signal); zero the diagonal to get the pure
interference matrix `X`, and take its SVD.

![Interference spectrum](./fig4_factor.png)

Printed results (seed 2):

```
C = Wo @ Mcross exact?                       True       # exact neuron factorization
participation ratio (effective rank):        5.5
  top- 1 components:  41.5% of interference variance
  top- 2 components:  45.0%
  top-16 components:  71.8%
  top-64 components:  98.1%        # knee at n_hidden = 64 (red line)
cross-target corr after removing top- 1:    -0.010  (was +0.41)
signal energy inside top-64 interference subspace:  13.2%
```

What this says:

- **One dominant mode.** Despite full numerical rank (496), the interference has
  effective rank ≈ 5.5 and a **single component explains 41.5%** of it. The
  singular-value spectrum has a sharp knee at exactly `n_hidden = 64` (the weights
  can only produce a rank-64 family of forms; the tail past 64 is the
  diagonal-zeroing residual).
- **The shared mode IS the cross-target correlation.** Different targets' inter­
  ference patterns correlate at +0.41; **removing just the top-1 mode collapses
  that to ≈ 0**. So the "shared structure" across targets is essentially this one
  direction. (Per `CONTEXT.md`, that mode is identified as **embedding crosstalk**:
  its pair-side vector tracks the embedding Gram overlaps `G = EᵀE` and its
  target-side vector is ~constant — i.e. the dominant interference is *inherited
  from the non-orthogonal embedding `E`, not learned*.)
- **Signal lives outside the interference subspace.** Only **13.2%** of the signal
  energy falls inside even the top-64 interference directions. Signal and inter­
  ference are nearly orthogonal — which is exactly why the layer can tolerate the
  interference instead of having to cancel it.

---

## Takeaways

1. Each Universal-AND output is `signal ⊕ inhibitory diagonal ⊕ rank-≈1
   embedding-geometry crosstalk ⊕ small residual`.
2. The diagonal is **linear inhibition** (boolean idempotence), and it — not
   cancellation — is the network's learned defense; it separates positives from
   the hardest negatives on the logit axis.
3. The dominant interference is **not learned**: it is geometric crosstalk
   inherited from the non-orthogonal embedding, captured by a single SVD mode.
4. A linear/ridge probe measures the interference-laden interface (high FVU); the
   thresholded readout measures the actual, nearly-exact computation. The whole
   point of the decomposition is to recover the latter.

> Follow-on (`hollow.py`): pushing the diagonal into an explicit linear vector
> ("hollowing") is exact on booleans but does **not** make the signal edge surface
> in the per-target eigenspectrum — the off-diagonal embedding crosstalk analyzed
> here still dominates. See `../CONTEXT.md` open threads #1/#5.

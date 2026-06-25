# Is there math to find the secrets from Q?

Short answer: **yes, and it says the secrets are exactly the hard part.** Recovering
a secret = finding an `x ∈ {±1}ⁿ` that maximises `xᵀQx`. That is **Boolean quadratic
optimization** — the **Ising ground-state / MAX-CUT** problem — which is **NP-hard in
general**. So your instinct is right: you're doing a *search*, not reading it off.
But the structure is worth knowing, and it explains exactly *why* it's hard here.

## Your "equal to each other, or overcome by other stuff" intuition = local stability

Flipping bit `i` changes the score by `Δᵢ = −4 xᵢ (Qx)ᵢ + 4 Qᵢᵢ`. Since `xᵢ² = 1`, the
diagonal `Qᵢᵢ` is a constant offset and drops out, so `x` is a **local maximum** iff

    xᵢ = sign( (Q̃ x)ᵢ )   for every i ,     Q̃ = Q with zero diagonal.

That's exactly what you were sensing: **each bit must agree with the weighted "vote"
of all the other bits** (`(Q̃x)ᵢ = Σ_{j≠i} Qᵢⱼ xⱼ`). A bit's own preference can be
*overcome* by the couplings to the rest — and a secret is a configuration where every
bit is simultaneously consistent with everyone else. (This is also why one bit flip
swings the logit a lot: it changes its vote to all `n−1` others at once.)

## It's a Hopfield network — and we're over capacity

`x ← sign(Q̃ x)` is the **Hopfield associative-memory update**, and a secret being a
fixed point of it is exactly "the pattern is stored." So `Q` is a memory matrix and the
secrets are its stored patterns. Hopfield theory then tells us the catch:

- a Hopfield net reliably stores only about **0.14·n** patterns;
- here `n = 64`, so capacity ≈ **9**, but we store **16** — **over capacity**.

`finding_secrets_math.py` shows the consequence directly:

| | stable fixed points / 16 | retrieved from 2000 random starts |
|---|---|---|
| trained `Q` | 7 / 16 | 7 / 16 |
| ideal Hebbian `Q = Σ sₛsₛᵀ` | 4 / 16 | — |

Even the *textbook* Hebbian memory keeps only 4/16 secrets stable — because 16 > 9.
Over capacity the stored patterns stop being clean attractors: the energy landscape
fills with **spurious minima** (mixtures of patterns) and the basins go leaky — exactly
the rough landscape we measured (≈1800 local maxima, a spurious string scoring higher
than the secrets, 1-bit flips escaping >50% of the time). So the hardness isn't a
training artifact; it's that the task asks the quadratic form to store more patterns
than a quadratic form comfortably can.

## What the matrix *can* and *cannot* give you

- **Can:** the secret *subspace*. The top-16 eigenspace of `Q` captures **96%** of each
  secret's energy — so `Q` does tell you the 16-dimensional subspace the secrets live
  in.
- **Cannot:** the individual `±1` vectors. A symmetric matrix is only defined up to
  rotation within a degenerate/secret subspace, so `eigh` returns orthogonal **mixtures**
  of the secrets, not the secrets themselves. (We verified eigh-sign recovers a planted
  secret *only* when the secrets are mutually orthogonal **and** have distinct eigenvalues
  — random `±1` secrets break both; see `eig_conditions.py`.) Pinning down the actual
  `±1` vectors inside the subspace is the combinatorial part.

## How much does knowing the subspace save? (≈16 bits, not 2×)

Knowing the secrets live in the top-`d=16` eigen-subspace lets you search only the
`±1` strings expressible as `sign(B c)` for `c ∈ ℝ¹⁶` (an arrangement of 64 hyperplanes
in 16-D): `Σ_{k<16} C(63,k) ≈ 2⁴⁷` strings, versus `2⁶³` naively (up to the ± symmetry).

    naive          ≈ 2⁶³  candidate strings
    subspace-only  ≈ 2⁴⁷  candidate strings
    saving         ≈ 2¹⁶  ≈ 53,000×   (≈ d = 16 bits, NOT a factor of 2)

So the subspace buys you ~16 bits. **Rotation ambiguity is not a factor of 2** — it is
the *entire residual* `~2⁴⁷` search *inside* the subspace: the matrix tells you the
16-D subspace exactly but gives **zero** information about orientation within it (a
continuous `O(16)` of equally-valid orthogonal bases), so you still have to find which
`±1` vectors in it are the secrets.

## Where the leverage is

1. **Search, but smart.** Do the `±1` search / SDP-rounding / Hopfield retrieval *inside*
   the top-16 eigen-subspace (`~2⁴⁷`, not `2⁶³`), and use that there are exactly 16
   (keep the best-16 distinct local maxima).
2. **Go to higher order — this is the real win, and why 2 layers help.** A symmetric
   *matrix* `Σ sₛsₛᵀ` is rotation-ambiguous, so even with the perfect subspace you face
   the `2⁴⁷` residual search. A symmetric *4th-order tensor* `Σ sₛ⊗sₛ⊗sₛ⊗sₛ` is **not**
   ambiguous: tensor / CP decomposition (order ≥ 3) is essentially **unique**
   (Kruskal/Jennrich) and recovers **non-orthogonal** components **algebraically — no
   search at all**, in polynomial time. A **2-bilinear-layer** organism folds to exactly
   such a degree-4 tensor, so in principle it is *more* extractable: the higher order
   collapses the `2⁴⁷` orientation search to a direct read-out, via the de-mixing /
   sparse-pursuit machinery in `../basic_circuits/two_layer/sparse_pursuit.py`. The
   caveat (from the `two_layer/` toys) is whether a *trained* 2-layer organism stores its
   secrets as a clean 4th-moment tensor or, as we kept seeing, a messy distributed one —
   that's the experiment to run.

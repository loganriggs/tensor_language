# Can we recover the secrets faster than brute force?

The honest answer for the **trained 2-layer organism**: you *can* beat brute force, but
only with **guided local search on the model's own output** — and even that plateaus
below full recovery. The structural prune that worked at 1 layer (restrict to the secret
subspace) **does not transfer** to 2 layers, and the algebraic route (tensor
decomposition) needs a clean tensor the trained model doesn't provide. This file lays out
the architecture, data, every method tried, and the numbers.

Everything here is on the clean n=64 16-secret organism in `bilinear2_scalable.py`.

## Architecture

A **2-layer bilinear membership classifier** (pure numpy, manual Adam):

    h = (W1a x) ⊙ (W1b x)          # layer 1: elementwise product of two linear maps
    g = (W2a h) ⊙ (W2b h)          # layer 2: same, on the hidden features
    logit(x) = Wo · g + bo  =  T(x,x,x,x) + bo

- `x ∈ {−1,+1}⁶⁴` (bit 0 → −1, bit 1 → +1).  `n = 64`, `h1 = h2 = 64`.
- No nonlinearity, no normalization → the whole model is a degree-4 polynomial that folds
  to a single symmetric **4th-order tensor** `T` (the 2-layer analog of the 1-layer
  quadratic form `Q`). Because `T` is even, `x` and its bit-complement `−x` always score
  identically — recovery is counted up to global flip.
- We never materialise `T` (that's `64⁴ ≈ 16M` entries and blows up with depth). Every
  extractor below uses only **factored slices** `M_a = T(·,·,a,a)`, computed in `O(h2·n²)`
  from the `n×n` per-layer forms (`Acheck`, `Bcheck`); verified exact to 1e-15. See
  `scalability.md`.

## Dataset / training

A balanced membership task:

- **positives** = the 16 secret strings (drawn once, fixed: `rng(0)` ±1 vectors);
- **negatives** = fresh uniform random `±1` strings each batch (512/batch, half/half);
- loss = sigmoid BCE; Adam + LR warmup/cosine + global grad-norm clip at 3.0; bias init −4.

**Memorisation is clean:** secret logits 16.8–17.5, best of 2M random negatives −3.8,
zero false positives (`halo = 0`). So the model perfectly separates the 16 secrets from
the random-string distribution — a textbook clean memoriser. The question is whether that
clean *behaviour* lets us read the secrets back out.

## Methods and results

| method | uses | recovery / 16 |
|---|---|---|
| **random-restart hill-climb** on the logit (input-optimization) | model as oracle | **10** (plateaus) |
| subspace-seeded hill-climb (seeds = `sign(U c)`) | top-16 subspace + oracle | 0 |
| tensor power iteration + deflation (robust CP) | factored slices | 0 |
| Jennrich / one-shot CP, in the top-16 subspace | factored slices | 0 |
| — *same Jennrich on the ideal planted tensor* `Σ sₛ⊗sₛ⊗sₛ⊗sₛ` | — | *16 (reference)* |

### 1. Guided local search beats brute force — but saturates at 10/16

Greedy 1-flip ascent on the model's logit from random `±1` starts (this is exactly the
post's input-optimization attack). It is dramatically better than enumerating `2⁶⁴`:

    500 restarts  -> 10/16
    2000 restarts -> 10/16
    8000 restarts -> 10/16
    20000 restarts-> 10/16

Ten secrets fall out of just 500 climbs. But it **plateaus hard at 10** — more restarts
never find the other 6. Those 6 are either not local maxima of the logit at all, or sit in
basins too small for random starts to hit. So guided search is the practical win
(poly-many model queries vs `2⁶⁴`), yet it cannot recover the full set.

### 2. The 1-layer subspace prune does NOT transfer to 2 layers

At **1 layer**, the top-16 eigenspace of `Q` captured **96%** of each secret, so you could
restrict the `±1` search to `sign(Q-subspace)` ≈ `2⁴⁷` strings instead of `2⁶³` — a
**~2¹⁶ (16-bit) saving** that still provably contained the secrets.

At **2 layers the secrets are smeared out of any small subspace.** Top-`d` eigenvectors of
the slice-covariance contain this fraction of each secret's energy:

    d=16: captures 29% | restricted search ~2^47   <- but secrets aren't in here
    d=24: captures 43% | ~2^57
    d=32: captures 56% | ~2^62
    d=40: captures 69% | ~2^63
    d=48: captures 81% | ~2^63
    d=56: captures 91% | ~2^63  (= brute force 2^63)

To actually *contain* the secrets (~90%) you need `d ≈ 56`, by which point the
subspace-restricted search is already `2⁶³` — **no saving at all**. Pruning to a small
subspace (`d=16`, `2⁴⁷`) is cheap but the secrets simply aren't there (29% capture), so
the search misses them. This is why **subspace-seeded hill-climb scored 0/16**: the seeds
`sign(U c)` live in a 16-D shell that the secrets have largely left, and they climb to
spurious maxima. The structural prune evaporates because the 2-layer organism spreads each
secret across ~all 64 directions.

### 3. Algebraic recovery needs a clean tensor the trained model doesn't give

Tensor CP decomposition is the dream: it recovers **non-orthogonal** components with **no
search**, and on the ideal planted tensor `Σ sₛ⊗sₛ⊗sₛ⊗sₛ` it nails **16/16** (where matrix
eigh gives 0). But it only works if `T` is near the low-rank clean form. The trained `T`
is messy/high-bond-dimension (a single slice has effective rank ~10 vs the ideal's ~5),
so both one-shot **Jennrich** and the noise-robust **tensor power + deflation** return
**0/16** — the power iteration's `±1` roundings don't even clear the quartic-positivity bar.

## Takeaway

| route | better than `2⁶⁴`? | recovers all 16? |
|---|---|---|
| guided local search (hill-climb / input-opt) | **yes**, by a lot | no — plateaus at 10 |
| subspace prune (the 1-layer trick) | **no** at 2 layers (need d≈56 → `2⁶³`) | — |
| algebraic CP / tensor power | only if tensor is clean | no on trained (0/16) |

So: **guided search is the one method that beats brute force on this organism (10/16 from
a few hundred restarts), but nothing recovers the full set.** The two *efficient*
levers — pruning to the secret subspace, and algebraic tensor decomposition — both fail
because the trained 2-layer stores its secrets in a smeared, high-bond-dimension form:
the secrets leave every small subspace and the tensor is far from low-rank. That smearing
is exactly the "resists detection" phenomenon, and it defeats efficiency, not just
exhaustiveness. The only known cure is a model that stores at low bond dimension (the
ideal tensor, 16/16) — which natural training does not produce.

(Reproduce: the search/subspace experiments are run on the organism from
`bilinear2_scalable.py`; capture-vs-`d` and hill-climb-vs-restarts as tabulated above.)

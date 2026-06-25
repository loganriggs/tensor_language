# MLP secret extraction with bilinear layers

A bilinear take on the LessWrong post *"Naturally learned behaviors in deep MLPs
resist detection..."* (emanuelr, Jun 2026). We train a **membership classifier**
to memorise 16 secret `n`-bit strings (high logit on the secrets, low on random
strings) and ask: **can we recover the secrets by reading the weights?** Strings are
in `{−1,+1}ⁿ` (`n = 64`). Pure numpy, manual Adam.

## Why bilinear: the model folds to one quadratic form

A 1-layer bilinear membership classifier has a single output:

    logit(x) = Σ_k Wo[k] (W1[k]·x)(W2[k]·x) + bo  =  xᵀ Q x + bo ,
    Q = Σ_k Wo[k] · sym(W1[k] ⊗ W2[k])   (one n×n matrix)

So — answering the obvious first question — **folding the whole 1-layer model back
gives a single `n×n` matrix `Q`** (one output → one quadratic form). The memorised
secrets are exactly the `x ∈ {±1}ⁿ` that maximise `xᵀQx`. (Note `xᵀQx` is *even*, so
`x` and `−x` score the same: a pure bilinear model can't distinguish a secret from
its bit-complement — we count recovery up to global flip.)

The natural hope: read `Q` by eigendecomposition and the top eigenvectors *are* the
secrets. **This does not work** — and the reasons are instructive.

## Finding 1 — memorisation is perfect, extraction fails (`bilinear1.py`)

Balanced training (positives = the 16 secrets, negatives = random strings). The model
memorises cleanly: secret logits ≈ 13, best random logit ≈ 0.8, and no false positive
in 2M random samples. Yet every weight-reading / search method recovers almost nothing:

| method | recovery / 16 |
|---|---|
| eigh of `Q`, sign of top-16 eigenvectors | **0** |
| eigh + hill-climb | 1 |
| neuron seeds (`sign W1[k]`, `sign W2[k]`) | **0** |
| neuron seeds + hill-climb | 1 |
| input-optimization (2000 random-restart hill-climbs) | 5 |

Diagnostics: the eigenvalue spectrum has **no gap at 16** (smooth decay), secrets have
best `|cos|` only ≈ 0.53 with any eigenvector and ≈ 0.33 with any `W1/W2` row (≈ chance
0.12). The secrets are stored **fully distributed / in superposition**, not in neurons
or eigenvectors.

## Finding 2 — the secrets aren't even the global optima

Why is search so weak? The quadratic landscape over `{±1}⁶⁴` is **rough**:

- secret basins are leaky — a single bit flip climbs back only ~44% of the time;
- 2000 restarts find **~1800 distinct local maxima**;
- and crucially, the **highest** local maximum found scores `xᵀQx ≈ 21` vs the secrets'
  `≈ 16.4` — **the secrets are not the global maxima**. The model separates the secrets
  from *random* strings (the training distribution) but not from *optimised* ones, so
  input-optimization surfaces spurious high strings before the real secrets.

## Finding 3 — when *would* eigh work? (`eig_conditions.py`)

eigh of a planted `Q = Σ aₛ sₛsₛᵀ` recovers a secret as `sign(eigenvector)` **only if
two conditions both hold**:

| secrets | weights | eig-sign recovery |
|---|---|---|
| orthogonal | distinct | **16 / 16** |
| orthogonal | equal (degenerate) | 0 / 16 |
| random `±1` (`\|cos\|≈0.1`) | distinct | 1 / 16 |
| random `±1` | equal | 0 / 16 |

eigh isolates a vector only when it is **an orthogonal direction with a distinct
eigenvalue**. Random secrets break orthogonality (eigh returns rotated *mixtures* of
them); equal weights make the secret subspace degenerate (eigh returns an arbitrary
basis of it). A trained organism breaks both. This is the **same orthogonal-mixing
pathology** as the 2-layer pairing problem in `../basic_circuits/two_layer/`
(`decomp_exact.py`, `sparse_pursuit.py`): an orthogonal basis can't return non-orthogonal
factors — you'd need a non-orthogonal / combinatorial `±1` pursuit inside the secret
subspace.

## Takeaway

Even a **1-layer, balanced, bilinear** organism — no obfuscation, just memorisation —
stores 16 secrets so they resist eigh, neuron-reading and hill-climbing, and aren't even
the optima of its own quadratic form. This is a sharper version of the post's result
(their 1-layer was extractable; bilinear's even, rough quadratic landscape is harder).

## Files
- `bilinear1.py` — train the 1-layer organism, fold to `Q`, run all extractors + the
  landscape diagnostics. Saves `organism_1lay_balanced.npz`.
- `eig_conditions.py` — the idealized "when does eigh recover planted secrets" table.
- `small_organism.py` → `small_organism.md` — a tiny human-readable organism (n=12,
  4 secrets, h=6) with **all weights printed in full** (L=W1, R=W2, D=Wo, bias, and
  the folded Q) plus the secret strings, so you can try to spot the secrets by eye
  (you can't: L/R-row and eigenvector reads both score 0/4, even though it's a clean
  memoriser — top-8 strings are exactly the 4 secrets + their complements). Also saves
  imshow figures of L, R, D, Q and of Q's top eigen-directions vs the secrets.
- `tiny_organism.py` → `tiny_organism.md` — even smaller (n=5, 2^5=32 strings, 4
  secrets): brute-forces **all 32** so the **bit-complement symmetry** is explicit —
  `xᵀQx` is even, so every string and its all-bits-flipped complement get the identical
  logit (each secret and its complement tied at the top of the list). Full weights +
  imshow figures included.

## 2 layers: tensor methods (`bilinear2.py`)

A 2-bilinear-layer organism folds to a **4th-order tensor** `T` (not a matrix). Unlike a
matrix, a symmetric tensor's CP decomposition is essentially unique, so in principle it
breaks the rotation ambiguity. Results (n=32, 8 secrets, clean memoriser):
- **subspace analog of `eigh(Q)`:** the top-d eigenvectors of the **mode-1 covariance**
  `M = T₍₁₎T₍₁₎ᵀ` (contract `T` with itself over 3 of 4 modes → `n×n`) span the secret
  subspace — but here capture only **44%** of each secret (vs 96% for 1 layer: the
  trained 2-layer spreads them *more*).
- **ideal planted tensor `Σ sₛ⊗sₛ⊗sₛ⊗sₛ`:** Jennrich/CP recovers **8/8** even though the
  secrets are non-orthogonal (matrix eigh → 0) — the in-principle win.
- **trained organism:** Jennrich → **0/8**. The trained tensor is messy/distributed (as
  in the `two_layer/` toys), not the clean 4th-moment — so 2 layers are *harder* to
  extract in practice, not easier.

## Can we recover faster than brute force? (`efficient_recovery.md`)

On the clean n=64 2-layer organism, beating `2⁶⁴` is possible but limited:
- **guided local search** (random-restart hill-climb on the logit = input-optimization)
  gets **10/16** from a few hundred restarts — far better than brute force, but plateaus
  at 10 (the other 6 secrets aren't reachable local maxima);
- the **1-layer subspace prune does not transfer**: top-16 of the slice-covariance
  contains only **29%** of each secret (vs 96% at 1 layer); you need `d≈56` to contain
  them, by which point the restricted search is `2⁶³` ≈ brute force;
- **algebraic CP** (Jennrich / tensor power + deflation) → **0/16** on the trained tensor
  (needs low bond dim; recovers the ideal planted tensor 16/16).

The efficient levers (subspace pruning, tensor decomposition) both fail because the
trained 2-layer smears each secret across ~all directions — see `scalability.md`.

## Next steps
- the **hard-negative** regime (near-miss negatives) — expected to be even harder;
- a **non-orthogonal `±1` pursuit** in the top-16 eigen-subspace (the proper extractor);
- **2 bilinear layers** → a degree-4 form (folds to a quartic tensor, not a matrix),
  the same object the `two_layer/` toys analyse.
